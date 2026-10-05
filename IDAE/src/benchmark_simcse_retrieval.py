#!/usr/bin/env python3
"""Benchmark the *main* SimCSE retriever against the full training corpus.

This follows build_nearest_examples_test.py: use the local Sup-SimCSE model,
256-token fixed padding, normalized 768-dimensional code vectors, and a
separate GPU FAISS IndexFlatIP for each intention label. No retrieval pool or
precomputed embedding cache is used. Unlike the old script, inputs are sent
to the GPU in batches to avoid allocating the entire corpus token tensor at
once; this does not change the vectors or the FAISS retrieval algorithm.

Only timing/size metadata is written; no model, index or exemplar JSON files
are modified. Set CUDA_VISIBLE_DEVICES to one *idle* physical GPU per run.
"""

import argparse
import json
import os
import random
import resource
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from build_nearest_examples_test import MAX_LEN, SIMCSE_MODEL_NAME, SimCSEEncoder


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def _sync():
    torch.cuda.synchronize()


def _time_sync(call):
    _sync()
    start = time.perf_counter()
    result = call()
    _sync()
    return result, time.perf_counter() - start


def _read_split(dataset, split, limit=0):
    codes, labels = [], []
    with (DATA_ROOT / f"{dataset}.{split}").open(encoding="utf-8") as stream:
        for row in stream:
            example = json.loads(row)
            codes.append(example["raw_code"])
            labels.append(example["label"])
            if limit and len(codes) >= limit:
                break
    return codes, labels


def _encode_codes(model, tokenizer, codes, batch_size, max_len, title):
    vectors = []
    token_seconds = 0.0
    forward_seconds = 0.0
    for start in tqdm(range(0, len(codes), batch_size), desc=title, mininterval=15):
        batch = codes[start:start + batch_size]
        t0 = time.perf_counter()
        encoded = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        token_seconds += time.perf_counter() - t0

        def forward():
            encoded_gpu = {key: value.to("cuda") for key, value in encoded.items()}
            with torch.no_grad():
                output = model(
                    input_ids=encoded_gpu["input_ids"],
                    attention_mask=encoded_gpu["attention_mask"],
                )
            return output.cpu().numpy().astype(np.float32, copy=False)

        vector_batch, duration = _time_sync(forward)
        forward_seconds += duration
        vectors.append(vector_batch)
    return np.concatenate(vectors, axis=0), {
        "tokenization_seconds": round(token_seconds, 4),
        "forward_and_transfer_seconds": round(forward_seconds, 4),
        "total_seconds": round(token_seconds + forward_seconds, 4),
    }


def _gpu_memory_mib():
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 1024**2, 2)


def _rss_gib():
    # Linux ru_maxrss is in KiB, unlike macOS where it is in bytes.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)


def _index_size_bytes(index):
    """Measure CPU-serializable FAISS index bytes; no disk writes required."""
    cpu_index = faiss.index_gpu_to_cpu(index)
    return int(faiss.serialize_index(cpu_index).nbytes)


def _percentile(values, percent):
    return round(float(np.percentile(np.asarray(values), percent)), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("tlcodesum", "funcom"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--single-query-samples", type=int, default=1000)
    parser.add_argument(
        "--smoke-train-limit", type=int, default=0,
        help="For a quick smoke test only; omit for full-corpus measurements",
    )
    args = parser.parse_args()
    if any(value <= 0 for value in (args.batch_size, args.query_batch_size, args.max_len, args.top_k)):
        parser.error("batch sizes, max length, and top-k must be positive")
    if args.single_query_samples < 0 or args.smoke_train_limit < 0:
        parser.error("sample counts must be non-negative")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("Set CUDA_VISIBLE_DEVICES to exactly one free GPU")
    if args.output.exists():
        parser.error(f"Won't overwrite existing report: {args.output}")

    print(
        f"Dataset={args.dataset}, physical CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')}, gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    started = time.perf_counter()
    (train_codes, train_labels), train_read_sec = _time_sync(
        lambda: _read_split(args.dataset, "train", args.smoke_train_limit)
    )
    (test_codes, test_labels), test_read_sec = _time_sync(
        lambda: _read_split(args.dataset, "test")
    )
    print(f"Loaded {len(train_codes):,} train and {len(test_codes):,} test records", flush=True)

    def _load_model():
        tokenizer = AutoTokenizer.from_pretrained(SIMCSE_MODEL_NAME)
        model = SimCSEEncoder(SIMCSE_MODEL_NAME).to("cuda").eval()
        return model, tokenizer

    (model, tokenizer), load_sec = _time_sync(_load_model)
    model_gpu_mib = _gpu_memory_mib()
    print(f"Model loaded in {load_sec:.2f}s; encoding full training corpus", flush=True)
    vectors, train_encoding = _encode_codes(
        model, tokenizer, train_codes, args.batch_size, args.max_len, "Train encoding"
    )
    dim = int(vectors.shape[1])
    print(f"Corpus encoded in {train_encoding['total_seconds']:.2f}s; building GPU indices", flush=True)

    labels_to_indices = defaultdict(list)
    for i, label in enumerate(train_labels):
        labels_to_indices[label].append(i)
    gpu_resources = faiss.StandardGpuResources()
    indices = {}
    label_gpu_bytes = {}
    build_start = time.perf_counter()
    for label, row_ids in labels_to_indices.items():
        cpu_empty = faiss.IndexFlatIP(dim)
        gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_empty)
        data = np.ascontiguousarray(vectors[row_ids], dtype=np.float32)
        _sync()
        gpu_index.add(data)
        _sync()
        indices[label] = gpu_index
        label_gpu_bytes[str(label)] = int(gpu_index.ntotal * dim * 4)
        print(f"Indexed label {label}: {gpu_index.ntotal:,} vectors", flush=True)
    index_build_sec = time.perf_counter() - build_start
    index_gpu_mib = _gpu_memory_mib()
    print(f"FAISS index built in {index_build_sec:.2f}s; encoding test queries", flush=True)

    queries, query_encoding = _encode_codes(
        model, tokenizer, test_codes, args.batch_size, args.max_len, "Test encoding"
    )
    label_test_indices = defaultdict(list)
    for i, label in enumerate(test_labels):
        label_test_indices[label].append(i)
    n_missing_label = sum(len(rows) for label, rows in label_test_indices.items() if label not in indices)

    search_start = time.perf_counter()
    searched = 0
    for label, rows in tqdm(label_test_indices.items(), desc="Batched FAISS search"):
        if label not in indices:
            continue
        for start in range(0, len(rows), args.query_batch_size):
            batch_rows = rows[start:start + args.query_batch_size]
            selected = np.ascontiguousarray(queries[batch_rows])
            _sync()
            _, neighbors = indices[label].search(selected, args.top_k)
            _sync()
            searched += len(neighbors)
    batched_search_sec = time.perf_counter() - search_start
    print(f"Searched {searched:,} test queries in {batched_search_sec:.2f}s", flush=True)

    rng = random.Random(42)
    available = [i for i, label in enumerate(test_labels) if label in indices]
    sampled_rows = rng.sample(available, k=min(args.single_query_samples, len(available)))
    # Fixed number of warm-up calls, not included in latency statistics.
    for row in sampled_rows[:min(20, len(sampled_rows))]:
        indices[test_labels[row]].search(np.ascontiguousarray(queries[row:row + 1]), args.top_k)
    _sync()
    single_latency_ms = []
    for row in tqdm(sampled_rows, desc="Single-query FAISS latency", mininterval=15):
        query = np.ascontiguousarray(queries[row:row + 1])
        _, seconds = _time_sync(lambda: indices[test_labels[row]].search(query, args.top_k))
        single_latency_ms.append(seconds * 1000)

    # Index serialization is for reporting size only, not part of retrieval
    # runtime. Copying it back to CPU creates a temporary extra allocation.
    runtime_peak_rss_gib = _rss_gib()
    serialized_index_bytes = 0
    print("Measuring serialized index size in RAM (no disk write)", flush=True)
    for label, index in indices.items():
        serialized_index_bytes += _index_size_bytes(index)

    result = {
        "dataset": args.dataset,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "smoke_train_limit": args.smoke_train_limit,
        "full_corpus": args.smoke_train_limit == 0,
        "device": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "index_type": "per-label GPU FAISS IndexFlatIP (exact)",
        "model": str(SIMCSE_MODEL_NAME),
        "candidate_count": len(train_codes),
        "test_query_count": len(test_codes),
        "vector_dim": dim,
        "max_len": args.max_len,
        "encode_batch_size": args.batch_size,
        "search_batch_size": args.query_batch_size,
        "top_k": args.top_k,
        "train_read_seconds": round(train_read_sec, 4),
        "test_read_seconds": round(test_read_sec, 4),
        "model_load_seconds": round(load_sec, 4),
        "train_encoding": train_encoding,
        "faiss_gpu_index_build_seconds": round(index_build_sec, 4),
        "candidate_embeddings_bytes": int(vectors.nbytes),
        "faiss_gpu_flat_vector_bytes": sum(label_gpu_bytes.values()),
        "faiss_serialized_cpu_index_bytes": serialized_index_bytes,
        "index_label_bytes": label_gpu_bytes,
        "gpu_memory_after_model_load_mib": model_gpu_mib,
        "gpu_memory_after_index_build_mib": index_gpu_mib,
        "torch_peak_allocated_mib_excludes_faiss": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "process_peak_rss_gib_before_size_serialization": runtime_peak_rss_gib,
        "query_encoding": query_encoding,
        "batched_faiss_search_seconds": round(batched_search_sec, 4),
        "batched_faiss_query_count": searched,
        "test_queries_without_matching_label": n_missing_label,
        "single_query_faiss_samples": len(single_latency_ms),
        "single_query_faiss_p50_ms": _percentile(single_latency_ms, 50) if single_latency_ms else None,
        "single_query_faiss_p95_ms": _percentile(single_latency_ms, 95) if single_latency_ms else None,
        "query_encode_plus_batched_search_seconds": round(
            query_encoding["total_seconds"] + batched_search_sec, 4
        ),
        "total_benchmark_seconds": round(time.perf_counter() - started, 4),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Saved benchmark: {args.output}", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
