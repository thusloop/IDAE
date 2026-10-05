#!/usr/bin/env python3
"""Measure decoder-input truncation by exemplar count.

The decoder training/evaluation code builds one encoder input as::

    target_code + intention + ``M`` rendered nearest examples

and then calls the CodeT5 tokenizer with ``truncation=True, max_length=512``.
This script reproduces that exact input construction and reports, for every
requested M:

* the fraction of records whose complete encoder input exceeds MAX_LEN;
* the fraction of records for which target code loses at least one token;
* the fraction for which exemplar position 1, 2, ..., M loses at least one
  token; and
* token-level loss ratios for the same components.

Only source data and existing nearest-example JSON files are read. The script
creates a new report and never writes or rewrites an index/exemplar file.
"""

import argparse
from bisect import bisect_left, bisect_right
import json
import math
import os
from collections import Counter
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from ablation_utils import get_ablation_tag, nearest_examples_filename
from model_paths import CODET5P_220M_MODEL


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
DEFAULT_MAX_LEN = 512


def _csv_ints(raw):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            value = int(item)
            if value < 0:
                raise ValueError("M/num-examples values must be non-negative")
            values.append(value)
    if not values:
        raise ValueError("at least one integer is required")
    return sorted(set(values))


def _csv_names(raw, allowed):
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one name is required")
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"unsupported value(s): {', '.join(unknown)}")
    return values


def _dataset_files(dataset, split):
    path = DATA_ROOT / f"{dataset}.{split}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_jsonl(path, limit=0):
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def _nearest_path(dataset, ablation, split="train"):
    """Match codet5_dec.py's dataset-specific fallback resolution."""
    dataset_path = Path(nearest_examples_filename(split, "codebert", dataset, ablation))
    if dataset_path.exists():
        return dataset_path
    all_path = Path(nearest_examples_filename(split, "codebert", "all", ablation))
    if all_path.exists():
        return all_path
    raise FileNotFoundError(
        f"No nearest-example file found for dataset={dataset}, split={split}, "
        f"ablation={ablation}; checked {dataset_path} and {all_path}"
    )


def _nearest_split_for_runtime(split):
    """Match the current training/evaluation scripts' lookup behavior.

    codet5_dec.py loads the train-neighbor map once and reuses it for its
    validation subset; evaluate.py loads the test-neighbor map. This matters
    because validation IDs normally are not keys in the train map.
    """
    return "test" if split == "test" else "train"


def _load_nearest(path):
    # Existing files are large, but loading is read-only and allows exact ID
    # lookup. The report itself is small and is written to a new path.
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _render_input(record, nearest, num_examples):
    code = record["raw_code"]
    label = record["label"]
    pieces = [code, "\nIntention:\n", label, "\nSimilar Examples:\n"]
    spans = {"target_code": (0, len(code))}
    cursor = sum(len(piece) for piece in pieces)
    selected = list(nearest[:num_examples])
    for position, item in enumerate(selected, start=1):
        rendered = (
            f"<code>{item['raw_code']}</code> => "
            f"<comment>{item['comment']}</comment>\n"
        )
        start = cursor
        pieces.append(rendered)
        cursor += len(rendered)
        spans[f"example_{position}"] = (start, cursor)
    return "".join(pieces), spans, len(selected)


def _new_component():
    return {
        "records_with_tokens": 0,
        "records_truncated": 0,
        "total_tokens": 0,
        "tokens_truncated": 0,
    }


def _update_component(component, total, truncated):
    component["records_with_tokens"] += int(total > 0)
    component["records_truncated"] += int(truncated > 0)
    component["total_tokens"] += total
    component["tokens_truncated"] += truncated


def _finalize_component(component, record_count):
    denominator_records = max(1, record_count)
    denominator_tokens = max(1, component["total_tokens"])
    return {
        "records_with_tokens": component["records_with_tokens"],
        "records_truncated": component["records_truncated"],
        "record_truncation_ratio": round(
            component["records_truncated"] / denominator_records, 8
        ),
        "total_tokens": component["total_tokens"],
        "tokens_truncated": component["tokens_truncated"],
        "token_truncation_ratio": round(
            component["tokens_truncated"] / denominator_tokens, 8
        ),
    }


def _percentiles(values):
    if not values:
        return {}
    values = sorted(values)

    def percentile(q):
        if len(values) == 1:
            return values[0]
        rank = (len(values) - 1) * q
        low = math.floor(rank)
        high = math.ceil(rank)
        if low == high:
            return values[low]
        return values[low] + (values[high] - values[low]) * (rank - low)

    return {
        "p50": round(percentile(0.50), 4),
        "p95": round(percentile(0.95), 4),
        "p99": round(percentile(0.99), 4),
        "max": values[-1],
    }


def _component_token_counts(offsets, spans, kept_start, kept_end):
    """Count the same overlapping tokens as the original per-span scan.

    Byte-level tokenizers can give several tokens the same Unicode-character
    offset. Bisect handles these duplicates without decoding or estimating
    component lengths. Fall back to the original scan for non-monotonic
    offsets, rather than assuming every tokenizer is monotonic.
    """
    starts = [start for start, _ in offsets]
    ends = [end for _, end in offsets]
    monotonic = all(
        starts[i] >= starts[i - 1] and ends[i] >= ends[i - 1]
        for i in range(1, len(offsets))
    )
    counts = {}
    for name, (char_start, char_end) in spans.items():
        if char_end <= char_start:
            counts[name] = (0, 0)
            continue
        if monotonic:
            first = bisect_right(ends, char_start)
            last = bisect_left(starts, char_end)
            total = max(0, last - first)
            retained = max(0, min(last, kept_end) - max(first, kept_start))
            counts[name] = (total, total - retained)
        else:
            positions = [
                i for i, (start, end) in enumerate(offsets)
                if end > char_start and start < char_end
            ]
            counts[name] = (
                len(positions),
                sum(i < kept_start or i >= kept_end for i in positions),
            )
    return counts


def _measure_records(records, nearest_map, tokenizer, max_len, num_examples, batch_size):
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    content_limit = max_len - special_count
    if content_limit <= 0:
        raise ValueError("max_len must exceed tokenizer special-token count")

    target = _new_component()
    examples = {f"example_{i}": _new_component() for i in range(1, num_examples + 1)}
    input_lengths = []
    overlength_count = 0
    missing_nearest_count = 0
    selected_count = Counter()

    # Fast tokenizers return exact character offsets. Processing in batches
    # keeps the run practical on the 1.2M-record FunCom training split.
    for start in tqdm(
        range(0, len(records), batch_size),
        desc=f"M={num_examples}",
        mininterval=10,
    ):
        batch = records[start : start + batch_size]
        texts = []
        metadata = []
        for record in batch:
            nearest = nearest_map.get(str(record["id"]), [])
            if num_examples and not nearest:
                missing_nearest_count += 1
            text, spans, selected = _render_input(record, nearest, num_examples)
            texts.append(text)
            metadata.append(spans)
            selected_count[selected] += 1

        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        for token_ids, offsets, spans in zip(
            encoded["input_ids"], encoded["offset_mapping"], metadata
        ):
            total_length = len(token_ids) + special_count
            input_lengths.append(total_length)
            overlength_count += int(total_length > max_len)

            # codet5_dec.py uses the tokenizer default (right truncation).
            # Keep this explicit so the report remains correct if a tokenizer
            # with left truncation is selected later.
            if tokenizer.truncation_side == "left":
                kept_start = max(0, len(token_ids) - content_limit)
                kept_end = len(token_ids)
            else:
                kept_start = 0
                kept_end = min(len(token_ids), content_limit)

            counts = _component_token_counts(offsets, spans, kept_start, kept_end)
            for name, (total, truncated) in counts.items():
                if name == "target_code":
                    _update_component(target, total, truncated)
                else:
                    _update_component(examples[name], total, truncated)

    record_count = len(records)
    return {
        "record_count": record_count,
        "input_overlength_count": overlength_count,
        "input_overlength_ratio": round(overlength_count / max(1, record_count), 8),
        "input_token_length": _percentiles(input_lengths),
        "target_code": _finalize_component(target, record_count),
        "examples": {
            name: _finalize_component(component, record_count)
            for name, component in examples.items()
        },
        "selected_example_count_distribution": {
            str(key): value for key, value in sorted(selected_count.items())
        },
        "records_missing_nearest_examples": missing_nearest_count,
    }


def _text_report(result):
    lines = [
        "CodeT5p-220m encoder-input truncation report",
        f"MAX_LEN={result['max_len']}; special tokens={result['special_tokens_added']}; "
        f"truncation_side={result['truncation_side']}; ablation={result['ablation']}",
        "Record percentage: fraction of inputs losing at least one token from the component.",
        "Each example includes its code, comment, wrapper tags and trailing newline.",
        "Token percentage: removed component tokens / original component tokens (micro-average).",
        "All percentages below use the full split; --limit results are marked as sampled.",
        f"Per-split limit: {result.get('limit', 'unknown')}",
        "",
    ]
    highest_m = max(result["num_examples"])
    header = ["dataset", "split", "N", "M", "input_overlength_%", "target_code_%"]
    header += [f"example_{i}_%" for i in range(1, highest_m + 1)]
    for token_level in (False, True):
        lines += [
            "TOKEN-LEVEL LOSS (%)" if token_level else "RECORD-LEVEL TRUNCATION (%)",
            "\t".join(header),
        ]
        ratio_key = "token_truncation_ratio" if token_level else "record_truncation_ratio"
        for dataset, splits in result["datasets"].items():
            for split, measurements in splits.items():
                for m in result["num_examples"]:
                    data = measurements[str(m)]
                    row = [dataset, split, str(data["record_count"]), str(m)]
                    row += [
                        f"{100 * data['input_overlength_ratio']:.4f}",
                        f"{100 * data['target_code'][ratio_key]:.4f}",
                    ]
                    for i in range(1, highest_m + 1):
                        component = data["examples"].get(f"example_{i}")
                        row.append(f"{100 * component[ratio_key]:.4f}" if component else "-")
                    lines.append("\t".join(row))
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="tlcodesum,funcom")
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--num-examples", default="0,1,2,3,4,5,6")
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ablation", default=os.getenv("IDAE_ABLATION", "main"))
    parser.add_argument(
        "--base-report", type=Path,
        help="Reuse matching measurements from this read-only report; measure only missing M values",
    )
    parser.add_argument(
        "--text-output", type=Path,
        help="New TXT summary path (default: output JSON path with .txt suffix)",
    )
    parser.add_argument(
        "--nearest-policy",
        choices=("runtime", "matching-split"),
        default="runtime",
        help="runtime reproduces codet5_dec.py/evaluate.py; matching-split uses each split's own map",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("truncation_reports") / "codet5p220m.max512.json",
    )
    args = parser.parse_args()
    if args.max_len <= 0 or args.batch_size <= 0 or args.limit < 0:
        parser.error("max-len and batch-size must be positive; limit must be non-negative")
    text_output = args.text_output or args.output.with_suffix(".txt")
    if args.output.resolve() == text_output.resolve():
        parser.error("JSON and TXT report paths must differ")
    for path in (args.output, text_output):
        if path.exists():
            parser.error(f"Refusing to overwrite existing report: {path}")
        if not (Path(__file__).resolve().parent / "truncation_reports") in path.resolve().parents:
            parser.error("Report outputs must be inside this script's truncation_reports directory")

    datasets = _csv_names(args.datasets, ("tlcodesum", "funcom"))
    splits = _csv_names(args.splits, ("train", "valid", "test"))
    num_examples = _csv_ints(args.num_examples)
    ablation = get_ablation_tag() if args.ablation is None else args.ablation
    # Normalize aliases without mutating the caller's environment.
    old_ablation = os.environ.get("IDAE_ABLATION")
    os.environ["IDAE_ABLATION"] = ablation
    try:
        ablation = get_ablation_tag()
    finally:
        if old_ablation is None:
            os.environ.pop("IDAE_ABLATION", None)
        else:
            os.environ["IDAE_ABLATION"] = old_ablation

    tokenizer = AutoTokenizer.from_pretrained(CODET5P_220M_MODEL)
    if not tokenizer.is_fast:
        parser.error("A fast tokenizer with offset mappings is required")
    # We intentionally tokenize without truncation to measure the full input
    # length. Disable the tokenizer's advisory 512-token warning; no model
    # forward pass is performed by this script.
    tokenizer.model_max_length = 10**9
    result = {
        "model": CODET5P_220M_MODEL,
        "max_len": args.max_len,
        "tokenizer": tokenizer.__class__.__name__,
        "special_tokens_added": tokenizer.num_special_tokens_to_add(pair=False),
        "truncation_side": tokenizer.truncation_side,
        "ablation": ablation,
        "nearest_policy": args.nearest_policy,
        "num_examples": num_examples,
        "limit": args.limit,
        "datasets": {},
    }
    base_report = {}
    if args.base_report:
        with args.base_report.open(encoding="utf-8") as stream:
            base_report = json.load(stream)
        for key in (
            "model", "max_len", "tokenizer", "special_tokens_added",
            "truncation_side", "ablation", "nearest_policy",
        ):
            if base_report.get(key) != result[key]:
                parser.error(f"Base-report configuration mismatch for {key}")
        if args.limit or base_report.get("limit", 0):
            parser.error("Base-report reuse is supported only for full-split measurements")
        result["base_report"] = str(args.base_report.resolve())

    nearest_by_dataset = {}
    nearest_paths = {}
    source_snapshot = {}

    def record_source(path):
        path = Path(path).resolve()
        stat = path.stat()
        source_snapshot[str(path)] = {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    if any(m > 0 for m in num_examples):
        for dataset in datasets:
            for split in splits:
                nearest_split = (
                    _nearest_split_for_runtime(split)
                    if args.nearest_policy == "runtime"
                    else split
                )
                nearest_path = _nearest_path(dataset, ablation, nearest_split)
                cache_key = str(nearest_path)
                if cache_key not in nearest_by_dataset:
                    record_source(nearest_path)
                    print(f"Loading nearest examples: {nearest_path}", flush=True)
                    nearest_by_dataset[cache_key] = _load_nearest(nearest_path)
                nearest_paths[(dataset, split)] = (nearest_path, cache_key)

    for dataset in datasets:
        result["datasets"][dataset] = {}
        for split in splits:
            data_path = _dataset_files(dataset, split)
            record_source(data_path)
            records = _load_jsonl(data_path, args.limit)
            nearest_path = None
            if any(m > 0 for m in num_examples):
                nearest_path, cache_key = nearest_paths[(dataset, split)]
                nearest_map = nearest_by_dataset[cache_key]
            else:
                nearest_map = {}

            print(
                f"[{dataset}/{split}] records={len(records):,}, "
                f"nearest={nearest_path if nearest_path else 'none'}",
                flush=True,
            )
            result["datasets"][dataset][split] = {}
            cached_split = base_report.get("datasets", {}).get(dataset, {}).get(split, {})
            if cached_split and cached_split.get("_nearest_examples_file") != str(nearest_path):
                parser.error(f"Base-report nearest-example file mismatch for {dataset}/{split}")
            for m in num_examples:
                cached_measurement = cached_split.get(str(m))
                if cached_measurement:
                    if cached_measurement["record_count"] != len(records):
                        parser.error(f"Base-report record-count mismatch for {dataset}/{split}/M={m}")
                    print(f"Reusing base-report measurement: {dataset}/{split}/M={m}", flush=True)
                    measurement = cached_measurement
                else:
                    measurement = _measure_records(
                        records, nearest_map, tokenizer, args.max_len, m, args.batch_size
                    )
                result["datasets"][dataset][split][str(m)] = measurement
            result["datasets"][dataset][split]["_nearest_examples_file"] = (
                str(nearest_path) if nearest_path else None
            )

    for path, before in source_snapshot.items():
        after = Path(path).stat()
        if after.st_size != before["size_bytes"] or after.st_mtime_ns != before["mtime_ns"]:
            raise RuntimeError(f"Source data changed during measurement: {path}")
    result["source_files_unchanged"] = True
    result["source_files"] = source_snapshot
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {args.output}")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Saved report: {args.output}", flush=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    with text_output.open("x", encoding="utf-8") as stream:
        stream.write(_text_report(result))
    print(f"Saved TXT summary: {text_output}", flush=True)


if __name__ == "__main__":
    main()
