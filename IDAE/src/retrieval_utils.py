"""Retrieval baselines used by the FSMIC token/semantic ablations.

The implementation follows ``LLM_Comment_Generation/preprocess.py``:

* token: tokenize code, count common *unique* tokens, and rank descending;
* semantic: encode code with ``st-codesearch-distilroberta-base`` and rank by
  cosine similarity, matching the default ``sentence_transformers.util``
  semantic-search score function used by FSMIC.

The token implementation uses an inverted index, which is equivalent to the
pairwise common-token count but avoids materialising a test-by-train matrix.
The semantic implementation encodes the training corpus once and searches it
in batches with FAISS.
"""

from __future__ import annotations

import os
import re
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


_TOKEN_RE = re.compile(r"[a-z]+|[A-Z][a-z]*|[0-9]+|[^\w\s]+")


def tokenize_code(code_str) -> List[str]:
    """Match FSMIC's source-code tokenization exactly."""
    code_str = str(code_str)
    code_str = re.sub(r"//.*|/\*[\s\S]*?\*/", "", code_str)
    code_str = re.sub(r"[\.\,\;\:\(\)\{\}\[\]]", " ", code_str)
    code_str = re.sub(r"\s+", " ", code_str)
    tokens = _TOKEN_RE.findall(code_str)
    for i in range(len(tokens)):
        if i > 0 and tokens[i - 1].islower() and tokens[i].isupper():
            tokens[i] = tokens[i].lower()
    return tokens


def _record(example):
    return {
        "id": example["id"],
        "raw_code": example["raw_code"],
        "comment": example["comment"],
        "label": example["label"],
    }


def select_retrieval_pool(
    train_list: Sequence[dict],
    max_size: int = 0,
    seed: int = 42,
) -> List[dict]:
    """Select a deterministic, label-stratified approximate retrieval pool.

    ``max_size=0`` keeps the complete corpus.  When a positive limit is used,
    sampling is stratified by intent label and the selected records are put
    back in their original corpus order.  The same seed and pool must be used
    for token and semantic ablations to keep the comparison fair.
    """
    max_size = int(max_size)
    if max_size <= 0 or max_size >= len(train_list):
        return list(train_list)

    label_to_indices = {}
    for index, example in enumerate(train_list):
        label_to_indices.setdefault(example.get("label"), []).append(index)

    rng = random.Random(seed)
    selected = set()
    # Proportional allocation, with at least one item for each non-empty label.
    labels = list(label_to_indices)
    quotas = {}
    remaining = max_size
    if max_size < len(labels):
        # Extremely small debug pools cannot contain every label.  Keep the
        # largest label groups in a deterministic order rather than exceeding
        # the requested pool size.
        keep_labels = set(
            sorted(labels, key=lambda item: (-len(label_to_indices[item]), str(item)))[:max_size]
        )
    else:
        keep_labels = set(labels)
    for label in labels:
        quota = 1 if label in keep_labels else 0
        if max_size >= len(labels):
            quota = max(1, int(round(max_size * len(label_to_indices[label]) / len(train_list))))
        quota = min(quota, len(label_to_indices[label]))
        quotas[label] = quota
        remaining -= quota
    while remaining > 0:
        candidates = [
            label for label in labels
            if quotas[label] < len(label_to_indices[label])
        ]
        if not candidates:
            break
        label = max(
            candidates,
            key=lambda item: len(label_to_indices[item]) - quotas[item],
        )
        quotas[label] += 1
        remaining -= 1
    while remaining < 0:
        candidates = [label for label in labels if quotas[label] > 1]
        if not candidates:
            break
        label = max(candidates, key=lambda item: quotas[item])
        quotas[label] -= 1
        remaining += 1

    for label in labels:
        indices = list(label_to_indices[label])
        rng.shuffle(indices)
        selected.update(indices[:quotas[label]])
    return [example for index, example in enumerate(train_list) if index in selected]


class TokenRetriever:
    """Exact common-token retrieval using sparse matrix multiplication.

    A Python inverted-index loop is still too slow for the 1.23M-example
    FunCom+TLCodesum corpus: common tokens make each query touch most of the
    corpus.  ``CountVectorizer`` builds the same binary token matrix as the
    FSMIC set intersection, and SciPy performs query-by-corpus multiplication
    in compiled code.
    """

    def __init__(
        self,
        train_list: Sequence[dict],
        batch_size: int = 32,
        cache_prefix: Optional[str] = None,
    ):
        self.train_list = train_list
        self.batch_size = max(1, int(batch_size))

        def analyzer(code):
            # FSMIC intersects sets, so repeated occurrences count once.
            return set(tokenize_code(code))

        self.vectorizer = CountVectorizer(
            analyzer=analyzer,
            lowercase=False,
            binary=True,
            dtype=np.int16,
        )
        cache_matrix = Path(f"{cache_prefix}.npz") if cache_prefix else None
        cache_vocab = Path(f"{cache_prefix}.vocab.json") if cache_prefix else None
        cache_ok = (
            cache_matrix is not None
            and cache_vocab is not None
            and cache_matrix.is_file()
            and cache_vocab.is_file()
        )
        if cache_ok:
            print(f"Loading cached token matrix from {cache_matrix}...", flush=True)
            with cache_vocab.open("r", encoding="utf-8") as stream:
                self.vectorizer.vocabulary_ = {
                    str(key): int(value) for key, value in json.load(stream).items()
                }
            self.vectorizer.fixed_vocabulary_ = True
            self.train_matrix = load_npz(cache_matrix).tocsr()
            if self.train_matrix.shape[0] != len(train_list):
                print("Token cache size mismatch; rebuilding it.", flush=True)
                cache_ok = False
        if not cache_ok:
            print("Building sparse token matrix for FSMIC token retrieval...", flush=True)
            self.train_matrix = self.vectorizer.fit_transform(
                example["raw_code"]
                for example in tqdm(train_list, desc="Tokenizing train")
            ).tocsr()
            if cache_matrix is not None and cache_vocab is not None:
                cache_matrix.parent.mkdir(parents=True, exist_ok=True)
                save_npz(cache_matrix, self.train_matrix)
                with cache_vocab.open("w", encoding="utf-8") as stream:
                    json.dump(self.vectorizer.vocabulary_, stream)
        self.train_matrix.sort_indices()
        print(
            f"Token matrix shape={self.train_matrix.shape}, "
            f"nonzeros={self.train_matrix.nnz:,}",
            flush=True,
        )

    @staticmethod
    def _top_nonzero(indices, scores, limit):
        """Return the first ``limit`` items in FSMIC's score/index order."""
        if limit <= 0 or len(indices) == 0:
            return []
        indices = np.asarray(indices, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.int32)
        if len(indices) <= limit:
            order = np.lexsort((indices, -scores))
            return indices[order].tolist()

        # Selecting the threshold avoids sorting millions of tied score-1
        # candidates.  Among equal scores, FSMIC's stable sort uses corpus
        # order, i.e. the smallest original indices first.
        threshold = np.partition(scores, -limit)[-limit]
        high_mask = scores > threshold
        high_indices = indices[high_mask]
        high_scores = scores[high_mask]
        high_order = np.lexsort((high_indices, -high_scores))
        selected = high_indices[high_order].tolist()

        remaining = limit - len(selected)
        if remaining > 0:
            equal_indices = np.sort(indices[scores == threshold])
            selected.extend(equal_indices[:remaining].tolist())
        return selected

    def retrieve(
        self,
        query_list: Sequence[dict],
        top_k: int = 10,
        exclude_self: bool = False,
    ) -> Dict[str, list]:
        result = {}
        if top_k <= 0:
            return {query["id"]: [] for query in query_list}

        for start in tqdm(
            range(0, len(query_list), self.batch_size),
            desc="Token retrieval",
        ):
            end = min(start + self.batch_size, len(query_list))
            query_matrix = self.vectorizer.transform(
                query["raw_code"] for query in query_list[start:end]
            )
            score_matrix = query_matrix.dot(self.train_matrix.T).tocsr()
            for offset in range(end - start):
                query_index = start + offset
                row_start = score_matrix.indptr[offset]
                row_end = score_matrix.indptr[offset + 1]
                indices = score_matrix.indices[row_start:row_end]
                scores = score_matrix.data[row_start:row_end]

                # Request one extra result when the query is a training item;
                # the query itself must be removed after ranking.
                limit = top_k + 1 if exclude_self else top_k
                ranked = self._top_nonzero(indices, scores, limit)
                if exclude_self:
                    query_id = query_list[query_index]["id"]
                    ranked = [
                        i for i in ranked
                        if self.train_list[i]["id"] != query_id
                    ]

                # Complete with zero-score items in corpus order, exactly like
                # FSMIC's stable sort over all training examples.
                selected = set(ranked)
                if len(ranked) < top_k:
                    for train_index in range(len(self.train_list)):
                        if (
                            exclude_self
                            and self.train_list[train_index]["id"]
                            == query_list[query_index]["id"]
                        ):
                            continue
                        if train_index in selected:
                            continue
                        ranked.append(train_index)
                        selected.add(train_index)
                        if len(ranked) >= top_k:
                            break

                query = query_list[query_index]
                result[query["id"]] = [
                    _record(self.train_list[i]) for i in ranked[:top_k]
                ]
        return result


class SemanticRetriever:
    """FSMIC's CodeSearch sentence-embedding retrieval baseline."""

    def __init__(
        self,
        train_list: Sequence[dict],
        model_name: str,
        batch_size: int = 64,
        device: Optional[str] = None,
        cache_path: Optional[str] = None,
        index_mode: Optional[str] = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "semantic retrieval requires sentence-transformers; install it "
                "or use IDAE_ABLATION=token"
            ) from exc

        self.train_list = train_list
        self.batch_size = batch_size
        self.device = device or os.getenv(
            "IDAE_RETRIEVAL_DEVICE",
            "cuda" if __import__("torch").cuda.is_available() else "cpu",
        )
        print(f"Loading FSMIC semantic retriever from {model_name} on {self.device}...", flush=True)
        self.model = SentenceTransformer(model_name, device=self.device)
        train_codes = [example["raw_code"] for example in train_list]
        embedding_cache = Path(cache_path) if cache_path else None
        if embedding_cache is not None and embedding_cache.is_file():
            print(f"Loading cached semantic embeddings from {embedding_cache}...", flush=True)
            train_embeddings = np.load(embedding_cache, mmap_mode="r")
            if train_embeddings.shape[0] != len(train_list):
                print("Semantic cache size mismatch; rebuilding it.", flush=True)
                train_embeddings = None
        else:
            train_embeddings = None
        if train_embeddings is None:
            print("Encoding semantic training corpus...", flush=True)
            train_embeddings = self.model.encode(
                train_codes,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                # FSMIC's util.semantic_search defaults to cosine similarity.
                # Normalization makes cosine equivalent to FAISS inner product.
                normalize_embeddings=True,
            )
            train_embeddings = np.asarray(train_embeddings, dtype=np.float32)
            if embedding_cache is not None:
                embedding_cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(embedding_cache, train_embeddings)
        train_embeddings = np.asarray(train_embeddings, dtype=np.float32)

        self.index_mode = (
            index_mode or os.getenv("IDAE_SEMANTIC_INDEX", "flat")
        ).strip().lower()
        cpu_index = faiss.IndexFlatIP(train_embeddings.shape[1])
        if self.index_mode == "hnsw":
            m = int(os.getenv("IDAE_SEMANTIC_HNSW_M", "32"))
            self.index = faiss.IndexHNSWFlat(train_embeddings.shape[1], m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = int(os.getenv("IDAE_SEMANTIC_HNSW_EF_CONSTRUCTION", "100"))
            self.index.hnsw.efSearch = int(os.getenv("IDAE_SEMANTIC_HNSW_EF_SEARCH", "64"))
        elif self.index_mode in {"gpu", "gpu_flat"}:
            if not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError("IDAE_SEMANTIC_INDEX=gpu requires faiss-gpu")
            self._gpu_resources = faiss.StandardGpuResources()
            gpu_id = int(os.getenv("IDAE_SEMANTIC_FAISS_DEVICE", "0"))
            self.index = faiss.index_cpu_to_gpu(self._gpu_resources, gpu_id, cpu_index)
        elif self.index_mode == "flat":
            self.index = cpu_index
        else:
            raise ValueError(
                "IDAE_SEMANTIC_INDEX must be flat, gpu, gpu_flat, or hnsw"
            )
        self.index.add(train_embeddings)

    def retrieve(
        self,
        query_list: Sequence[dict],
        top_k: int = 10,
        exclude_self: bool = False,
    ) -> Dict[str, list]:
        query_codes = [example["raw_code"] for example in query_list]
        query_embeddings = self.model.encode(
            query_codes,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        search_k = min(len(self.train_list), top_k + (1 if exclude_self else 0))
        result = {}
        for start in tqdm(
            range(0, len(query_list), self.batch_size),
            desc="Semantic retrieval",
        ):
            end = min(start + self.batch_size, len(query_list))
            _, indices = self.index.search(query_embeddings[start:end], search_k)
            for offset, retrieved_indices in enumerate(indices):
                query_index = start + offset
                selected = []
                for train_index in retrieved_indices:
                    if train_index < 0:
                        continue
                    if (
                        exclude_self
                        and self.train_list[train_index]["id"]
                        == query_list[query_index]["id"]
                    ):
                        continue
                    selected.append(train_index)
                    if len(selected) >= top_k:
                        break
                result[query_list[query_index]["id"]] = [
                    _record(self.train_list[i]) for i in selected
                ]
        return result
