# -*- coding: utf-8 -*-
import json
import torch
import faiss
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch.nn as nn
import os
import random
from ablation_utils import (
    get_ablation_tag,
    nearest_examples_filename,
    use_fsmic_retrieval,
    use_contrastive_encoder,
    use_random_exemplars,
)
from model_paths import CODEBERT_MODEL, CODET5_MODEL, CODESEARCH_MODEL, SIMCSE_MODEL
from retrieval_utils import SemanticRetriever, TokenRetriever, select_retrieval_pool
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
MODEL_TYPE_ENC = "codebert"  # "codebert" or "codet5"
CODEBERT_NAME = CODEBERT_MODEL
CODET5_NAME = CODET5_MODEL
MODEL_NAME = CODEBERT_NAME if MODEL_TYPE_ENC == "codebert" else CODET5_NAME
SIMCSE_MODEL_NAME = SIMCSE_MODEL

DEVICE = "cuda"
BATCH_SIZE = 128
MAX_LEN = 256
NUM_EXAMPLES = 10
CHECKPOINT_DIR = f"./checkpoints_enc"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f"best_enc_{MODEL_TYPE_ENC}.pth")
RANDOM_SEED = 42

# =====================
# Encoder model (unchanged)
# =====================
class CodeEncoderWithProjection(nn.Module):
    def __init__(self, model_name: str, proj_dim: int = 128, model_type: str = "codebert"):
        super().__init__()
        self.model_type = model_type
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim),
        )

    def forward(self, input_ids, attention_mask):
        if self.model_type == "codet5" and hasattr(self.encoder, "encoder"):
            out = self.encoder.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_tokens = out.last_hidden_state[:, 0, :]
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_tokens = out.last_hidden_state[:, 0, :]
        z = self.proj(cls_tokens)
        return nn.functional.normalize(z, p=2, dim=1)


class SimCSEEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            z = out.pooler_output
        else:
            z = out.last_hidden_state[:, 0, :]
        return nn.functional.normalize(z, p=2, dim=1)


def load_trained_codebert(model_path, model_type):
    model = CodeEncoderWithProjection(MODEL_NAME, proj_dim=128, model_type=model_type)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    return model.to(DEVICE).eval()


def _simcse_model_path():
    return SIMCSE_MODEL_NAME


def load_retriever(model_type: str, ablation_tag: str):
    if use_contrastive_encoder(ablation_tag):
        print(f"Loading contrastive retriever from {BEST_MODEL_PATH} ...")
        model = load_trained_codebert(BEST_MODEL_PATH, model_type)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return model, tokenizer
    simcse_path = _simcse_model_path()
    print(f"Loading SimCSE retriever for w/o CL from {simcse_path} ...")
    model = SimCSEEncoder(simcse_path).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(simcse_path)
    return model, tokenizer


# =====================
# Batch encoding helper (unchanged)
# =====================
def encode_batch(model, input_ids, attention_mask):
    with torch.no_grad():
        z = model(input_ids=input_ids, attention_mask=attention_mask)
    return z


# =====================
# Core logic: build label-specific indexes and search only within the same label.
# =====================
def build_nearest_examples_old(train_list, model, tokenizer, top_k=NUM_EXAMPLES):

    # ────────────────────────────────────────────────
    # Step 1: Prepare the data.
    # ────────────────────────────────────────────────
    all_codes = [ex["raw_code"] for ex in train_list]
    all_ids   = [ex["id"] for ex in train_list]
    all_labels = [ex["label"] for ex in train_list]

    print("Tokenizing...")
    enc = tokenizer(
        all_codes, padding=True, truncation=True,
        max_length=MAX_LEN, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # ────────────────────────────────────────────────
    # Step 2: Encode all code vectors.
    # ────────────────────────────────────────────────
    print("Encoding...")
    vecs = []
    for i in tqdm(range(0, len(train_list), BATCH_SIZE)):
        z = encode_batch(
            model,
            input_ids[i:i+BATCH_SIZE],
            attention_mask[i:i+BATCH_SIZE]
        )
        vecs.append(z.cpu())

    vecs = torch.cat(vecs, dim=0).numpy().astype(np.float32)
    dim = vecs.shape[1]

    # ────────────────────────────────────────────────
    # Step 3: Group examples by label.
    # ────────────────────────────────────────────────
    label2idx = {}
    for idx, ex in enumerate(train_list):
        label = ex["label"]
        label2idx.setdefault(label, []).append(idx)

    print("Labels:", {k: len(v) for k, v in label2idx.items()})

    # ────────────────────────────────────────────────
    # Step 4: Build an independent FAISS index for each label.
    # ────────────────────────────────────────────────
    label2index = {}
    gpu_res = faiss.StandardGpuResources()

    for lbl, idx_list in label2idx.items():
        print(f"Building FAISS for label={lbl}, size={len(idx_list)}")
        cpu_index = faiss.IndexFlatIP(dim)
        gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
        sub_vecs = vecs[idx_list]
        gpu_index.add(sub_vecs)
        label2index[lbl] = gpu_index

    # ────────────────────────────────────────────────
    # Step 5: Search and build id2nearest (same-label search only).
    # ────────────────────────────────────────────────
    id2nearest = {}

    print("Searching nearest neighbors by label...")
    for i in tqdm(range(len(train_list))):
        ex = train_list[i]
        label = ex["label"]
        faiss_index = label2index[label]
        idx_list = label2idx[label]   # Global indices for this label.

        # Query vector.
        q = vecs[i].reshape(1, -1)
        D, I = faiss_index.search(q, top_k + 1)

        neighbors = []
        for j_idx in I[0]:
            global_j = idx_list[j_idx]  # Map back to the global index.

            if global_j == i:
                continue  # Exclude the query itself.

            neighbors.append({
                "id": train_list[global_j]["id"],
                "raw_code": train_list[global_j]["raw_code"],
                "comment": train_list[global_j]["comment"],
                "label": train_list[global_j]["label"]
            })

            if len(neighbors) >= top_k:
                break

        id2nearest[all_ids[i]] = neighbors

    return id2nearest

def build_nearest_examples(train_list, model, tokenizer, top_k=NUM_EXAMPLES):

    # ─────────────────────────────────────────────
    # Step 1: Tokenize all code.
    # ─────────────────────────────────────────────
    all_codes = [ex["raw_code"] for ex in train_list]
    all_ids   = [ex["id"] for ex in train_list]
    all_labels = [ex["label"] for ex in train_list]

    print("Tokenizing...")
    enc = tokenizer(
        all_codes, padding=True, truncation=True,
        max_length=MAX_LEN, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # ─────────────────────────────────────────────
    # Step 2: Compute all code vectors (the training set is also the query set).
    # ─────────────────────────────────────────────
    print("Encoding...")
    vecs = []
    for i in tqdm(range(0, len(train_list), BATCH_SIZE)):
        z = encode_batch(
            model,
            input_ids[i:i+BATCH_SIZE],
            attention_mask[i:i+BATCH_SIZE]
        )
        vecs.append(z.cpu())

    vecs = torch.cat(vecs, dim=0).numpy().astype(np.float32)
    dim = vecs.shape[1]

    # ─────────────────────────────────────────────
    # Step 3: Group examples by label.
    # ─────────────────────────────────────────────
    label2idx = {}
    for idx, ex in enumerate(train_list):
        label = ex["label"]
        label2idx.setdefault(label, []).append(idx)

    print("Labels:", {k: len(v) for k, v in label2idx.items()})

    # ─────────────────────────────────────────────
    # Step 4: Build one FAISS index per label.
    # ─────────────────────────────────────────────
    label2faiss = {}
    gpu_res = faiss.StandardGpuResources()

    for lbl, idx_list in label2idx.items():
        cpu_index = faiss.IndexFlatIP(dim)
        gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)

        sub_vecs = vecs[idx_list]
        gpu_index.add(sub_vecs)

        label2faiss[lbl] = gpu_index

    # ─────────────────────────────────────────────
    # Step 5 (main optimization): search all queries for each label at once.
    # ─────────────────────────────────────────────
    id2nearest = {}

    print("Batch searching inside each label...")
    for lbl, train_idx_list in tqdm(label2idx.items(), desc="Labels"):

        faiss_index = label2faiss[lbl]

        # Search all vectors for this label in one call.
        q_vecs = vecs[train_idx_list]  # shape: (num_label_samples, dim)

        # Retrieve top_k+1 so the query itself can be removed.
        D, I = faiss_index.search(q_vecs, top_k + 1)

        # I has shape (num_label_samples, top_k+1).

        for local_i, global_train_i in enumerate(train_idx_list):

            neighbors = []
            for local_j in I[local_i]:
                global_j = train_idx_list[local_j]

                if global_j == global_train_i:
                    continue  # Skip the query itself.

                neighbors.append({
                    "id": train_list[global_j]["id"],
                    "raw_code": train_list[global_j]["raw_code"],
                    "comment": train_list[global_j]["comment"],
                    "label": train_list[global_j]["label"]
                })

                if len(neighbors) >= top_k:
                    break

            id2nearest[train_list[global_train_i]["id"]] = neighbors

    return id2nearest


def _sample_random_neighbors(train_list, candidate_indices, exclude_index, top_k, rng):
    selectable_count = len(candidate_indices) - (1 if exclude_index in candidate_indices else 0)
    if selectable_count <= 0 or top_k <= 0:
        return []

    if selectable_count <= top_k:
        selected_indices = [idx for idx in candidate_indices if idx != exclude_index]
    else:
        selected = set()
        while len(selected) < top_k:
            idx = rng.choice(candidate_indices)
            if idx != exclude_index:
                selected.add(idx)
        selected_indices = list(selected)

    neighbors = []
    for idx in selected_indices:
        neighbors.append({
            "id": train_list[idx]["id"],
            "raw_code": train_list[idx]["raw_code"],
            "comment": train_list[idx]["comment"],
            "label": train_list[idx]["label"],
        })
    return neighbors


def build_random_examples(train_list, top_k=NUM_EXAMPLES, seed=RANDOM_SEED):
    rng = random.Random(seed)
    id2nearest = {}

    label2idx = {}
    for idx, ex in enumerate(train_list):
        label2idx.setdefault(ex["label"], []).append(idx)
    print("Random exemplar mode: same-intent constrained")
    for idx, ex in enumerate(tqdm(train_list, desc="Sampling random neighbors")):
        candidates = label2idx.get(ex["label"], [])
        id2nearest[ex["id"]] = _sample_random_neighbors(train_list, candidates, idx, top_k, rng)
    return id2nearest


# =====================
# Main program
# =====================
if __name__ == "__main__":
    def _dataset_name_from_files(files):
        names = []
        for f in files:
            base = os.path.basename(f)
            if base.startswith("tlcodesum."):
                names.append("tlcodesum")
            elif base.startswith("funcom."):
                names.append("funcom")
        names = sorted(set(names))
        if len(names) == 2:
            return "all"
        if len(names) == 1:
            return names[0]
        return "all"

    def _train_files_from_env():
        files = {
            "tlcodesum": "../../data/tlcodesum.train",
            "funcom": "../../data/funcom.train",
        }
        raw = os.getenv("IDAE_RETRIEVAL_DATASETS", "tlcodesum,funcom")
        names = [name.strip().lower() for name in raw.split(",") if name.strip()]
        if not names or any(name not in files for name in names):
            raise ValueError(
                "IDAE_RETRIEVAL_DATASETS must contain tlcodesum and/or funcom"
            )
        return [files[name] for name in names]

    ablation_tag = get_ablation_tag()
    print(f"Ablation mode: {ablation_tag}")
    model = None
    tokenizer = None
    if not use_random_exemplars(ablation_tag) and not use_fsmic_retrieval(ablation_tag):
        model, tokenizer = load_retriever(MODEL_TYPE_ENC, ablation_tag)

    # The training set consists of two files.
    train_files = _train_files_from_env()
    dataset_name = _dataset_name_from_files(train_files)

    train_list = []
    print("Loading train files...")
    for f in train_files:
        with open(f, "r", encoding="utf8") as fin:
            for line in fin:
                train_list.append(json.loads(line))

    # train_list = train_list[:10000]  # Test-only shortcut; remove for full runs.
    print("Train size =", len(train_list))

    retrieval_pool_size = int(os.getenv("IDAE_RETRIEVAL_POOL_SIZE", "0"))
    retrieval_pool_seed = int(os.getenv("IDAE_RETRIEVAL_POOL_SEED", "42"))
    retrieval_train_list = select_retrieval_pool(
        train_list, max_size=retrieval_pool_size, seed=retrieval_pool_seed
    )
    if len(retrieval_train_list) != len(train_list):
        print(
            f"Approximate retrieval pool: {len(retrieval_train_list)} / "
            f"{len(train_list)} training samples (seed={retrieval_pool_seed})"
        )
    cache_suffix = f".pool{len(retrieval_train_list)}.seed{retrieval_pool_seed}"

    if ablation_tag == "token":
        print("Retrieval mode: FSMIC token-based (global corpus)")
        retriever = TokenRetriever(
            retrieval_train_list,
            batch_size=int(os.getenv("IDAE_TOKEN_BATCH_SIZE", "32")),
            cache_prefix=f"retrieval_cache/token_{dataset_name}{cache_suffix}",
        )
        id2nearest = retriever.retrieve(
            train_list, top_k=NUM_EXAMPLES, exclude_self=True
        )
    elif ablation_tag == "semantic":
        print("Retrieval mode: FSMIC semantic-based (global corpus)")
        retriever = SemanticRetriever(
            retrieval_train_list,
            model_name=CODESEARCH_MODEL,
            batch_size=int(os.getenv("IDAE_RETRIEVAL_BATCH_SIZE", "64")),
            cache_path=f"retrieval_cache/semantic_{dataset_name}{cache_suffix}.npy",
            index_mode=os.getenv("IDAE_SEMANTIC_INDEX", "flat"),
        )
        id2nearest = retriever.retrieve(
            train_list, top_k=NUM_EXAMPLES, exclude_self=True
        )
    elif use_random_exemplars(ablation_tag):
        id2nearest = build_random_examples(train_list, top_k=NUM_EXAMPLES)
    else:
        print("Retrieval mode: same-intent constrained")
        id2nearest = build_nearest_examples(train_list, model, tokenizer)

    out_path = nearest_examples_filename("train", MODEL_TYPE_ENC, dataset_name, ablation_tag)
    with open(out_path, "w", encoding="utf8") as fout:
        json.dump(id2nearest, fout, ensure_ascii=False, indent=2)

    print("DONE!")
