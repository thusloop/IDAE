import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# =========================
# Parameter settings
# =========================
MODEL_NAME = "princeton-nlp/sup-simcse-roberta-base"  # Recommended compatible version
TOP_K = 10
SIM_THRESHOLD = 0.8         # Similarity threshold; adjust as needed, with 0.7 to 0.9 recommended
BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1. Read and merge data
# =========================
def load_and_merge_datasets(file_paths):
    all_data = []
    for path in file_paths:
        print(f"Loading {path} ...")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    print(f"Total samples after merging: {len(all_data)}")
    return all_data


# =========================
# 2. Encode comments with SimCSE
# =========================
def encode_comments(model, comments):
    embeddings = []
    for i in tqdm(range(0, len(comments), BATCH_SIZE), desc="Encoding comments"):
        batch = comments[i:i+BATCH_SIZE]
        emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, device=DEVICE)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    print(f"Encoded shape: {embeddings.shape}")
    return embeddings


# =========================
# 3. Build an index with FAISS
# =========================
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product equals cosine similarity because vectors are normalized
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors")
    return index


# =========================
# 4. Find top-k most similar samples and filter by threshold
# =========================
def find_topk_similar(index, embeddings, ids, top_k, sim_threshold):
    scores, indices = index.search(embeddings, top_k + 1)  # +1 because the sample itself is included
    results = {}
    total_selected = 0

    for i, (score_list, idx_list) in enumerate(zip(scores, indices)):
        anchor_id = ids[i]
        similar_items = []

        for s, j in zip(score_list, idx_list):
            if ids[j] == anchor_id:
                continue  # Skip the sample itself
            if s >= sim_threshold:
                similar_items.append({"id": ids[j], "score": float(s)})

        if similar_items:
            results[anchor_id] = similar_items
            total_selected += len(similar_items)

    avg_selected = total_selected / len(results) if results else 0
    print(f"Found {total_selected} positive pairs (avg {avg_selected:.2f} per anchor)")
    print(f"Using similarity threshold: {sim_threshold}")
    return results


# =========================
# Main pipeline
# =========================
if __name__ == "__main__":
    file_paths = [
        "../../data/tlcodesum.train",
        "../../data/funcom.train"
    ]
    all_data = load_and_merge_datasets(file_paths)
    all_data = all_data[:100000]
    comments = [d["comment"] for d in all_data]
    ids = [d["id"] for d in all_data]

    print("Loading SimCSE model...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # Step 1: Encode comments
    embeddings = encode_comments(model, comments)

    # Step 2: Build the index
    index = build_faiss_index(embeddings)

    # Step 3: Find top-k similar samples and filter
    results = find_topk_similar(index, embeddings, ids, TOP_K, SIM_THRESHOLD)

    # Step 4: Write results
    output_file = f"positive_pool_faiss_th{SIM_THRESHOLD}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved successfully: {output_file}")
