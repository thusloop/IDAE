import json
import torch
import faiss
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch.nn as nn
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
DEVICE = "cuda"
BATCH_SIZE = 128
MAX_LEN = 256
NUM_EXAMPLES = 5
MODEL_NAME = "microsoft/codebert-base"


# =====================
# Load the model class used during training
# =====================
class CodeEncoderWithProjection(nn.Module):
    def __init__(self, model_name: str, proj_dim: int = 128):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size
        
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        cls_tokens = last[:, 0, :]
        z = self.proj(cls_tokens)
        z = nn.functional.normalize(z, p=2, dim=1)
        return z


def load_trained_codebert(model_path):
    model = CodeEncoderWithProjection(MODEL_NAME, proj_dim=128)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])    # Load the state dict correctly
    return model.to(DEVICE).eval()


# =====================
# Encoding helper
# =====================
def encode_batch(model, input_ids, attention_mask):
    with torch.no_grad():
        z = model(input_ids=input_ids, attention_mask=attention_mask)
    return z


# =====================
# Build nearest examples
# =====================
def build_nearest_examples(train_list, model, tokenizer, top_k=NUM_EXAMPLES):
    all_codes = [ex["raw_code"] for ex in train_list]
    ids = [ex["id"] for ex in train_list]

    print("Tokenizing all codes...")
    enc = tokenizer(
        all_codes,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    print("Encoding vectors...")
    vecs = []
    for i in tqdm(range(0, len(all_codes), BATCH_SIZE)):
        z = encode_batch(
            model,
            input_ids[i:i+BATCH_SIZE],
            attention_mask[i:i+BATCH_SIZE],
        )
        vecs.append(z.cpu())

    vecs = torch.cat(vecs, dim=0).numpy().astype(np.float32)

    print("Building FAISS GPU index...")
    dim = vecs.shape[1]

    cpu_index = faiss.IndexFlatIP(dim)
    gpu_res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)

    index.add(vecs)

    print("Batch searching...")
    batch_size = 4096
    all_I = []
    all_D = []

    for i in tqdm(range(0, len(vecs), batch_size)):
        D, I = index.search(vecs[i:i+batch_size], top_k+1)
        all_D.append(D)
        all_I.append(I)

    all_D = np.vstack(all_D)
    all_I = np.vstack(all_I)

    print("Building dict...")
    id2nearest = {}

    for i in tqdm(range(len(train_list))):
        neighbors = []
        for j in all_I[i]:
            if ids[j] != ids[i]:
                neighbors.append({
                    "id": ids[j],
                    "raw_code": train_list[j]["raw_code"],
                    "comment": train_list[j]["comment"],
                })
        id2nearest[ids[i]] = neighbors[:top_k]

    # print("Building FAISS index...")
    # index = faiss.IndexFlatIP(vecs.shape[1])
    # index.add(vecs)

    # print("Searching nearest neighbors...")
    # id2nearest = {}

    # for i in tqdm(range(len(train_list))):
    #     D, I = index.search(vecs[i:i+1], top_k + 1)

    #     nearest = []
    #     for j in I[0]:
    #         if ids[j] != ids[i]:
    #             nearest.append({
    #                 "id": ids[j],
    #                 "raw_code": train_list[j]["raw_code"],
    #                 "comment": train_list[j]["comment"],
    #             })
    #     id2nearest[ids[i]] = nearest[:top_k]

    return id2nearest


# =====================
# Main program
# =====================
if __name__ == "__main__":


    print("Loading trained CodeBERT...")
    model = load_trained_codebert("./checkpoints/best_model.pth")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_files = [
        "../../data/tlcodesum.train",
        "../../data/funcom.train"
        ]
    train_list = []

    print("Loading train files...")
    for f in train_files:
        with open(f, "r", encoding="utf8") as fin:
            for line in fin:
                train_list.append(json.loads(line))

    print("Train size =", len(train_list))

    id2nearest = build_nearest_examples(train_list, model, tokenizer)

    with open("nearest_examples.json", "w", encoding="utf8") as fout:
        json.dump(id2nearest, fout, ensure_ascii=False, indent=2)

    print("DONE!")
