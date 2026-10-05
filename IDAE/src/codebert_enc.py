"""
Codes with similar comments should be close in the embedding space.

"""

import json
import random
from typing import List, Tuple
#from datasets import load_dataset, Dataset, DatasetDict
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoConfig, AutoTokenizer, AutoModel
import faiss
from tqdm import tqdm
import os
from model_paths import CODEBERT_MODEL, CODET5_MODEL

# ----------------------------
# Hyperparameter configuration
# ----------------------------
MODEL_TYPE_ENC = "codebert"  # "codebert" or "codet5"
CODEBERT_NAME = CODEBERT_MODEL
CODET5_NAME = CODET5_MODEL
MODEL_NAME = CODEBERT_NAME if MODEL_TYPE_ENC == "codebert" else CODET5_NAME
MAX_LEN = 256
BATCH_SIZE = 128
LR = 5e-5
EPOCHS = 3
PROJECT_DIM = 128
NEG_SAMPLE_PER_POS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = f"./checkpoints_enc"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f"best_enc_{MODEL_TYPE_ENC}.pth")
# Create the checkpoint directory.
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------------------
# Load data and the positive-example pool
# ----------------------------
def load_jsonlines(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    return data

def build_dataset(train_paths: List[str], test_paths: List[str]):
    train_list = []
    test_list = []
    for p in train_paths:
        train_list.extend(load_jsonlines(p))
    for p in test_paths:
        test_list.extend(load_jsonlines(p))
    return train_list, test_list

def load_positive_pool(path: str):
    with open(path, "r", encoding="utf-8") as f:
        pool = json.load(f)
    return pool

# ----------------------------
# Dataset and DataLoader
class CodeCommentDataset(torch.utils.data.Dataset):
    def __init__(self, examples, id_to_index):
        self.examples = examples
        self.id_to_index = id_to_index
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex["raw_code"], ex["comment"], ex["id"]

def collate_fn(batch, tokenizer):
    codes, comments, ids = zip(*batch)
    enc = tokenizer(
        list(codes),
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    return enc, list(comments), list(ids)

# ----------------------------
# Model definition
class CodeEncoderWithProjection(nn.Module):
    def __init__(self, model_name: str, proj_dim: int, model_type: str):
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
            last = out.last_hidden_state
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(out, "last_hidden_state"):
                last = out.last_hidden_state
            else:
                raise ValueError("Model output does not have last_hidden_state")
        cls_tokens = last[:, 0, :]
        z = self.proj(cls_tokens)
        z = nn.functional.normalize(z, p=2, dim=1)
        return z

# ----------------------------
# Loss function
def contrastive_loss(z_anchor: torch.Tensor,
                     z_pos: torch.Tensor,
                     z_neg: torch.Tensor,
                     temperature: float = 0.07):
    B, D = z_anchor.shape
    pos_sim = torch.sum(z_anchor * z_pos, dim=1, keepdim=True)
    sim_neg = torch.bmm(z_neg, z_anchor.unsqueeze(2)).squeeze(2)
    logits = torch.cat([pos_sim, sim_neg], dim=1)
    labels = torch.zeros(B, dtype=torch.long, device=z_anchor.device)
    logits = logits / temperature
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

# ----------------------------
# Validation: retrieval accuracy at Top-k or mean similarity
def evaluate(model, tokenizer, eval_data, id_to_idx, top_k: int = 5):
    model.eval()
    # Build the index.
    ids = [ex["id"] for ex in eval_data]
    all_z = []
    with torch.no_grad():
        dataset = CodeCommentDataset(eval_data, id_to_idx)
        dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda x: collate_fn(x, tokenizer))
        for batch_enc, batch_comments, batch_ids in tqdm(dl, desc="Eval indexing"):
            batch_enc = {k: v.to(DEVICE) for k, v in batch_enc.items()}
            z = model(batch_enc["input_ids"], batch_enc["attention_mask"])
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0).numpy().astype("float32")
    index = faiss.IndexFlatIP(all_z.shape[1])
    index.add(all_z)
    # Retrieve for each query. Here each code is used as its own query.
    correct = 0
    total = len(eval_data)
    for i, ex in enumerate(eval_data):
        enc = tokenizer(ex["raw_code"], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            zq = model(enc["input_ids"], enc["attention_mask"]).cpu().numpy().astype("float32")
        D, I = index.search(zq, top_k)
        # Count the query as correct if it appears in the retrieved results.
        if i in I[0]:
            correct += 1
    accuracy = correct / total
    model.train()
    return accuracy

# ----------------------------
# Model save/load helpers
def save_checkpoint(state: dict, filename: str):
    torch.save(state, filename)

def load_checkpoint(filename: str, model, optimizer=None):
    checkpoint = torch.load(filename, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", None)
    best_metric = checkpoint.get("best_metric", None)
    return epoch, best_metric

# ----------------------------
# Main training loop, including validation and checkpoints
def train(train_data, valid_data, tokenizer, model, optimizer,
          positive_pool, id_to_idx):
    best_acc = 0.0
    id2code = {ex["id"]: ex["raw_code"] for ex in train_data + valid_data}
    for epoch in range(1, EPOCHS+1):
        model.train()
        dataset = CodeCommentDataset(train_data, id_to_idx)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                collate_fn=lambda x: collate_fn(x, tokenizer))
        #pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True, leave=True)
        for step, (batch_enc, batch_comments, batch_ids) in enumerate(pbar):

            # ------ 1. Encode anchors ------
            batch_enc = {k: v.to(DEVICE) for k, v in batch_enc.items()}
            z_anchor = model(batch_enc["input_ids"], batch_enc["attention_mask"])   # (B, D)

            # ------ 2. Encode positive examples across batches ------
            pos_codes = []
            for i, aid in enumerate(batch_ids):
                if aid in positive_pool and len(positive_pool[aid]) > 0:
                    pos_id = positive_pool[aid][0]["id"]
                    pos_codes.append(id2code[pos_id])
                else:
                    # If no positive example is available, choose a different random example.
                    rand_id = random.choice([id for id in id2code.keys() if id != aid])
                    pos_codes.append(id2code[rand_id])

            pos_enc = tokenizer(
                pos_codes,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():   # Positive examples do not require backpropagation.
                z_pos = model(pos_enc["input_ids"], pos_enc["attention_mask"])  # (B, D)

            # ------ 3. Sample negatives randomly within the batch ------
            B = len(batch_ids)
            neg_list = []
            for i in range(B):
                neg_idxes = [j for j in range(B) if j != i]
                chosen = random.sample(neg_idxes, min(NEG_SAMPLE_PER_POS, len(neg_idxes)))
                neg_samples = torch.stack([z_anchor[j] for j in chosen], dim=0)
                neg_list.append(neg_samples)

            max_neg = max(n.size(0) for n in neg_list)
            neg_padded = []
            for n in neg_list:
                if n.size(0) < max_neg:
                    pad = n[-1].unsqueeze(0).repeat(max_neg - n.size(0), 1)
                    n = torch.cat([n, pad], dim=0)
                neg_padded.append(n)

            z_neg = torch.stack(neg_padded, dim=0)  # (B, K, D)

            loss = contrastive_loss(z_anchor, z_pos, z_neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #pbar.set_postfix(loss=loss.item())
            pbar.set_postfix_str(f"loss={loss.item():.4f}")


        # Evaluate on the validation set after each epoch.
        val_acc = evaluate(model, tokenizer, valid_data, id_to_idx, top_k=5)
        print(f"Epoch {epoch} validation accuracy @Top-5: {val_acc:.4f}")

        # Save the best model according to the validation metric.
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"New best model found (acc {best_acc:.4f}), saving checkpoint.")
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_acc
            }, BEST_MODEL_PATH)

    print("Training complete. Best validation accuracy:", best_acc)

# ----------------------------
# Indexing and retrieval helpers
def build_index(model, tokenizer, train_data):
    model.eval()
    ids = [ex["id"] for ex in train_data]
    all_z = []
    with torch.no_grad():
        dataset = CodeCommentDataset(train_data, {ex["id"]: i for i, ex in enumerate(train_data)})
        dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda x: collate_fn(x, tokenizer))
        for batch_enc, _, _ in tqdm(dl, desc="Indexing"):
            batch_enc = {k: v.to(DEVICE) for k, v in batch_enc.items()}
            z = model(batch_enc["input_ids"], batch_enc["attention_mask"])
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0).numpy().astype("float32")
    idx = faiss.IndexFlatIP(all_z.shape[1])
    idx.add(all_z)
    return idx, ids, all_z

def retrieve(idx, ids, all_z, model, tokenizer, raw_code_query, top_k=5):
    model.eval()
    enc = tokenizer(raw_code_query, padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        zq = model(enc["input_ids"], enc["attention_mask"]).cpu().numpy().astype("float32")
    D, I = idx.search(zq, top_k)
    results = []
    for dist, i in zip(D[0], I[0]):
        results.append((ids[i], dist))
    return results

# ----------------------------
# Main entry point
def main():
    train_list, test_list = build_dataset(
        train_paths=[
            "../../data/tlcodesum.train",
            "../../data/funcom.train",
            ],
        test_paths=[
            "../../data/tlcodesum.test",
            "../../data/funcom.test",
            ]
    )
    print("Training set size:", len(train_list))
    print("Test set size:", len(test_list))

    # Split out a validation subset from the training data.
    # The simple strategy below uses the first 10% of the training data.
    split_idx = int(len(train_list) * 0.1)
    valid_list = train_list[:split_idx]
    actual_train_list = train_list[split_idx:]

    id_to_idx = {ex["id"]: i for i, ex in enumerate(actual_train_list)}

    positive_pool = load_positive_pool("../../data/positive_pool_sorted.json")

    #tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = CodeEncoderWithProjection(MODEL_NAME, proj_dim=PROJECT_DIM, model_type=MODEL_TYPE_ENC).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Optionally resume from an existing checkpoint.
    if os.path.exists(BEST_MODEL_PATH):
        print("Found an existing best-model checkpoint; loading it...")
        epoch0, best_metric0 = load_checkpoint(BEST_MODEL_PATH, model, optimizer)
        print(f"Resumed from epoch {epoch0}, best_metric {best_metric0:.4f}")

    train(actual_train_list, valid_list, tokenizer, model, optimizer, positive_pool, id_to_idx)

    # Build an index and retrieve with the trained (or resumed best) model.
    idx, ids, all_z = build_index(model, tokenizer, actual_train_list)

    # Demonstrate retrieval.
    sample = test_list[0]
    print("Query comment:", sample["comment"])
    print("Query code:", sample["raw_code"][:200])
    results = retrieve(idx, ids, all_z, model, tokenizer, sample["raw_code"], top_k=5)
    print("Top-5 retrieval results (id, similarity) =", results)

if __name__ == "__main__":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    main()
