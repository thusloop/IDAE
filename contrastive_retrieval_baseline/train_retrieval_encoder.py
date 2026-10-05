import argparse
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from common import ensure_dir, get_retriever_checkpoint_path, load_dataset_split, resolve_retriever_model_name, set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CodeCommentPairs(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex["raw_code"], ex["comment"], ex["id"]


class BiEncoderRetriever(nn.Module):
    def __init__(self, model_name: str, proj_dim: int):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.code_encoder = AutoModel.from_pretrained(model_name, config=config)
        self.comment_encoder = AutoModel.from_pretrained(model_name, config=AutoConfig.from_pretrained(model_name))
        hidden_size = self.code_encoder.config.hidden_size
        self.code_proj = nn.Linear(hidden_size, proj_dim)
        self.comment_proj = nn.Linear(hidden_size, proj_dim)

    def _pool(self, outputs):
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]

    def encode_code(self, input_ids, attention_mask):
        outputs = self.code_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs)
        return F.normalize(self.code_proj(pooled), p=2, dim=1)

    def encode_comment(self, input_ids, attention_mask):
        outputs = self.comment_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs)
        return F.normalize(self.comment_proj(pooled), p=2, dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tlcodesum", "funcom"], required=True)
    parser.add_argument("--retriever", choices=["codebert", "simcse"], default="simcse")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="checkpoints/retrieval")
    return parser.parse_args()


def make_collate(tokenizer, max_len: int):
    def collate(batch):
        codes, comments, ids = zip(*batch)
        code_enc = tokenizer(
            list(codes),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        comment_enc = tokenizer(
            list(comments),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return code_enc, comment_enc, list(ids)

    return collate


def bi_encoder_loss(model, code_batch, comment_batch, temperature: float):
    code_z = model.encode_code(code_batch["input_ids"], code_batch["attention_mask"])
    comment_z = model.encode_comment(comment_batch["input_ids"], comment_batch["attention_mask"])
    logits = torch.matmul(code_z, comment_z.transpose(0, 1)) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_code = F.cross_entropy(logits, labels)
    loss_comment = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_code + loss_comment)


def encode_split(model, tokenizer, examples, batch_size: int, max_len: int):
    model.eval()
    dataset = CodeCommentPairs(examples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=make_collate(tokenizer, max_len))
    all_code = []
    all_comment = []
    with torch.no_grad():
        for code_batch, comment_batch, _ in tqdm(loader, desc="Encoding valid", leave=False):
            code_batch = {k: v.to(DEVICE) for k, v in code_batch.items()}
            comment_batch = {k: v.to(DEVICE) for k, v in comment_batch.items()}
            all_code.append(model.encode_code(code_batch["input_ids"], code_batch["attention_mask"]).cpu())
            all_comment.append(model.encode_comment(comment_batch["input_ids"], comment_batch["attention_mask"]).cpu())
    return torch.cat(all_code, dim=0).numpy().astype("float32"), torch.cat(all_comment, dim=0).numpy().astype("float32")


def evaluate(model, tokenizer, examples, batch_size: int, max_len: int):
    code_vecs, comment_vecs = encode_split(model, tokenizer, examples, batch_size, max_len)
    index = faiss.IndexFlatIP(comment_vecs.shape[1])
    index.add(comment_vecs)
    _, indices = index.search(code_vecs, 5)
    recall_at_1 = float(np.mean(indices[:, 0] == np.arange(len(examples))))
    recall_at_5 = float(np.mean([i in indices[i_idx] for i_idx, i in enumerate(range(len(examples)))]))
    return recall_at_1, recall_at_5


def main():
    args = parse_args()
    set_seed(args.seed)
    resolved_model_name = resolve_retriever_model_name(args.retriever, args.model_name)

    print(f"retriever={args.retriever}, model={resolved_model_name}")

    train_examples = load_dataset_split(args.dataset, "train")
    valid_examples = load_dataset_split(args.dataset, "valid")

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, use_fast=True)
    model = BiEncoderRetriever(resolved_model_name, args.proj_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        CodeCommentPairs(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer, args.max_len),
    )

    output_dir = ensure_dir(Path(__file__).resolve().parent / args.output_dir)
    best_path = get_retriever_checkpoint_path(output_dir, args.dataset, args.retriever)
    best_recall = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}")
        for code_batch, comment_batch, _ in progress:
            code_batch = {k: v.to(DEVICE) for k, v in code_batch.items()}
            comment_batch = {k: v.to(DEVICE) for k, v in comment_batch.items()}
            loss = bi_encoder_loss(model, code_batch, comment_batch, args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress.set_postfix_str(f"loss={loss.item():.4f}")

        recall_at_1, recall_at_5 = evaluate(model, tokenizer, valid_examples, args.batch_size, args.max_len)
        print(f"valid recall@1={recall_at_1:.4f}, recall@5={recall_at_5:.4f}")
        if recall_at_1 > best_recall:
            best_recall = recall_at_1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": resolved_model_name,
                    "retriever": args.retriever,
                    "proj_dim": args.proj_dim,
                    "max_len": args.max_len,
                    "dataset": args.dataset,
                },
                best_path,
            )
            print(f"saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
