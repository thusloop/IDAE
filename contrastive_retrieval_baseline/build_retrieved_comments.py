import argparse
import json
from pathlib import Path

import faiss
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from common import (
    ensure_dir,
    get_legacy_retrieved_comments_path,
    get_legacy_retriever_checkpoint_path,
    get_retrieved_comments_path,
    get_retriever_checkpoint_path,
    load_dataset_split,
    resolve_checkpoint_model_name,
    resolve_retriever_model_name,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        return torch.nn.functional.normalize(self.code_proj(pooled), p=2, dim=1)

    def encode_comment(self, input_ids, attention_mask):
        outputs = self.comment_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs)
        return torch.nn.functional.normalize(self.comment_proj(pooled), p=2, dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tlcodesum", "funcom"], required=True)
    parser.add_argument("--retriever", choices=["codebert", "simcse"], default="simcse")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="artifacts/retrieved_comments")
    return parser.parse_args()


def resolve_checkpoint_path(base_dir: Path, dataset: str, retriever: str, override: str = None) -> Path:
    if override:
        return Path(override)
    checkpoint_path = get_retriever_checkpoint_path(base_dir, dataset, retriever)
    if checkpoint_path.exists():
        return checkpoint_path
    if retriever == "codebert":
        legacy_path = get_legacy_retriever_checkpoint_path(base_dir, dataset)
        if legacy_path.exists():
            return legacy_path
    return checkpoint_path


def load_model(checkpoint_path: Path, retriever: str):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    resolved_retriever = checkpoint.get("retriever", retriever)
    model_name = resolve_checkpoint_model_name(checkpoint.get("model_name"), resolved_retriever)
    proj_dim = checkpoint.get("proj_dim", 256)
    max_len = checkpoint.get("max_len", 256)
    model = BiEncoderRetriever(model_name, proj_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model.eval()
    return model, tokenizer, max_len


def encode_comments(model, tokenizer, examples, batch_size: int, max_len: int):
    comments = [ex["comment"] for ex in examples]
    vectors = []
    with torch.no_grad():
        for i in tqdm(range(0, len(comments), batch_size), desc="Encoding train comments"):
            enc = tokenizer(
                comments[i:i + batch_size],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            vectors.append(model.encode_comment(enc["input_ids"], enc["attention_mask"]).cpu())
    return torch.cat(vectors, dim=0).numpy().astype("float32")


def encode_codes(model, tokenizer, examples, batch_size: int, max_len: int, desc: str):
    codes = [ex["raw_code"] for ex in examples]
    vectors = []
    with torch.no_grad():
        for i in tqdm(range(0, len(codes), batch_size), desc=desc):
            enc = tokenizer(
                codes[i:i + batch_size],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            vectors.append(model.encode_code(enc["input_ids"], enc["attention_mask"]).cpu())
    return torch.cat(vectors, dim=0).numpy().astype("float32")


def build_for_split(model, tokenizer, max_len: int, train_examples, query_examples, split_name: str, top_k: int, batch_size: int):
    train_comment_vecs = encode_comments(model, tokenizer, train_examples, batch_size, max_len)
    query_code_vecs = encode_codes(model, tokenizer, query_examples, batch_size, max_len, desc=f"Encoding {split_name} codes")

    index = faiss.IndexFlatIP(train_comment_vecs.shape[1])
    index.add(train_comment_vecs)
    search_k = top_k + 1 if split_name == "train" else top_k
    scores, indices = index.search(query_code_vecs, search_k)

    id2retrieved = {}
    for q_idx, ex in enumerate(query_examples):
        retrieved = []
        for score, doc_idx in zip(scores[q_idx], indices[q_idx]):
            candidate = train_examples[doc_idx]
            if split_name == "train" and candidate["id"] == ex["id"]:
                continue
            retrieved.append({
                "id": candidate["id"],
                "raw_code": candidate["raw_code"],
                "comment": candidate["comment"],
                "score": float(score),
            })
            if len(retrieved) >= top_k:
                break
        id2retrieved[ex["id"]] = retrieved
    return id2retrieved


def main():
    args = parse_args()
    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints" / "retrieval"
    checkpoint_path = resolve_checkpoint_path(checkpoint_dir, args.dataset, args.retriever, args.checkpoint)
    model, tokenizer, max_len = load_model(checkpoint_path, args.retriever)

    print(f"retriever={args.retriever}, checkpoint={checkpoint_path}")

    train_examples = load_dataset_split(args.dataset, "train")
    output_dir = ensure_dir(Path(__file__).resolve().parent / args.output_dir)

    for split in ["train", "valid", "test"]:
        query_examples = load_dataset_split(args.dataset, split)
        id2retrieved = build_for_split(
            model,
            tokenizer,
            max_len,
            train_examples,
            query_examples,
            split,
            args.top_k,
            args.batch_size,
        )
        out_path = get_retrieved_comments_path(output_dir, args.dataset, args.top_k, split, args.retriever)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(id2retrieved, handle, ensure_ascii=False, indent=2)
        print(f"saved {out_path}")
        if args.retriever == "codebert":
            legacy_out_path = get_legacy_retrieved_comments_path(output_dir, args.dataset, args.top_k, split)
            with legacy_out_path.open("w", encoding="utf-8") as handle:
                json.dump(id2retrieved, handle, ensure_ascii=False, indent=2)
            print(f"saved compatibility copy {legacy_out_path}")


if __name__ == "__main__":
    main()
