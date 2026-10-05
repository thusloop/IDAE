import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common import (
    CODET5P_220M_PATH,
    average_text_score,
    ensure_dir,
    get_generator_checkpoint_path,
    get_legacy_generator_checkpoint_path,
    get_legacy_retrieved_comments_path,
    get_retrieved_comments_path,
    load_dataset_split,
    set_seed,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256


def format_retrieved_examples(retrieved_items):
    return "\n".join(
        f"<code>{item['raw_code']}</code> => <comment>{item['comment']}</comment>"
        for item in retrieved_items
    )


class CommentGenerationDataset(Dataset):
    def __init__(self, examples, retrieved_map, top_k: int):
        self.examples = examples
        self.retrieved_map = retrieved_map
        self.top_k = top_k

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        retrieved = self.retrieved_map.get(ex["id"], [])[:self.top_k]
        retrieved_text = format_retrieved_examples(retrieved)
        input_text = f"{ex['raw_code']}\nIntention:\n{ex['label']}\nSimilar Examples:\n{retrieved_text}"
        return input_text, ex["comment"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tlcodesum", "funcom"], required=True)
    parser.add_argument("--model-name", default=CODET5P_220M_PATH)
    parser.add_argument("--retriever", choices=["codebert", "simcse"], default="simcse")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieved-dir", default="artifacts/retrieved_comments")
    parser.add_argument("--output-dir", default="checkpoints/generator")
    return parser.parse_args()


def load_retrieved_map(base_dir: Path, dataset: str, split: str, top_k: int, retriever: str):
    path = get_retrieved_comments_path(base_dir, dataset, top_k, split, retriever)
    if not path.exists() and retriever == "codebert":
        legacy_path = get_legacy_retrieved_comments_path(base_dir, dataset, top_k, split)
        if legacy_path.exists():
            path = legacy_path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_collate(tokenizer, max_len: int):
    def collate(batch):
        inputs, targets = zip(*batch)
        model_inputs = tokenizer(
            list(inputs),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        labels = tokenizer(
            list(targets),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        return model_inputs, labels, list(targets)

    return collate


def evaluate(model, tokenizer, examples, retrieved_map, batch_size: int, max_len: int, top_k: int):
    model.eval()
    dataset = CommentGenerationDataset(examples, retrieved_map, top_k)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=make_collate(tokenizer, max_len))
    total_score = 0.0
    count = 0
    with torch.no_grad():
        for model_inputs, labels, targets in tqdm(loader, desc="Evaluating", leave=False):
            model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
            generator = model.module if isinstance(model, nn.DataParallel) else model
            outputs = generator.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_len,
            )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for pred, target in zip(preds, targets):
                total_score += average_text_score(target, pred)
                count += 1
    model.train()
    return total_score / max(count, 1)


def main():
    args = parse_args()
    set_seed(args.seed)

    retrieved_dir = Path(__file__).resolve().parent / args.retrieved_dir
    train_examples = load_dataset_split(args.dataset, "train")
    valid_examples = load_dataset_split(args.dataset, "valid")
    train_retrieved = load_retrieved_map(retrieved_dir, args.dataset, "train", args.top_k, args.retriever)
    valid_retrieved = load_retrieved_map(retrieved_dir, args.dataset, "valid", args.top_k, args.retriever)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        CommentGenerationDataset(train_examples, train_retrieved, args.top_k),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer, args.max_len),
    )

    output_dir = ensure_dir(Path(__file__).resolve().parent / args.output_dir)
    best_path = get_generator_checkpoint_path(output_dir, args.dataset, args.top_k, args.retriever)
    best_score = -1.0

    print(f"generator retriever source={args.retriever}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}")
        for model_inputs, labels, _ in progress:
            model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
            labels = labels.to(DEVICE)
            outputs = model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                labels=labels,
            )
            loss = outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress.set_postfix_str(f"loss={loss.item():.4f}")

        valid_score = evaluate(
            model,
            tokenizer,
            valid_examples,
            valid_retrieved,
            args.batch_size,
            args.max_len,
            args.top_k,
        )
        print(f"valid avg score={valid_score:.4f}")
        if valid_score > best_score:
            best_score = valid_score
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, best_path)
            print(f"saved best checkpoint to {best_path}")
            if args.retriever == "codebert":
                legacy_path = get_legacy_generator_checkpoint_path(output_dir, args.dataset, args.top_k)
                torch.save(state, legacy_path)
                print(f"saved compatibility copy {legacy_path}")


if __name__ == "__main__":
    main()


"""
CUDA_VISIBLE_DEVICES=0,2,3 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
python -u train_comment_generator.py \
  --dataset tlcodesum \
  --retriever codebert \
  --top-k 3 \
  --max-len 512 \
  --batch-size 16 \
  --epochs 1 \
  --seed 42 \
  --retrieved-dir artifacts/retrieved_comments \
  --output-dir checkpoints/generator/seed42


CUDA_VISIBLE_DEVICES=4,5,6,7 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
python -u train_comment_generator.py \
  --dataset funcom \
  --retriever codebert \
  --top-k 3 \
  --max-len 512 \
  --batch-size 16 \
  --epochs 1 \
  --seed 42 \
  --retrieved-dir artifacts/retrieved_comments \
  --output-dir checkpoints/generator/seed42

"""