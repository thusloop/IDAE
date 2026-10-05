import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common import (
    CODET5P_220M_PATH,
    calc_text_similarity,
    get_generator_checkpoint_path,
    get_legacy_generator_checkpoint_path,
    get_legacy_predictions_path,
    get_legacy_retrieved_comments_path,
    get_retrieved_comments_path,
    load_dataset_split,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 1024
CANONICAL_ABLATIONS = ("main", "contrastive", "rand_ex", "token", "semantic")


def canonical_prediction_filename(
    dataset: str,
    top_k: int,
    ablation: str,
    seed: int,
) -> str:
    """Return the prediction name used by compare_llm/cal_pre.py.

    Every ablation includes its tag before the optional seed suffix, for
    example
    ``predictions_label_num3_funcom.contrastive.seed42.jsonl`` or
    ``predictions_label_num3_funcom.token.seed42.jsonl``.
    """
    seed_suffix = f".seed{seed}" if seed is not None else ""
    return f"predictions_label_num{top_k}_{dataset}.{ablation}{seed_suffix}.jsonl"


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
        return input_text, ex["comment"], ex["label"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tlcodesum", "funcom"], required=True)
    parser.add_argument("--model-name", default=CODET5P_220M_PATH)
    parser.add_argument("--retriever", choices=["codebert", "simcse"], default="simcse")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retrieved-dir", default="artifacts/retrieved_comments")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--ablation",
        choices=CANONICAL_ABLATIONS,
        default=None,
        help=(
            "Canonical ablation tag used in the output filename. If omitted, "
            "codebert maps to contrastive and simcse maps to main."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("IDAE_SEED", "42")),
        help="Seed included in the canonical prediction filename (default: 42)",
    )
    parser.add_argument("--output", default=None)
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
        inputs, targets, labels = zip(*batch)
        model_inputs = tokenizer(
            list(inputs),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return model_inputs, list(targets), list(labels)

    return collate


def resolve_checkpoint_path(base_dir: Path, dataset: str, top_k: int, retriever: str, override: str = None) -> Path:
    if override:
        return Path(override)
    checkpoint_path = get_generator_checkpoint_path(base_dir, dataset, top_k, retriever)
    if not checkpoint_path.exists() and retriever == "codebert":
        legacy_path = get_legacy_generator_checkpoint_path(base_dir, dataset, top_k)
        if legacy_path.exists():
            return legacy_path
    return checkpoint_path


def main():
    args = parse_args()
    # Keep the two retriever variants separate by default.  The mapping can be
    # overridden explicitly when a particular paper table uses another label.
    ablation = args.ablation or (
        "contrastive" if args.retriever == "codebert" else "main"
    )
    project_dir = Path(__file__).resolve().parent
    checkpoint_dir = project_dir / "checkpoints" / "generator"
    checkpoint_path = resolve_checkpoint_path(checkpoint_dir, args.dataset, args.top_k, args.retriever, args.checkpoint)
    output_path = (
        Path(args.output)
        if args.output
        else project_dir
        / canonical_prediction_filename(
            args.dataset, args.top_k, ablation, args.seed
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = load_dataset_split(args.dataset, "test")
    retrieved_map = load_retrieved_map(Path(__file__).resolve().parent / args.retrieved_dir, args.dataset, "test", args.top_k, args.retriever)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(
        CommentGenerationDataset(examples, retrieved_map, args.top_k),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer, args.max_len),
    )

    total_bleu = 0.0
    total_rouge = 0.0
    total_meteor = 0.0
    total_count = 0
    prediction_lines = []

    print(
        f"evaluate retriever source={args.retriever}, "
        f"ablation={ablation}, seed={args.seed}, checkpoint={checkpoint_path}"
    )

    with torch.no_grad():
        for model_inputs, targets, labels in tqdm(loader, desc="Testing"):
            model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
            outputs = model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=args.max_len,
            )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for pred, target, label in zip(preds, targets, labels):
                bleu, rouge, meteor = calc_text_similarity(target, pred)
                total_bleu += bleu
                total_rouge += rouge
                total_meteor += meteor
                total_count += 1
                prediction_lines.append(json.dumps({
                    "prediction": pred,
                    "ground_truth": target,
                    "label": label,
                    "bleu": bleu,
                    "rouge": rouge,
                    "meteor": meteor,
                }, ensure_ascii=False))

    with output_path.open("w", encoding="utf-8", newline="\n") as writer:
        for line in prediction_lines:
            writer.write(line + "\n")

    if args.retriever == "codebert":
        legacy_output_path = get_legacy_predictions_path(Path(__file__).resolve().parent, args.dataset, args.top_k)
        with legacy_output_path.open("w", encoding="utf-8", newline="\n") as writer:
            for line in prediction_lines:
                writer.write(line + "\n")
        print(f"saved compatibility copy {legacy_output_path}")

    if total_count == 0:
        raise RuntimeError("test split is empty; no predictions were generated")
    print(f"BLEU={total_bleu / total_count:.4f}")
    print(f"ROUGE-L={total_rouge / total_count:.4f}")
    print(f"METEOR={total_meteor / total_count:.4f}")
    print(f"saved predictions to {output_path}")


if __name__ == "__main__":
    main()

