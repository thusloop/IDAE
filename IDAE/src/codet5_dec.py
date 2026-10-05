# -*- coding: utf-8 -*-
import os
import json
import random
from functools import partial
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, EncoderDecoderModel
#from transformers.optimization import AdamW
from tqdm import tqdm
from ablation_utils import (
    decoder_checkpoint_filename,
    get_ablation_tag,
    nearest_examples_filename,
)
from model_paths import (
    CODEBERT_MODEL,
    CODET5_MODEL,
    CODET5P_220M_MODEL,
    CODET5P_770M_MODEL,
)

# ----------------------------
# Evaluation metrics
# ----------------------------
import nltk
from nltk.translate.bleu_score import SmoothingFunction

from eval.bleu import compute_bleu
from eval.cbleu import nltk_sentence_bleu
from eval.rouge import Rouge
from metric_utils import calculate_meteor
#from eval.meteor import Meteor

def calc_text_similarity(ref, hpy):
    ref = str(ref)
    hpy = str(hpy)
    if not ref.strip() or not hpy.strip():
        return 0.0, 0.0, 0.0
    #cc = SmoothingFunction()
    #bleu = nltk.translate.bleu_score.sentence_bleu([ref], hpy, smoothing_function=cc.method4) 
    bleu = compute_bleu([[ref.split()]], [hpy.split()], smooth=True)[0]
    rouge_score = Rouge().calc_score(hpy.split(), [ref.split()])
    meteor = calculate_meteor(ref, hpy)
    #reference_tokens = [word_tokenize(ref)]
    #hypothesis_tokens = word_tokenize(hpy)
    #meteor = meteor_score(reference_tokens, hypothesis_tokens)
    return bleu, rouge_score, meteor

# ----------------------------
# Configuration
# ----------------------------
def _env_int(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


# MODEL_TYPE = "codet5"  # "codet5" or "codebert"
MODEL_TYPE_ENC = "codebert"
MODEL_TYPE_DEC = "codet5p-220m"  # "codet5", "codet5p-220m", "codet5p-770m", or "codebert"
ABLATION_TAG = get_ablation_tag()

nearest_examples = nearest_examples_filename("train", MODEL_TYPE_ENC, "all", ABLATION_TAG)
CODET5_NAME = CODET5_MODEL
CODET5P_220_PATH = CODET5P_220M_MODEL
CODET5P_770_PATH = CODET5P_770M_MODEL
CODEBERT_NAME = CODEBERT_MODEL
if MODEL_TYPE_DEC == "codet5":
    MODEL_NAME = CODET5_NAME
elif MODEL_TYPE_DEC == "codet5p-220m":
    MODEL_NAME = CODET5P_220_PATH
elif MODEL_TYPE_DEC == "codet5p-770m":
    MODEL_NAME = CODET5P_770_PATH
else:
    MODEL_NAME = CODEBERT_NAME
MAX_LEN = _env_int("IDAE_MAX_LEN", 512)
BATCH_SIZE = _env_int("IDAE_BATCH_SIZE", 32)
NUM_WORKERS = _env_int("IDAE_NUM_WORKERS", 4)
VALIDATION_LIMIT = _env_int("IDAE_VALIDATION_LIMIT", 100)
LR = 5e-5
EPOCHS = _env_int("IDAE_EPOCHS", 1)
NUM_EXAMPLES = _env_int("IDAE_NUM_EXAMPLES", 3)
SEED = _env_int("IDAE_SEED", 42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = f"./checkpoints_dec"
BEST_MODEL_PATH = decoder_checkpoint_filename(
    CHECKPOINT_DIR,
    MODEL_TYPE_ENC,
    MODEL_TYPE_DEC,
    NUM_EXAMPLES,
    "all",
    ABLATION_TAG,
    seed=SEED,
)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

## ----------------------------
## Reinforcement-learning-related configuration
## ----------------------------
USE_RL = False
RL_WEIGHT = 0.1
SAMPLE_TOP_P = 0.6
SAMPLE_TOP_K = 50


def seed_everything(seed):
    """Seed all available RNGs used by data shuffling and model training."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------
# Dataset
# ----------------------------
def load_jsonlines(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    return data

class CodeT5Dataset(Dataset):
    def __init__(self, examples, id2nearest, tokenizer, max_len=None, num_examples=None):
        self.examples = examples
        self.id2nearest = id2nearest
        self.tokenizer = tokenizer
        self.max_len = MAX_LEN if max_len is None else max_len
        self.num_examples = NUM_EXAMPLES if num_examples is None else num_examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        code = ex["raw_code"]
        target = ex["comment"]
        label = ex["label"]
        # Build the similar-code exemplar text while excluding the query itself.
        nearest_list = self.id2nearest.get(ex["id"], [])
        examples_text = ""
        for n in nearest_list[:self.num_examples]:
            examples_text += f"<code>{n['raw_code']}</code> => <comment>{n['comment']}</comment>\n"
            #examples_text += f"<comment>{n['comment']}</comment>\n"
        input_text = f"{code}\nIntention:\n{label}\nSimilar Examples:\n{examples_text}"
        return input_text, target

def collate_fn(batch, tokenizer):
    inputs, targets = zip(*batch)
    enc = tokenizer(
        list(inputs),
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    labels = tokenizer(
        list(targets),
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    return enc, labels

# ----------------------------
# Model/tokenizer construction
# ----------------------------
def build_model_and_tokenizer(model_type: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if model_type == "codet5":
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
        return tokenizer, model
    if model_type == "codet5p-220m":
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
        return tokenizer, model
    if model_type == "codet5p-770m":
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
        return tokenizer, model
    if model_type == "codebert":
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            MODEL_NAME, MODEL_NAME
        ).to(DEVICE)
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return tokenizer, model
    raise ValueError(f"Unsupported MODEL_TYPE: {model_type}")

def compute_seq_logprob(model, input_ids, attention_mask, gen_ids, pad_token_id):
    labels = gen_ids.clone()
    labels[labels == pad_token_id] = -100
    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        use_cache=False,
    )
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = -F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    log_probs = log_probs.view(shift_labels.size())
    token_mask = shift_labels.ne(-100)
    seq_logprob = (log_probs * token_mask).sum(dim=1)
    return seq_logprob

# ----------------------------
# Training function
# ----------------------------
def train(model, tokenizer, train_data, id2nearest, valid_data=None):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_metric = -float("inf")

    train_dataset = CodeT5Dataset(
        train_data,
        id2nearest,
        tokenizer,
        num_examples=NUM_EXAMPLES,
    )
    train_loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "collate_fn": partial(collate_fn, tokenizer=tokenizer),
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }
    if NUM_WORKERS > 0:
        train_loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
        })
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)

    for epoch in range(1, EPOCHS+1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        total_loss = 0.0
        for batch_enc, labels in pbar:
            batch_enc = {
                k: v.to(DEVICE, non_blocking=True)
                for k, v in batch_enc.items()
            }
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(
                input_ids=batch_enc["input_ids"],
                attention_mask=batch_enc["attention_mask"],
                labels=labels,
            )
            loss = outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()

            if USE_RL:
                generator = model.module if isinstance(model, nn.DataParallel) else model
                with torch.no_grad():
                    greedy_ids = generator.generate(
                        input_ids=batch_enc["input_ids"],
                        attention_mask=batch_enc["attention_mask"],
                        max_length=MAX_LEN,
                    )
                    sample_ids = generator.generate(
                        input_ids=batch_enc["input_ids"],
                        attention_mask=batch_enc["attention_mask"],
                        max_length=MAX_LEN,
                        do_sample=True,
                        top_p=SAMPLE_TOP_P,
                        top_k=SAMPLE_TOP_K,
                    )

                pad_id = tokenizer.pad_token_id
                labels_for_decode = torch.where(labels == -100, pad_id, labels)
                targets = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
                greedy_texts = tokenizer.batch_decode(greedy_ids, skip_special_tokens=True)
                sample_texts = tokenizer.batch_decode(sample_ids, skip_special_tokens=True)

                rewards = []
                baselines = []
                for t, s, g in zip(targets, sample_texts, greedy_texts):
                    b1, r1, m1 = calc_text_similarity(t, s)
                    b2, r2, m2 = calc_text_similarity(t, g)
                    rewards.append((b1 + r1 + m1) / 3.0)
                    baselines.append((b2 + r2 + m2) / 3.0)

                rewards = torch.tensor(rewards, device=DEVICE)
                baselines = torch.tensor(baselines, device=DEVICE)

                base_model = model.module if isinstance(model, nn.DataParallel) else model
                seq_logprob = compute_seq_logprob(
                    base_model,
                    batch_enc["input_ids"],
                    batch_enc["attention_mask"],
                    sample_ids,
                    pad_id,
                )
                rl_loss = -((rewards - baselines).detach() * seq_logprob).mean()
                loss = loss + RL_WEIGHT * rl_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix_str(f"loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}", flush=True)
        # torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"codet5_numexp{NUM_EXAMPLES}_epoch{epoch}.pth"))
        if valid_data is not None:
            acc = evaluate(model, tokenizer, valid_data, id2nearest)
            print(f"Validation similarity score: {acc:.4f}", flush=True)
            
            if acc > best_metric:
                best_metric = acc
                state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state, BEST_MODEL_PATH)
                print(f"New best model saved at epoch {epoch}")

# ----------------------------
# Validation/test evaluation
# ----------------------------
def evaluate(model, tokenizer, eval_data, id2nearest):
    model.eval()
    dataset = CodeT5Dataset(
        eval_data,
        id2nearest,
        tokenizer,
        num_examples=NUM_EXAMPLES,
    )
    eval_loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "collate_fn": partial(collate_fn, tokenizer=tokenizer),
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }
    if NUM_WORKERS > 0:
        eval_loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
        })
    loader = DataLoader(dataset, **eval_loader_kwargs)
    total_bleu, total_rouge, total_meteor, count = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch_enc, labels in tqdm(loader, desc="Evaluating"):
            batch_enc = {
                k: v.to(DEVICE, non_blocking=True)
                for k, v in batch_enc.items()
            }
            generator = model.module if isinstance(model, nn.DataParallel) else model
            outputs = generator.generate(
                input_ids=batch_enc["input_ids"],
                attention_mask=batch_enc["attention_mask"],
                max_length=MAX_LEN
            )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            targets = tokenizer.batch_decode(torch.where(labels==-100, tokenizer.pad_token_id, labels), skip_special_tokens=True)
            for p, t in zip(preds, targets):
                bleu, rouge, meteor = calc_text_similarity(t, p)
                total_bleu += bleu
                total_rouge += rouge
                total_meteor += meteor
                count += 1
    model.train()
    print(f"bleu:{total_bleu/count:.4f}, rouge: {total_rouge/count:.4f}, meteor: {total_meteor/count:.4f}")
    return (total_bleu/count + total_rouge/count + total_meteor/count)/3


# ----------------------------
# Main function
# ----------------------------
def _best_model_path(dataset_name: str):
    return decoder_checkpoint_filename(
        CHECKPOINT_DIR,
        MODEL_TYPE_ENC,
        MODEL_TYPE_DEC,
        NUM_EXAMPLES,
        dataset_name,
        ABLATION_TAG,
        seed=SEED,
    )

def _nearest_examples_path(dataset_name: str):
    dataset_specific = nearest_examples_filename("train", MODEL_TYPE_ENC, dataset_name, ABLATION_TAG)
    if os.path.exists(dataset_specific):
        return dataset_specific
    return nearest_examples


def _selected_datasets():
    dataset_files = {
        "tlcodesum": ("../../data/tlcodesum.train", "../../data/tlcodesum.valid"),
        "funcom": ("../../data/funcom.train", "../../data/funcom.valid"),
    }
    raw = os.getenv("IDAE_DATASETS", "tlcodesum,funcom")
    names = [name.strip().lower() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("IDAE_DATASETS must contain at least one dataset")
    unknown = sorted(set(names) - set(dataset_files))
    if unknown:
        raise ValueError(
            f"Unsupported dataset(s): {', '.join(unknown)}. "
            f"Expected one of: {', '.join(dataset_files)}"
        )
    return [(name, *dataset_files[name]) for name in names]

def _train_one_dataset(dataset_name: str, train_path: str, valid_path: str):
    train_list = load_jsonlines(train_path)
    valid_list = load_jsonlines(valid_path)
    #test_list = load_jsonlines("../../data/tlcodesum.test") + load_jsonlines("../../data/funcom.test")
    random.seed(SEED)
    random.shuffle(train_list)
    random.shuffle(valid_list)
    #train_list = train_list[:50000]
    #valid_list = valid_list[:10000]
    print(f"[{dataset_name}] Train size:", len(train_list))
    print(f"[{dataset_name}] Valid size:", len(valid_list))
    valid_for_training = valid_list[:VALIDATION_LIMIT]
    print(
        f"[{dataset_name}] Validation samples used during training:",
        len(valid_for_training),
    )

    # Build the nearest examples.
    #id2nearest = build_nearest_examples(train_list, codebert_model, codebert_tokenizer, top_k=NUM_EXAMPLES)
    if NUM_EXAMPLES == 0:
        id2nearest = {}
        print("NUM_EXAMPLES=0; skip loading nearest-example JSON", flush=True)
    else:
        with open(_nearest_examples_path(dataset_name), "r", encoding="utf8") as f:
            id2nearest = json.load(f)
    # ---------------- Load the model
    tokenizer, model = build_model_and_tokenizer(MODEL_TYPE_DEC)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    global BEST_MODEL_PATH
    BEST_MODEL_PATH = _best_model_path(dataset_name)
    train(model, tokenizer, train_list, id2nearest, valid_data=valid_for_training)

def main():
    seed_everything(SEED)
    print(
        f"Run configuration: seed={SEED}, num_examples={NUM_EXAMPLES}, "
        f"epochs={EPOCHS}, batch_size={BATCH_SIZE}, max_len={MAX_LEN}",
        flush=True,
    )
    print(f"Ablation mode: {ABLATION_TAG}")
    for dataset_name, train_path, valid_path in _selected_datasets():
        _train_one_dataset(dataset_name, train_path, valid_path)

if __name__ == "__main__":
    main()



