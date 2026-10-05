# -*- coding: utf-8 -*-
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from ablation_utils import (
    decoder_checkpoint_filename,
    get_ablation_tag,
    nearest_examples_filename,
    prediction_filename,
)
from metric_utils import calculate_meteor
from model_paths import (
    CODEBERT_MODEL,
    CODET5_MODEL,
    CODET5P_220M_MODEL,
    CODET5P_770M_MODEL,
)


def _env_int(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc

# ----------------------------
# Evaluation metrics
# ----------------------------


from eval.bleu import compute_bleu
from eval.cbleu import nltk_sentence_bleu
from eval.rouge import Rouge
#from eval.meteor import Meteor

def calc_text_similarity(ref, hpy):
    ref = str(ref)
    hpy = str(hpy)
    #cc = SmoothingFunction()
    #bleu = nltk.translate.bleu_score.sentence_bleu([ref], hpy, smoothing_function=cc.method4) 
    bleu = compute_bleu([[ref.split()]], [hpy.split()], smooth=True)[0]
    rouge_score = Rouge().calc_score(hpy.split(), [ref.split()])
    meteor = calculate_meteor(ref, hpy)
    # meteor = 0
    return bleu, rouge_score, meteor



# ----------------------------
# Configuration
# ----------------------------
MODEL_TYPE_ENC = "codebert"
MODEL_TYPE_DEC = "codet5p-220m"  # "codet5", "codet5p-220m", "codet5p-770m", or "codebert"
ABLATION_TAG = get_ablation_tag()

nearest_examples = nearest_examples_filename("test", MODEL_TYPE_ENC, "all", ABLATION_TAG)
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

NUM_EXAMPLES = _env_int("IDAE_NUM_EXAMPLES", 3)
MAX_LEN = _env_int("IDAE_MAX_LEN", 512)
BATCH_SIZE = _env_int("IDAE_EVAL_BATCH_SIZE", 128)
_seed_raw = os.getenv("IDAE_SEED")
SEED = int(_seed_raw) if _seed_raw not in (None, "") else None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Dataset
# ----------------------------
class CodeT5Dataset(Dataset):
    def __init__(self, examples, id2nearest, tokenizer, max_len=MAX_LEN, num_examples=NUM_EXAMPLES):
        self.examples = examples
        self.id2nearest = id2nearest
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_examples = num_examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        code = ex["raw_code"]
        target = ex["comment"]
        label = ex["label"]
        nearest_list = self.id2nearest.get(ex["id"], [])
        examples_text = ""
        for n in nearest_list[:self.num_examples]:
            examples_text += f"<code>{n['raw_code']}</code> => <comment>{n['comment']}</comment>\n"
            #examples_text += f"<comment>{n['comment']}</comment>\n"
        #input_text = f"{code}\nSimilar Examples:\n{examples_text}"
        input_text = f"{code}\nIntention:\n{label}\nSimilar Examples:\n{examples_text}"
        return input_text, target, label

def collate_fn(batch, tokenizer):
    inputs, targets, labels_raw = zip(*batch)
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
    return enc, labels, list(labels_raw)

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

# ----------------------------
# Evaluation function
# ----------------------------

def evaluate(model, tokenizer, eval_data, id2nearest, save_path="predictions.jsonl",
             label_list=None, dataset_name="dataset"):
    model.eval()
    dataset = CodeT5Dataset(eval_data, id2nearest, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda x: collate_fn(x, tokenizer))

    total_bleu, total_rouge, total_meteor, count = 0.0, 0.0, 0.0, 0
    label_stats = {}
    if label_list:
        for lab in label_list:
            label_stats[lab] = {"bleu": 0.0, "rouge": 0.0, "meteor": 0.0, "count": 0}

    # Open the output file.
    fw = open(save_path, "w", encoding="utf-8")

    with torch.no_grad():
        for batch_enc, labels, raw_labels in tqdm(loader, desc="Evaluating"):
            batch_enc = {k: v.to(DEVICE) for k, v in batch_enc.items()}

            generator = model.module if isinstance(model, nn.DataParallel) else model
            outputs = generator.generate(
                input_ids=batch_enc["input_ids"],
                attention_mask=batch_enc["attention_mask"],
                max_length=MAX_LEN
            )

            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            targets = tokenizer.batch_decode(
                torch.where(labels == -100, tokenizer.pad_token_id, labels),
                skip_special_tokens=True
            )

            for p, t, lab in zip(preds, targets, raw_labels):
                bleu, rouge, meteor = calc_text_similarity(t, p)
                total_bleu += bleu
                total_rouge += rouge
                total_meteor += meteor
                count += 1
                if lab not in label_stats:
                    label_stats[lab] = {"bleu": 0.0, "rouge": 0.0, "meteor": 0.0, "count": 0}
                label_stats[lab]["bleu"] += bleu
                label_stats[lab]["rouge"] += rouge
                label_stats[lab]["meteor"] += meteor
                label_stats[lab]["count"] += 1

                # Write one prediction record.
                fw.write(json.dumps({
                    "prediction": p,
                    "ground_truth": t,
                    "label": lab,
                    "bleu": bleu,
                    "rougeL": rouge,
                    "meteor": meteor
                }, ensure_ascii=False) + "\n")

    fw.close()  # Close the output file.

    avg_bleu = total_bleu / count
    avg_rouge = total_rouge / count
    avg_meteor = total_meteor / count
    print(f"[{dataset_name}] BLEU: {avg_bleu:.4f}, ROUGE-L: {avg_rouge:.4f}, METEOR: {avg_meteor:.4f}")
    for lab in (label_list or sorted(label_stats.keys())):
        s = label_stats.get(lab, None)
        if not s or s["count"] == 0:
            print(f"[{dataset_name}] Label {lab}: no samples")
            continue
        lab_bleu = s["bleu"] / s["count"]
        lab_rouge = s["rouge"] / s["count"]
        lab_meteor = s["meteor"] / s["count"]
        print(
            f"[{dataset_name}] Label {lab}: BLEU {lab_bleu:.4f}, "
            f"ROUGE-L {lab_rouge:.4f}, METEOR {lab_meteor:.4f} (n={s['count']})"
        )

    return (avg_bleu + avg_rouge + avg_meteor) / 3


# ----------------------------
# Main function
# ----------------------------
def _model_path(dataset_name: str):
    return decoder_checkpoint_filename(
        "./checkpoints_dec",
        MODEL_TYPE_ENC,
        MODEL_TYPE_DEC,
        NUM_EXAMPLES,
        dataset_name,
        ABLATION_TAG,
        seed=SEED,
    )

def _nearest_examples_path(dataset_name: str):
    dataset_specific = nearest_examples_filename("test", MODEL_TYPE_ENC, dataset_name, ABLATION_TAG)
    if os.path.exists(dataset_specific):
        return dataset_specific
    return nearest_examples

def _load_label_list(id2nearest):
    label_set = set()
    for v in id2nearest.values():
        for item in v:
            lab = item.get("label")
            if lab is not None:
                label_set.add(lab)
    return sorted(label_set)

def _load_test_list(file_path: str):
    test_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            test_list.append(json.loads(line))
    return test_list


def _selected_test_files():
    dataset_files = {
        "tlcodesum": "../../data/tlcodesum.test",
        "funcom": "../../data/funcom.test",
    }
    raw = os.getenv("IDAE_DATASETS", "tlcodesum")
    names = [name.strip().lower() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("IDAE_DATASETS must contain at least one dataset")
    unknown = sorted(set(names) - set(dataset_files))
    if unknown:
        raise ValueError(
            f"Unsupported dataset(s): {', '.join(unknown)}. "
            f"Expected one of: {', '.join(dataset_files)}"
        )
    return [(dataset_files[name], name) for name in names]

def _load_model():
    tokenizer, model = build_model_and_tokenizer(MODEL_TYPE_DEC)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return tokenizer, model

def main():
    print(f"Ablation mode: {ABLATION_TAG}")
    test_files = _selected_test_files()
    for file_path, dataset_name in test_files:
        if NUM_EXAMPLES == 0:
            id2nearest = {}
        else:
            with open(_nearest_examples_path(dataset_name), "r", encoding="utf8") as f:
                id2nearest = json.load(f)
        label_list = _load_label_list(id2nearest)

        tokenizer, model = _load_model()
        state = torch.load(_model_path(dataset_name), map_location=DEVICE)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
        model.eval()

        test_list = _load_test_list(file_path)
        print(f"Test size ({dataset_name}): {len(test_list)}")

        save_path = os.path.join(
            "../../data",
            prediction_filename(NUM_EXAMPLES, dataset_name, ABLATION_TAG, seed=SEED),
        )
        evaluate(
            model,
            tokenizer,
            test_list,
            id2nearest,
            save_path=save_path,
            label_list=label_list,
            dataset_name=dataset_name,
        )

if __name__ == "__main__":
    main()


"""
Main experiment
CUDA_VISIBLE_DEVICES=1 IDAE_ABLATION=main python evaluate.py

CUDA_VISIBLE_DEVICES=2 IDAE_ABLATION=main IDAE_DATASETS=tlcodesum,funcom IDAE_NUM_EXAMPLES=5 IDAE_SEED=42 python -u evaluate.py

# Contrastive-learning experiment (formerly full)
IDAE_ABLATION=contrastive  python evaluate.py

# Random-exemplar experiment
IDAE_ABLATION=rand_ex  python evaluate.py

CUDA_VISIBLE_DEVICES=7 IDAE_ABLATION=main IDAE_DATASETS=tlcodesum,funcom IDAE_NUM_EXAMPLES=6 IDAE_SEED=42 python -u evaluate.py

# Ablation configurations
token
semantic
rand_ex
main
contrastive

"""
