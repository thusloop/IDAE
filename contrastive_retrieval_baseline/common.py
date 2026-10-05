import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_ROOT = WORKSPACE_ROOT / "model"
COMPARE_SRC_ROOT = PROJECT_ROOT / "compare_llm" / "src"
if str(COMPARE_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_SRC_ROOT))

from eval.bleu import compute_bleu
from eval.rouge import Rouge
from metric_utils import calculate_meteor


CODEBERT_NAME = "microsoft/codebert-base"
SIMCSE_NAME = "princeton-nlp/sup-simcse-roberta-base"
CODEBERT_PATH = MODEL_ROOT / "codebert-base"
SIMCSE_PATH = MODEL_ROOT / "sup-simcse-roberta-base"
CODET5P_220M_LOCAL_PATH = MODEL_ROOT / "codet5p-220m"
CODET5P_220M_PATH = str(CODET5P_220M_LOCAL_PATH) if CODET5P_220M_LOCAL_PATH.is_dir() else "Salesforce/codet5p-220m"

DATASET_FILES = {
    "tlcodesum": {
        "train": DATA_ROOT / "tlcodesum.train",
        "valid": DATA_ROOT / "tlcodesum.valid",
        "test": DATA_ROOT / "tlcodesum.test",
    },
    "funcom": {
        "train": DATA_ROOT / "funcom.train",
        "valid": DATA_ROOT / "funcom.valid",
        "test": DATA_ROOT / "funcom.test",
    },
}

RETRIEVER_MODEL_NAMES = {
    "codebert": CODEBERT_NAME,
    "simcse": SIMCSE_NAME,
}

RETRIEVER_LOCAL_PATHS = {
    "codebert": CODEBERT_PATH,
    "simcse": SIMCSE_PATH,
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_jsonlines(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data.append(json.loads(line))
    return data


def dump_json(data, path: Path):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset_split(dataset: str, split: str):
    return load_jsonlines(DATASET_FILES[dataset][split])


def calc_text_similarity(ref, hpy):
    ref = str(ref)
    hpy = str(hpy)
    if not ref.strip() or not hpy.strip():
        return 0.0, 0.0, 0.0
    bleu = compute_bleu([[ref.split()]], [hpy.split()], smooth=True)[0]
    rouge_score = Rouge().calc_score(hpy.split(), [ref.split()])
    meteor = calculate_meteor(ref, hpy)
    return bleu, rouge_score, meteor


def average_text_score(ref, hpy):
    bleu, rouge, meteor = calc_text_similarity(ref, hpy)
    return (bleu + rouge + meteor) / 3.0


def resolve_retriever_model_name(retriever: str, override_name: str = None) -> str:
    if override_name:
        return override_name
    local_path = RETRIEVER_LOCAL_PATHS[retriever]
    if local_path.is_dir():
        return str(local_path)
    return RETRIEVER_MODEL_NAMES[retriever]


def resolve_checkpoint_model_name(saved_model_name: str, retriever: str) -> str:
    default_name = RETRIEVER_MODEL_NAMES[retriever]
    if not saved_model_name or saved_model_name == default_name:
        return resolve_retriever_model_name(retriever)

    saved_path = Path(saved_model_name)
    is_path_like = saved_path.is_absolute() or saved_model_name.startswith(".")
    if is_path_like:
        return str(saved_path) if saved_path.is_dir() else resolve_retriever_model_name(retriever)
    return saved_model_name


def get_retriever_checkpoint_path(base_dir: Path, dataset: str, retriever: str) -> Path:
    return base_dir / f"best_retriever.{dataset}.{retriever}.pth"


def get_legacy_retriever_checkpoint_path(base_dir: Path, dataset: str) -> Path:
    return base_dir / f"best_retriever.{dataset}.pth"


def get_retrieved_comments_path(base_dir: Path, dataset: str, top_k: int, split: str, retriever: str) -> Path:
    return base_dir / f"retrieved_comments.{dataset}.{retriever}.top{top_k}.{split}.json"


def get_legacy_retrieved_comments_path(base_dir: Path, dataset: str, top_k: int, split: str) -> Path:
    return base_dir / f"retrieved_comments.{dataset}.top{top_k}.{split}.json"


def get_generator_checkpoint_path(base_dir: Path, dataset: str, top_k: int, retriever: str) -> Path:
    return base_dir / f"best_generator.{dataset}.{retriever}.top{top_k}.pth"


def get_legacy_generator_checkpoint_path(base_dir: Path, dataset: str, top_k: int) -> Path:
    return base_dir / f"best_generator.{dataset}.top{top_k}.pth"


def get_predictions_path(base_dir: Path, dataset: str, top_k: int, retriever: str) -> Path:
    return base_dir / f"predictions.{dataset}.{retriever}.top{top_k}.jsonl"


def get_legacy_predictions_path(base_dir: Path, dataset: str, top_k: int) -> Path:
    return base_dir / f"predictions.{dataset}.top{top_k}.jsonl"
