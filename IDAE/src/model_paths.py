from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = WORKSPACE_ROOT / "model"


def resolve_model_source(directory_name: str, hub_name: str) -> str:
    local_path = MODEL_ROOT / directory_name
    return str(local_path) if local_path.is_dir() else hub_name


CODEBERT_MODEL = resolve_model_source("codebert-base", "microsoft/codebert-base")
CODET5_MODEL = resolve_model_source("codet5-base", "Salesforce/codet5-base")
CODET5P_220M_MODEL = resolve_model_source("codet5p-220m", "Salesforce/codet5p-220m")
CODET5P_770M_MODEL = resolve_model_source("codet5p-770m", "Salesforce/codet5p-770m")
SIMCSE_MODEL = resolve_model_source(
    "sup-simcse-roberta-base",
    "princeton-nlp/sup-simcse-roberta-base",
)
CODESEARCH_MODEL = resolve_model_source(
    "st-codesearch-distilroberta-base",
    "flax-sentence-embeddings/st-codesearch-distilroberta-base",
)
