import os

"""
Example commands:
                                                                                                                           
  $env:IDAE_ABLATION="contrastive"   
  export IDAE_ABLATION=contrastive                                                                                      
  python build_nearest_examples_train.py                                                                                   
  python build_nearest_examples_test.py                                                                                    
  python codet5_dec.py                                                                                                     
  python evaluate.py                                                                                                       
                                                                                                                           
  export IDAE_ABLATION=main                                                                                              
  python build_nearest_examples_train.py                                                                                   
  python build_nearest_examples_test.py                                                                                    
  python codet5_dec.py                                                                                                     
  python evaluate.py                                                                                                       
                                                                                                                           
  export IDAE_ABLATION=rand_ex
  python build_nearest_examples_train.py                                                                                   
  python build_nearest_examples_test.py                                                                                    
  python codet5_dec.py                                                                                                     
  python evaluate.py


"""
_ALIASES = {
    "": "main",
    "contrastive": "contrastive",
    "full": "contrastive",
    "with_cl": "contrastive",
    "idae": "main",
    "main": "main",
    "wo_cl": "main",
    "wocL": "main",
    "wocl": "main",
    "without_cl": "main",
    "no_cl": "main",
    "rand_ex": "rand_ex",
    "random": "rand_ex",
    "random_ex": "rand_ex",
    "random_example": "rand_ex",
    "random_examples": "rand_ex",
    "rand": "rand_ex",
    "token": "token",
    "token_based": "token",
    "token_based_retrieval": "token",
    "semantic": "semantic",
    "semantic_based": "semantic",
    "semantic_based_retrieval": "semantic",
}


def get_ablation_tag() -> str:
    raw = os.getenv("IDAE_ABLATION", "main").strip().lower()
    normalized = raw.replace("-", "_").replace("/", "_").replace(" ", "_")
    tag = _ALIASES.get(normalized)
    if tag is None:
        valid = ", ".join(sorted(set(_ALIASES.values())))
        raise ValueError(f"Unsupported IDAE_ABLATION={raw!r}. Expected one of: {valid}")
    return tag


def use_contrastive_encoder(ablation_tag: str) -> bool:
    return ablation_tag == "contrastive"


def use_random_exemplars(ablation_tag: str) -> bool:
    return ablation_tag == "rand_ex"


def use_fsmic_retrieval(ablation_tag: str) -> bool:
    """Whether to use one of FSMIC's standalone retrieval baselines."""
    return ablation_tag in {"token", "semantic"}


def fsmic_retrieval_method(ablation_tag: str):
    """Return ``token``/``semantic`` for the FSMIC baselines, otherwise None."""
    return ablation_tag if use_fsmic_retrieval(ablation_tag) else None


def nearest_examples_filename(split: str, model_type_enc: str, dataset_name: str, ablation_tag: str) -> str:
    if ablation_tag == "contrastive":
        return f"nearest_examples_{split}_label_{model_type_enc}.{dataset_name}.json"
    return f"nearest_examples_{split}_label_{model_type_enc}.{dataset_name}.{ablation_tag}.json"


def decoder_checkpoint_filename(
    checkpoint_dir: str,
    model_type_enc: str,
    model_type_dec: str,
    num_examples: int,
    dataset_name: str,
    ablation_tag: str,
    seed=None,
) -> str:
    seed_suffix = f".seed{seed}" if seed is not None else ""
    if ablation_tag == "contrastive":
        filename = f"best_enc_{model_type_enc}__dec_{model_type_dec}.{num_examples}.{dataset_name}{seed_suffix}.pth"
    else:
        filename = f"best_enc_{model_type_enc}__dec_{model_type_dec}.{num_examples}.{dataset_name}.{ablation_tag}{seed_suffix}.pth"
    return os.path.join(checkpoint_dir, filename)


def prediction_filename(num_examples: int, dataset_name: str, ablation_tag: str, seed=None) -> str:
    seed_suffix = f".seed{seed}" if seed is not None else ""
    return f"predictions_label_num{num_examples}_{dataset_name}.{ablation_tag}{seed_suffix}.jsonl"
