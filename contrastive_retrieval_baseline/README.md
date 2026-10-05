# Contrastive Retrieval Baseline

This directory is an independent training pipeline with its own retriever
and generator checkpoints. See the [root README](../README.md) for datasets
and dependencies. Commands below run from this directory after this setup:

```bash
cd contrastive_retrieval_baseline
export PYTHONPATH="$PWD/../IDAE/src${PYTHONPATH:+:$PYTHONPATH}"
```

`common.py` still refers to the legacy `compare_llm/src` metric directory.
The `PYTHONPATH` setting makes the current `IDAE/src` metrics available.
Datasets are read from `data/` at the repository root.

- `train_retrieval_encoder.py`: train the bi-encoder retriever with in-batch contrastive loss.
- `build_retrieved_comments.py`: retrieve top-k comments from the train split.
- `train_comment_generator.py`: fine-tune CodeT5+ 220M with retrieved comments.
- `evaluate_comment_generator.py`: evaluate the generator on the test split.

## Retriever Backbones

The retrieval encoder supports two backbones. Models in the workspace-level
`model/` directory (next to the repository root) are preferred, with the Hugging Face
model ID used as a fallback:

- `codebert`: `codebert-base`, otherwise `microsoft/codebert-base`
- `simcse`: `sup-simcse-roberta-base`, otherwise `princeton-nlp/sup-simcse-roberta-base`

The shell runner defaults to `codebert`, both datasets, all stages, and
`--top-k 5`. Its Python entry points default to `simcse`; specify
`--retriever` explicitly to keep all stages consistent. The generator
prefers `../model/codet5p-220m` relative to the repository root, with
`Salesforce/codet5p-220m` as the fallback.

All retrieval-related artifacts are named with the retriever tag to avoid overwriting each other:

- `checkpoints/retrieval/best_retriever.{dataset}.{retriever}.pth`
- `artifacts/retrieved_comments/retrieved_comments.{dataset}.{retriever}.top{k}.{split}.json`
- `checkpoints/generator/best_generator.{dataset}.{retriever}.top{k}.pth`
- `predictions_label_num{k}_{dataset}.{ablation}.seed{seed}.jsonl`
- `predictions.{dataset}.{retriever}.top{k}.jsonl`: additional compatibility copy for CodeBERT.

## Examples

```bash

bash run_all.sh --dataset tlcodesum --retriever codebert --top-k 0
bash run_all.sh --dataset tlcodesum --retriever codebert --top-k 3
bash run_all.sh --dataset tlcodesum --retriever codebert --top-k 5

bash run_all.sh --dataset funcom --retriever codebert --top-k 0
bash run_all.sh --dataset funcom --retriever codebert --top-k 3
bash run_all.sh --dataset funcom --retriever codebert --top-k 5
```

Use SimCSE as the contrastive retriever:

```bash
bash run_all.sh --dataset tlcodesum --retriever simcse
bash run_all.sh --dataset funcom --retriever simcse --top-k 3
```

Run by stage:

```bash
bash run_all.sh --stage retriever --retriever simcse
bash run_all.sh --stage retrieve --retriever simcse
bash run_all.sh --stage generator --retriever simcse
bash run_all.sh --stage evaluate --retriever simcse
```

The standalone evaluator writes prediction files using the same naming
convention as `IDAE/src/cal_pre.py`. By default, `codebert` is labelled as
the `contrastive` ablation and `simcse` as `main`:

```bash
python evaluate_comment_generator.py \
  --dataset tlcodesum \
  --retriever codebert \
  --top-k 3 \
  --seed 42 \
  --checkpoint checkpoints/generator/best_generator.tlcodesum.codebert.top3.pth \
  --output ../data/predictions_label_num3_tlcodesum.contrastive.seed42.jsonl
```

The canonical default filename for this command is:

```text
predictions_label_num3_tlcodesum.contrastive.seed42.jsonl
```

For a SimCSE run, the default name is:

```text
predictions_label_num3_tlcodesum.main.seed42.jsonl
```

Use `--ablation` to override the label explicitly. For example,
`--ablation contrastive` produces the explicit `.contrastive` suffix shown
above, while `--output` can be used to place the JSONL in the shared
repository-level `data/` directory used by `cal_pre.py`. The old
`predictions.<dataset>.<retriever>.top<k>.jsonl` file is still written as a
CodeBERT compatibility copy.

The evaluator's `--seed` labels the prediction filename; it does not train
a new model. To change the training seed or isolate checkpoints for several
seeds, invoke `train_retrieval_encoder.py` and `train_comment_generator.py`
directly with `--seed` and `--output-dir`, then pass the appropriate
`--checkpoint` to retrieval and evaluation. The wrapper exposes only
dataset, stage, retriever, and exemplar-count options.
