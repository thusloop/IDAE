# IDAE

IDAE generates intent-aware code comments using retrieved examples and a
CodeT5+ 220M generator. The current main configuration is `main`: SimCSE
retrieval constrained to the same intent label. 

## Repository Layout

| Directory | Purpose | Documentation |
| --- | --- | --- |
| `IDAE/` | Main method, retrieval ablations, training, prediction, and metrics | [Experiment guide](IDAE/README.md) |
| `contrastive_retrieval_baseline/` | Independent bi-encoder retrieval and comment generation baseline | [Baseline guide](contrastive_retrieval_baseline/README.md) |
| `llm_evaluate/` | Paired LLM judging of IDAE and three comparison systems | [LLM evaluation guide](llm_evaluate/README.md) |
| `data/` | Processed datasets, predictions, and comparison workbooks supplied separately | See below |

Commands below use Bash on Linux, or a Linux environment such as WSL, and
start at the repository root unless a `cd` command is shown.

## Environment

Use Python 3.9+ with a CUDA-compatible PyTorch installation and FAISS-GPU for
the main retrieval pipeline. Install dependencies into that environment:

```bash
python -m pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4 punkt punkt_tab averaged_perceptron_tagger_eng
```

## Data and Models

Place the processed datasets in the repository-level `data/` directory:

```text
data/
  tlcodesum.train
  tlcodesum.valid
  tlcodesum.test
  funcom.train
  funcom.valid
  funcom.test
```

Each file contains one JSON object per line. The main loaders expect `id`,
`raw_code`, `comment`, and `label`. Intent labels are `what`, `why`, `usage`,
`property`, and `done`. Preserve the original test IDs and row order for
alignment with comparison workbooks and prediction files.

Dataset and reproduction files: [data download](https://drive.google.com/file/d/1t_9IAQiZNKS6BmQG5S4fNC9Z0bkktnGb/view?usp=sharing).

`IDAE/src/model_paths.py` prefers local model directories in `../model/`,
relative to the repository root, and falls back to Hugging Face model IDs:

| Local directory | Hugging Face fallback | Use |
| --- | --- | --- |
| `sup-simcse-roberta-base` | `princeton-nlp/sup-simcse-roberta-base` | Main retriever |
| `codet5p-220m` | `Salesforce/codet5p-220m` | Default generator |
| `codebert-base` | `microsoft/codebert-base` | Contrastive retriever |
| `st-codesearch-distilroberta-base` | `flax-sentence-embeddings/st-codesearch-distilroberta-base` | Semantic retrieval ablation |

The helper also supports `codet5-base` and `codet5p-770m`. Several retrieval
scripts set `HF_ENDPOINT` to `https://hf-mirror.com`; check that setting if
your environment needs the default Hugging Face endpoint.

## Main Pipeline

For a TLCodesum run with three exemplars and training seed 42:

```bash
IDAE_RETRIEVAL_DATASETS=tlcodesum \
IDAE_DATASETS=tlcodesum \
IDAE_SEED=42 \
IDAE_NUM_EXAMPLES=3 \
bash IDAE/run_ablation.sh main
```

The wrapper changes into `IDAE/src/` and runs:

1. `build_nearest_examples_train.py`: build training exemplars.
2. `build_nearest_examples_test.py`: build validation and test exemplars.
3. `codet5_dec.py`: train one generator per selected dataset.
4. `evaluate.py`: load matching checkpoints and write test predictions.

To process both datasets, set both dataset variables to `tlcodesum,funcom`.
`IDAE_RETRIEVAL_DATASETS` selects the retrieval corpus; `IDAE_DATASETS`
selects generator training and prediction datasets. Without explicit
settings, retrieval and training use both datasets, while prediction uses
only TLCodesum. Set `IDAE_SEED` consistently for training and prediction:
training defaults to seed 42, whereas prediction omits the seed suffix if
the variable is unset.

| Mode | Retrieval behavior |
| --- | --- |
| `main` | SimCSE, same-intent candidates; |
| `contrastive` | Trained contrastive encoder, same-intent candidates; |
| `rand_ex` | Random training exemplars |
| `token` | Common unique code-token ranking, without an intent constraint |
| `semantic` | Code-search embeddings and cosine similarity, without an intent constraint |

The `contrastive` mode requires `IDAE/src/checkpoints_enc/best_enc_codebert.pth`.
Prepare `data/positive_pool_sorted.json` and train this encoder from
`IDAE/src/` with `python codebert_enc.py` before running that mode.
The independent baseline has its own checkpoint format.

Important generator settings:

| Variable | Default | Meaning |
| --- | --- | --- |
| `IDAE_NUM_EXAMPLES` | `3` | Exemplars included in each input; `0` skips exemplar loading in training/prediction |
| `IDAE_SEED` | `42` for training | RNG seed and checkpoint/prediction filename suffix |
| `IDAE_EPOCHS` | `1` | Training epochs |
| `IDAE_BATCH_SIZE` | `32` | Training batch size |
| `IDAE_MAX_LEN` | `512` | Tokenized input and target length limit |
| `IDAE_VALIDATION_LIMIT` | `100` | Validation examples used during training |
| `IDAE_NUM_WORKERS` | `4` | Training data-loader workers |
| `IDAE_EVAL_BATCH_SIZE` | `128` | Prediction batch size |

Exemplar maps are written in `IDAE/src/`, generator checkpoints in
`IDAE/src/checkpoints_dec/`, and predictions in `data/`. Example filenames:

```text
nearest_examples_train_label_codebert.tlcodesum.main.json
best_enc_codebert__dec_codet5p-220m.3.tlcodesum.main.seed42.pth
predictions_label_num3_tlcodesum.main.seed42.jsonl
```

The `codebert` filename tag is retained for compatibility even when `main`
uses SimCSE. See the [experiment guide](IDAE/README.md) for retrieval pool
controls, caching, and approximate semantic search.

## Multiple Training Runs and Metrics

After building the exemplar files for the requested modes, preview or run
the training matrix from `IDAE/src/`:

```bash
cd IDAE/src
python run_seeded_experiments.py \
  --ablations main \
  --datasets tlcodesum,funcom \
  --seeds 42,13,100 \
  --num-examples 0,3,5 \
  --epochs 1 \
  --dry-run
```

Remove `--dry-run` to train. With no ablation setting, the launcher uses
`token,semantic,rand_ex`. It retries CUDA OOM failures with batch sizes
`32,16,8,4`, records logs and `summary.jsonl` in `experiment_logs/`, and
currently accepts only `--epochs 1`. It launches training only; generate
predictions separately for each configuration:

```bash
IDAE_ABLATION=main IDAE_DATASETS=tlcodesum,funcom \
IDAE_SEED=42 IDAE_NUM_EXAMPLES=3 python evaluate.py
```

Repeat with each trained seed and exemplar count. The launcher currently
logs every `IDAE_` environment variable, so run it in a shell without
`IDAE_LLM_API_KEY` set.

List matched prediction pairs or compute final metrics:

```bash
python cal_pre.py --list-groups --ablation main --num-examples 0,3,5 --seeds 42,13,100
python cal_pre.py --ablation main --num-examples 0,3,5 --seeds 42,13,100
```

Automatic discovery pairs FunCom and TLCodesum files with the same mode,
exemplar count, and seed. Missing or incomplete pairs are skipped. Reports
use a timestamped TXT filename in `data/`; `--output PATH` selects another
filename, and existing reports are never overwritten. For explicitly chosen
pairs, use `--group NAME FUNCOM_JSONL TLCODESUM_JSONL`. A completed report
ends with `END OF REPORT`.

## Independent Baseline

The independent runner trains a retriever, retrieves comments, trains a
generator, and evaluates it. Its current shared-metric imports still refer
to `compare_llm/src`, so expose `IDAE/src` through `PYTHONPATH`. Run this
example from the repository root:

```bash
PYTHONPATH="$PWD/IDAE/src${PYTHONPATH:+:$PYTHONPATH}" \
bash contrastive_retrieval_baseline/run_all.sh \
  --dataset tlcodesum --retriever codebert --top-k 3
```

The shell runner defaults to `codebert`, while its Python entry points
default to `simcse`; specify `--retriever` explicitly. Predictions default
to the baseline directory. Use the standalone evaluator's `--output` option
to place them in `data/` for aggregation. See the [baseline guide](contrastive_retrieval_baseline/README.md)
for stage selection and output filenames.

## LLM Evaluation

The judge compares `ours_simcse_codet5`, `dome`,
`baseline_3shot_comment2code`, and `cot_3shot_comment2code` on the same sampled
test examples. It scores accuracy, adequacy, and naturalness from 1 to 5;
the requested overall score is their arithmetic mean rounded to two decimals.

Besides the `.test` files, the default inputs are:

- `data/funcom-test.xlsx` and `data/tlcodesum-test.xlsx`.
- `data/predictions_label_num3_funcom.main.seed42.jsonl`.
- `data/predictions_label_num3_tlcodesum.main.seed13.jsonl`.

From the repository root, preview request preparation without API calls:

```bash
export IDAE_LLM_MODEL="your-judge-model"
bash llm_evaluate/run_eval_pipeline.sh \
  --seeds 54 --sample-size 150 --prepare-only
```

The default sample contains 150 examples per dataset and four systems:
1,200 API requests for two datasets, per sampling seed. `--seeds` controls
evaluation sampling, independently of the model training seeds above.
Full runs require `IDAE_LLM_API_KEY` and an OpenAI-compatible
`IDAE_LLM_API_BASE`. The [LLM evaluation guide](llm_evaluate/README.md)
describes credential setup, pilot runs, resumable requests, custom prediction
inputs, and summary files.

## Diagnostics

Run these tools from `IDAE/src/` after preparing the required datasets and
exemplar files:

```bash
python measure_input_truncation.py \
  --ablation main --datasets tlcodesum --splits train,valid,test \
  --num-examples 0,3,5 --max-len 512 \
  --output truncation_reports/main.tlcodesum.json

CUDA_VISIBLE_DEVICES=0 python benchmark_simcse_retrieval.py \
  --dataset tlcodesum --output retrieval_reports/tlcodesum.json
```

The truncation tool writes JSON and TXT reports for code, intent, and
exemplar loss. The retrieval benchmark requires Linux and exactly one
visible GPU, and measures SimCSE encoding and FAISS query latency without
changing checkpoints or prediction files.
