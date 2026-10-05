# IDAE Experiments

Main comment-generation experiments and retrieval ablations. See the
[root README](../README.md) for dependency installation, datasets, and local
model lookup. Commands invoking Python directly below run from `IDAE/src/`;
commands invoking `run_ablation.sh` run from `IDAE/`.

## Experiment modes

- `main`: It uses the SimCSE retriever with a same-intent constraint.
- `contrastive`: It uses the trained contrastive retrieval encoder.
- `rand_ex`: the random-exemplar baseline.
- `token`: the FSMIC token-based retrieval method with 10 exemplars from the global training corpus, without an intent constraint.
- `semantic`: the FSMIC semantic-based retrieval method with `st-codesearch-distilroberta-base` and 10 exemplars from the global training corpus.

The retrieval builders save up to 10 exemplars per item. The generator uses
`IDAE_NUM_EXAMPLES` of them (default: 3).

## Main pipeline

From the repository root:

```bash
IDAE_RETRIEVAL_DATASETS=tlcodesum \
IDAE_DATASETS=tlcodesum \
IDAE_SEED=42 \
IDAE_NUM_EXAMPLES=3 \
bash IDAE/run_ablation.sh main
```

The wrapper builds train and valid/test exemplars, trains the generator,
then generates predictions. Set both dataset variables to `tlcodesum,funcom`
to train and evaluate both datasets. Keep `IDAE_SEED` and
`IDAE_NUM_EXAMPLES` consistent when invoking training and evaluation
separately. The `contrastive` mode first requires the main encoder checkpoint
`src/checkpoints_enc/best_enc_codebert.pth`, produced by `codebert_enc.py`
using `data/positive_pool_sorted.json` at the repository root.

## FSMIC retrieval ablations

```bash
cd IDAE
export IDAE_SEED=42
export IDAE_DATASETS=tlcodesum,funcom
bash run_ablation.sh token
# or
bash run_ablation.sh semantic
```

After the retrieval files are ready, one launcher can run the `token`, `semantic`, and `rand_ex` ablations in sequence. The default configuration uses three random seeds and `NUM_EXAMPLES=0,3,5`, for a total of `3 x 3 x 3 = 27` training configurations, each covering the selected datasets:

```bash
cd src
python run_seeded_experiments.py \
  --datasets funcom \
  --seeds 42,13,100 \
  --num-examples 0,3,5 \
  --epochs 1
```

To run one ablation only, use `--ablation`:

```bash
python run_seeded_experiments.py \
  --ablation token \
  --datasets funcom \
  --seeds 42,13,100 \
  --num-examples 0,3,5 \
  --epochs 1
```

Use `--ablations` to specify multiple methods:

```bash
python run_seeded_experiments.py \
  --ablations token,semantic,rand_ex \
  --datasets funcom \
  --seeds 42,13,100 \
  --num-examples 0,3,5 \
  --epochs 1
```

`IDAE_ABLATIONS=token,semantic,rand_ex` is equivalent to `--ablations`. For backward compatibility, `IDAE_ABLATION=token python run_seeded_experiments.py` still runs only the token ablation.

Add `--dry-run` to preview the matrix without training. The launcher accepts
only `--epochs 1`, checks that training exemplar files exist unless
`--skip-retrieval-check` is used, and launches training only. CUDA OOM
failures retry with the next batch size in `--batch-sizes` (default:
`32,16,8,4`); other failures are recorded without a smaller-batch retry.
Each attempt creates a log in `experiment_logs/` containing the ablation
name, seed, exemplar count, and batch size; the same fields are written to
`summary.jsonl`.

After training, generate predictions separately for every trained seed,
mode, and exemplar count, for example:

```bash
IDAE_ABLATION=token IDAE_DATASETS=funcom \
IDAE_SEED=42 IDAE_NUM_EXAMPLES=3 python evaluate.py
```

The launcher writes all `IDAE_` environment variables to its logs. Unset
`IDAE_LLM_API_KEY` in the training shell before running it.

For example, run all ablations in the background:

```bash
mkdir -p experiment_logs
nohup env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=0,3,5 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false \
  python -u run_seeded_experiments.py \
    --datasets funcom \
    --ablations token,semantic,rand_ex \
    --seeds 42,13,100 \
    --num-examples 0,3,5 \
    --batch-sizes 32,16,8,4 \
    --epochs 1 \
    --max-len 512 \
    --validation-limit 2000 \
    --workers 4 \
  > experiment_logs/all_ablations_launcher.log 2>&1 < /dev/null &
echo $! > experiment_logs/all_ablations_launcher.pid
```

The token, semantic, and random-exemplar modes write exemplar files with `.token.json`, `.semantic.json`, and `.rand_ex.json` suffixes, respectively. Training and evaluation select the corresponding files from `IDAE_ABLATION`; `cal_pre.py` uses `--ablation`. Semantic retrieval prefers `../model/st-codesearch-distilroberta-base` relative to the repository root and falls back to Hugging Face. Set `IDAE_RETRIEVAL_DEVICE=cpu` to force CPU semantic encoding.

The retrieval builders combine the complete TLCodesum and FunCom training sets by default and construct exemplars for every training sample. The FunCom training set contains about 1.18 million records, so full train-to-train token/semantic retrieval can take a long time. The original FSMIC scripts use a much smaller retrieval corpus. Token retrieval now uses batched sparse-matrix computation, but it is still recommended to validate the pipeline on TLCodesum or a reduced retrieval pool before starting a full run.

Select the retrieval corpus with `IDAE_RETRIEVAL_DATASETS`; for example, run only TLCodesum first:

```bash
cd ..
IDAE_RETRIEVAL_DATASETS=tlcodesum \
IDAE_DATASETS=tlcodesum \
IDAE_SEED=42 \
bash run_ablation.sh token
```

Token matrices and semantic training vectors are cached in `src/retrieval_cache/` and reused on later runs for the same dataset. Adjust the token batch size as follows:

```bash
cd src
IDAE_TOKEN_BATCH_SIZE=512 \
OMP_NUM_THREADS=16 \
MKL_NUM_THREADS=16 \
IDAE_ABLATION=token \
IDAE_RETRIEVAL_DATASETS=tlcodesum \
python build_nearest_examples_train.py
```

Semantic retrieval uses the exact `IndexFlatIP` index by default. To prioritize speed, explicitly select an approximate HNSW index:

```bash
IDAE_SEMANTIC_INDEX=hnsw \
IDAE_SEMANTIC_HNSW_EF_SEARCH=64 \
IDAE_ABLATION=semantic \
IDAE_RETRIEVAL_DATASETS=funcom \
python build_nearest_examples_train.py
```

`hnsw` can change a small number of Top-k exemplars. The paper must state that approximate retrieval was used. Keep the default `flat` mode to reproduce FSMIC's exact ranking.

To make token and semantic retrieval use the same approximate candidate pool, configure a fixed stratified retrieval pool. The following example uses 50,000 training samples:

```bash
export IDAE_RETRIEVAL_POOL_SIZE=50000
export IDAE_RETRIEVAL_POOL_SEED=42

IDAE_ABLATION=token \
IDAE_RETRIEVAL_DATASETS=funcom \
python build_nearest_examples_train.py

IDAE_ABLATION=semantic \
IDAE_RETRIEVAL_DATASETS=funcom \
python build_nearest_examples_train.py
```

The pool is sampled by `label` with a fixed random seed. Both methods must use the same `IDAE_RETRIEVAL_POOL_SIZE` and `IDAE_RETRIEVAL_POOL_SEED`. This changes the Top-k exemplars, but is much faster than exact train-to-train retrieval over the 1.18-million-record FunCom training set. The paper should report the candidate-pool size and random seed.

The `wo_lc` mode has been removed from the experiment workflow.

## Prediction aggregation

The following evaluator runs from `IDAE/src/` and requires NLTK resources,
Java, and the METEOR 1.5 data directory described in the root README. Final
reports use Java METEOR; training and prediction use NLTK METEOR instead.

```bash
python cal_pre.py --list-groups --ablation main --num-examples 0,3,5 --seeds 42,13,100
python cal_pre.py --ablation main --num-examples 0,3,5 --seeds 42,13,100 --output ../../data/metrics_main.txt
```

The script pairs FunCom and TLCodesum predictions by ablation mode, exemplar count, and random seed, then reports BLEU, Rouge-L, and METEOR for each dataset and for the combined data. Automatic discovery skips missing or incomplete prediction files and lists skipped configurations in the TXT report. Use `--group NAME FUNCOM_JSONL TLCODESUM_JSONL` repeatedly to provide custom file groups; paths are resolved from the current working directory. Results are written to TXT, and existing output files are never overwritten. The report is complete only when it ends with `END OF REPORT`.

## Exemplar files

- `nearest_examples_train_label_codebert.{dataset}.{mode}.json`: training exemplar maps.
- `nearest_examples_valid_label_codebert.{dataset}.{mode}.json`: validation exemplar maps.
- `nearest_examples_test_label_codebert.{dataset}.{mode}.json`: test exemplar maps.

`{dataset}` is `tlcodesum`, `funcom`, or `all` for a combined retrieval
corpus. The `contrastive` mode omits `.{mode}` from exemplar and checkpoint
filenames. The `codebert` tag is also used by `main` with SimCSE.
Training and prediction first try the dataset-specific exemplar file and
then fall back to the `all` file.

Generator checkpoints are stored in `src/checkpoints_dec/`. Predictions
are written to the repository-level `data/` directory as
`predictions_label_num{M}_{dataset}.{mode}.seed{seed}.jsonl` when
`IDAE_SEED` is set. Without it, prediction filenames omit the seed suffix.

## Diagnostics

From `IDAE/src/`, measure encoder-input truncation:

```bash
python measure_input_truncation.py \
  --datasets tlcodesum --splits train,valid,test --ablation main \
  --num-examples 0,3,5 --max-len 512 \
  --output truncation_reports/main.tlcodesum.json
```

The tool reports record-level and token-level losses for target code, intent,
and exemplars in JSON and TXT. `--nearest-policy runtime` (the default)
matches the training/prediction loaders; `matching-split` uses each split's
own exemplar map. `--base-report PATH` reuses matching earlier measurements.
Existing output reports are never overwritten.

Benchmark SimCSE encoding and FAISS retrieval on Linux with one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_simcse_retrieval.py \
  --dataset tlcodesum --output retrieval_reports/tlcodesum.json
```

Use `--smoke-train-limit N` to benchmark against a reduced training corpus.
