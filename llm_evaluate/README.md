# LLM Evaluation

This workflow judges generated comments for the same sampled code examples
across four systems: `ours_simcse_codet5`, `dome`,
`baseline_3shot_comment2code`, and `cot_3shot_comment2code`. The current prompt
scores accuracy, adequacy, and naturalness from 1 to 5 and requests their
arithmetic mean as `overall_score`, rounded to two decimals. Target intent
guides relevance rather than receiving a separate score.

Use Python 3.9+ and Bash on Linux with `flock` (`util-linux`). These scripts
use Python's standard library and do not require the training dependencies.
Commands below start at the repository root.

## Inputs

For each selected dataset, provide these files under repository-level `data/`:

| Dataset | Test file | Comparison workbook | Default IDAE predictions |
| --- | --- | --- | --- |
| FunCom | `funcom.test` | `funcom-test.xlsx` | `predictions_label_num3_funcom.main.seed42.jsonl` |
| TLCodesum | `tlcodesum.test` | `tlcodesum-test.xlsx` | `predictions_label_num3_tlcodesum.main.seed13.jsonl` |

Test files are JSONL with `id`, `raw_code`, `comment`, and `label`.
Prediction files must have one row per test example in the original test
order, with `ground_truth`, `label`, and `prediction`. The generator checks
row counts, reference comments, and labels before pairing the systems.

The first workbook sheet must have columns `ids`, `intent`, `ground_truth`,
`dome`, `baseline_3shot_comment2code`, and `cot_3shot_comment2code`.
Workbook IDs must match test IDs without duplicates, and intents must match
the corresponding test labels. The XLSX reader uses `zipfile` and XML, so
`pandas` and `openpyxl` are not required.

The Bash runner uses these prediction defaults. To select other prediction
files, pass `--ours-funcom PATH` and `--ours-tlcodesum PATH` to
`generate_eval_prompts.py` using the manual workflow below.

## Prepare Requests

Choose the judge model supported by your API provider and prepare requests
without making API calls:

```bash
export IDAE_LLM_MODEL="your-judge-model"
bash llm_evaluate/run_eval_pipeline.sh \
  --seeds 54 --sample-size 150 --prepare-only
```

`--seeds` is required and controls evaluation sampling, independently of
the training seeds in prediction filenames. Both datasets are selected by
default. Each system receives the same sampled examples, with at least one
sample from every available intent by default. A sample of 150 examples per
dataset yields 300 requests per system, or 1,200 total requests per sampling
seed. Preparing requests still requires the input files above.

## Call the Judge

In a separate evaluation shell, set your endpoint and read the key without
including its value in a command or source file:

```bash
export IDAE_LLM_API_BASE="https://api.example.com"
export IDAE_LLM_MODEL="your-judge-model"
read -rsp "API key: " IDAE_LLM_API_KEY
printf '\n'
export IDAE_LLM_API_KEY

bash llm_evaluate/run_eval_pipeline.sh \
  --seeds 54 --sample-size 150 --concurrency 4 --sleep-seconds 0.1
```

Replace the example endpoint and model with your provider's configuration.
The endpoint must serve OpenAI-compatible streaming chat completions at
`/v1/chat/completions`, relative to the configured base URL. Use the same
model when preparing and sending an existing run's requests.

For a small API pilot, add `--max-requests 5`; this limits calls per system.
Run again without that limit to finish the remaining requests. With
concurrent calls, `--sleep-seconds` controls submission pacing rather than
providing a strict rate limit.

The runner locks each sampling-seed directory, verifies existing prompts
and request files, skips successful IDs, and retries HTTP/request failures.
Changing the prompt, input paths, sample size, judge model, or API endpoint
can fail the reuse checks; use a fresh `--runs-dir` or sampling seed instead
of combining incompatible runs. Successful HTTP responses with unparseable
scores are reported for inspection and are not automatically regenerated.

## Options and Outputs

| Runner option | Default | Meaning |
| --- | --- | --- |
| `--datasets` | `tlcodesum,funcom` | Comma-separated dataset names |
| `--sample-size` | `150` | Sampled code examples per dataset |
| `--model` | `IDAE_LLM_MODEL` or `gpt-6-sol` | API judge model |
| `--api-base` | `IDAE_LLM_API_BASE` | API base URL |
| `--runs-dir` | `llm_evaluate/runs/` | Run output root; also configurable with `IDAE_LLM_RUNS_DIR` |
| `--python` | `IDAE_LLM_PYTHON` or `python` | Python interpreter |
| `--concurrency` | `IDAE_LLM_CONCURRENCY` or `1` | Maximum concurrent requests per system |
| `--timeout` | `120` | Per-request timeout in seconds |
| `--sleep-seconds` | `0.1` | Delay between submissions |
| `--max-requests` | `0` | Per-system request limit; `0` sends all remaining requests |
| `--prepare-only` | Off | Prepare prompts and requests without calling the API |

Each run writes:

```text
llm_evaluate/runs/seed_54/
  prompts/          # Prompt JSONL, sampling manifest, and CSV preview
  batch_requests/   # One chat-completion request JSONL per system
  api_results/      # *_success.jsonl and *_error.jsonl
  summary/          # summary.tsv
  logs/             # Stage logs and parse_failures.txt
  judge_config.json # Endpoint and judge model used for this run
```

`summary.tsv` reports the sample counts and averages for accuracy, adequacy,
naturalness, and overall score. Full runs check for missing IDs and invalid
scores after writing the summary; pilot summaries cover available successes
only. Review the counts and parse failures before using a summary.

Run outputs include API response content, the configured endpoint, and
absolute source paths in manifests. Review these files before publishing
them. Keep credentials out of the training shell: the current training
launcher records all `IDAE_` variables in its logs.

## Manual Workflow

From the repository root, enter this directory and generate prompts:

```bash
cd llm_evaluate
python generate_eval_prompts.py \
  --datasets tlcodesum funcom --sample-size 150 --seed 54 \
  --ours-funcom ../data/predictions_label_num3_funcom.main.seed42.jsonl \
  --ours-tlcodesum ../data/predictions_label_num3_tlcodesum.main.seed13.jsonl \
  --output-dir runs/manual_seed54/prompts

python split_prompts_to_batch_jsonl.py \
  --input runs/manual_seed54/prompts/eval_prompts.tlcodesum-funcom.n150.seed54.jsonl \
  --output-dir runs/manual_seed54/batch_requests \
  --model "$IDAE_LLM_MODEL"

python call_proxy_api.py runs/manual_seed54/batch_requests/*.jsonl \
  --api-base "$IDAE_LLM_API_BASE" --model "$IDAE_LLM_MODEL" \
  --output-dir runs/manual_seed54/api_results --retry-errors

python summarize_overall_scores.py \
  --root runs/manual_seed54/api_results \
  --parse-failure-output runs/manual_seed54/parse_failures.txt
```

The generator uses space-separated dataset arguments; the Bash runner uses
a comma-separated list. Set the evaluation environment variables as shown
above before running the manual API call. Explicitly select `--model` in
both request splitting and sending: the split script defaults to
`qwen3-max`, the API caller defaults to `gpt-5.4`, and the Bash runner has
its own default.

The API caller reads `IDAE_LLM_API_KEY` automatically. Successful IDs are
skipped on resume; `--retry-errors` retries failed IDs. `--overwrite`
replaces existing results and should be reserved for a deliberate restart.

To compare a shared subset of successful IDs within the same run:

```bash
python sample_compare_scores.py \
  --root runs/seed_54/api_results --sample-size 100 --seed 42
```

`single_prompt_request.py` is a standalone scratch client with placeholder
endpoint, key, and model constants. The environment-based pipeline above
is the documented evaluation workflow.
