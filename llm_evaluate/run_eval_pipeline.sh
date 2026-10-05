#!/usr/bin/env bash
# Linux runner: prepare paired LLM judging prompts, send requests, summarize.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
python_bin="${IDAE_LLM_PYTHON:-python}"
runs_dir="${IDAE_LLM_RUNS_DIR:-$script_dir/runs}"
api_base="${IDAE_LLM_API_BASE:-}"
model="${IDAE_LLM_MODEL:-gpt-6-sol}"
seeds_arg=""
datasets_arg="tlcodesum,funcom"
sample_size=150
sleep_seconds=0.1
timeout=120
max_requests=0
concurrency="${IDAE_LLM_CONCURRENCY:-1}"
prepare_only=false

usage() {
    cat <<'HELP'
Usage: bash run_eval_pipeline.sh --seeds 54[,55] [options]

Options:
  --datasets tlcodesum,funcom  Dataset list (default: both)
  --sample-size N            Examples per dataset (default: 150)
  --model NAME               Judge model (default: gpt-6-sol)
  --api-base URL             OpenAI-compatible proxy base URL
  --runs-dir PATH            Runs root (default: ./runs)
  --python PATH              Python interpreter (default: python)
  --sleep-seconds N          Pause between API calls (default: 0.1)
  --concurrency N            Max in-flight API requests per method (default: 1)
  --timeout N                Per-request timeout in seconds (default: 120)
  --max-requests N           Limit API calls per system (0 = all)
  --prepare-only             Create prompts and request files; NEVER call API
  -h, --help                 Show this help

Full runs require IDAE_LLM_API_KEY and IDAE_LLM_API_BASE (or --api-base).
The --seeds flag controls EVALUATION SAMPLING, not model training seeds.
Re-running the same configuration reuses verified prompts/requests and
skips successful API IDs; failed IDs are retried. No old files are deleted.
HELP
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --prepare-only) prepare_only=true; shift ;;
        --seeds|--datasets|--sample-size|--model|--api-base|--runs-dir|--python|--sleep-seconds|--concurrency|--timeout|--max-requests)
            option="$1"
            (($# >= 2)) || die "Missing value for $option"
            case "$option" in
                --seeds) seeds_arg="$2" ;;
                --datasets) datasets_arg="$2" ;;
                --sample-size) sample_size="$2" ;;
                --model) model="$2" ;;
                --api-base) api_base="$2" ;;
                --runs-dir) runs_dir="$2" ;;
                --python) python_bin="$2" ;;
                --sleep-seconds) sleep_seconds="$2" ;;
                --concurrency) concurrency="$2" ;;
                --timeout) timeout="$2" ;;
                --max-requests) max_requests="$2" ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1 (see --help)" ;;
    esac
done

[[ -n "$seeds_arg" ]] || die "--seeds is required; choose an unused sampling seed, e.g. 54"
[[ "$sample_size" =~ ^[1-9][0-9]*$ ]] || die "--sample-size must be positive"
[[ "$timeout" =~ ^[1-9][0-9]*$ ]] || die "--timeout must be positive"
[[ "$max_requests" =~ ^(0|[1-9][0-9]*)$ ]] || die "--max-requests must be non-negative"
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || die "--concurrency must be at least 1"
[[ "$sleep_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--sleep-seconds must be non-negative"
[[ -n "$model" ]] || die "--model cannot be empty"
command -v "$python_bin" >/dev/null || die "Python not found: $python_bin"
command -v flock >/dev/null || die "flock (util-linux) is required to prevent duplicate API calls"

IFS=, read -r -a seeds <<< "$seeds_arg"
IFS=, read -r -a datasets <<< "$datasets_arg"
[[ "${#seeds[@]}" -gt 0 && "${#datasets[@]}" -gt 0 ]] || die "Empty seed or dataset list"
for seed in "${seeds[@]}"; do
    [[ "$seed" =~ ^(0|[1-9][0-9]*)$ ]] || die "Invalid seed: $seed"
done
for dataset in "${datasets[@]}"; do
    [[ "$dataset" == "funcom" || "$dataset" == "tlcodesum" ]] || die "Invalid dataset: $dataset"
done
[[ "$(printf '%s\n' "${seeds[@]}" | sort -u | wc -l)" -eq "${#seeds[@]}" ]] || die "Duplicate seed"
[[ "$(printf '%s\n' "${datasets[@]}" | sort -u | wc -l)" -eq "${#datasets[@]}" ]] || die "Duplicate dataset"

if [[ "$prepare_only" != true ]]; then
    [[ -n "${IDAE_LLM_API_KEY:-}" ]] || die "Set IDAE_LLM_API_KEY (never put keys in command arguments or files)"
    [[ "$api_base" == http://* || "$api_base" == https://* ]] || die "Set IDAE_LLM_API_BASE or --api-base to your API endpoint"
fi

dataset_tag="$(IFS=-; printf '%s' "${datasets[*]}")"
dataset_csv="$(IFS=,; printf '%s' "${datasets[*]}")"
mkdir -p -- "$runs_dir"
cd -- "$script_dir"

run_logged() {
    local log_file="$1"
    shift
    "$@" 2>&1 | tee -a -- "$log_file"
}

verify_existing_prompts() {
    "$python_bin" - "$1" "$2" "$3" "$4" "$dataset_csv" "$script_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_file, prompt_file, expected_size, expected_seed, dataset_list, script_dir = sys.argv[1:]
sys.path.insert(0, script_dir)
from generate_eval_prompts import DEFAULT_OURS, KUMIC_WORKBOOKS, PROMPT_TEMPLATE

manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
datasets = dataset_list.split(",")
expected = {
    "datasets": datasets,
    "sample_size_per_dataset": int(expected_size),
    "seed": int(expected_seed),
    "num_prompt_records": len(datasets) * int(expected_size) * 4,
    "systems": sorted(("ours_simcse_codet5", "cot_3shot_comment2code", "baseline_3shot_comment2code", "dome")),
    "prompt_template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest(),
    "ours_prediction_files": {ds: str(DEFAULT_OURS[ds].resolve()) for ds in datasets},
    "kumic_workbooks": {ds: str(KUMIC_WORKBOOKS[ds].resolve()) for ds in datasets},
}
for field, value in expected.items():
    if manifest.get(field) != value:
        raise SystemExit(f"Existing prompt manifest differs in {field}; choose a new --runs-dir/--seeds, do not mix runs")
with Path(prompt_file).open(encoding="utf-8") as stream:
    count = sum(bool(line.strip()) for line in stream)
if count != expected["num_prompt_records"]:
    raise SystemExit(f"Existing prompt file has {count} rows; expected {expected['num_prompt_records']}")
print(f"Reusing verified prompt file ({count} rows): {prompt_file}")
PY
}

verify_existing_requests() {
    "$python_bin" - "$1" "$2" "$model" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

prompt_file, batch_dir, expected_model = sys.argv[1:]
by_system = defaultdict(list)
with Path(prompt_file).open(encoding="utf-8") as stream:
    for line in stream:
        record = json.loads(line)
        by_system[record["system_name"]].append(record["prompt"])
for system, prompts in by_system.items():
    path = Path(batch_dir) / f"{system}.jsonl"
    with path.open(encoding="utf-8") as stream:
        requests = (json.loads(line) for line in stream if line.strip())
        for index, prompt in enumerate(prompts):
            request = next(requests, None)
            if request is None or request.get("custom_id") != str(index) or request.get("url") != "/v1/chat/completions" or request["body"].get("model") != expected_model or request["body"]["messages"][0].get("content") != prompt:
                raise SystemExit(f"Existing batch request differs at {path}, item {index}; use another --runs-dir/--seeds")
        if next(requests, None) is not None:
            raise SystemExit(f"Existing batch request has extra rows: {path}")
print(f"Reusing verified request files: {batch_dir}")
PY
}

verify_or_save_api_config() {
    "$python_bin" - "$1" "$2" "$model" "$3" <<'PY'
import json
import sys
from pathlib import Path

config_file, api_base, model, results_dir = sys.argv[1:]
path = Path(config_file)
expected = {"api_base": api_base.rstrip("/"), "model": model, "request_url": "/v1/chat/completions"}
if path.exists():
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise SystemExit(f"Existing judge configuration differs: {path}; choose a new --runs-dir/--seeds")
else:
    if any(Path(results_dir).glob("*.jsonl")):
        raise SystemExit(f"API results exist without judge configuration: {results_dir}; choose a new --runs-dir/--seeds")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(expected, stream, indent=2)
    print(f"Saved judge configuration (no credentials): {path}")
PY
}

check_complete() {
    "$python_bin" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

from summarize_overall_scores import summarize_file

batch_dir, output_dir = map(Path, sys.argv[1:])
incomplete = []
for source in sorted(batch_dir.glob("*.jsonl")):
    expected = {json.loads(line)["custom_id"] for line in source.open(encoding="utf-8") if line.strip()}
    output = output_dir / f"{source.stem}_success.jsonl"
    completed = {json.loads(line)["custom_id"] for line in output.open(encoding="utf-8") if line.strip()} if output.is_file() else set()
    summary = summarize_file(output) if output.is_file() else None
    missing = len(expected - completed)
    parse_failures = summary["parse_failures"] + summary["score_failures"] if summary else 0
    if missing or parse_failures:
        incomplete.append((source.stem, missing, parse_failures))
if incomplete:
    raise SystemExit(f"Incomplete/unparseable API evaluations: {incomplete}. Results were saved; rerun to retry HTTP errors or inspect parse_failures.txt")
print("All requests succeeded and returned parseable scores.")
PY
}

for seed in "${seeds[@]}"; do
    seed_root="$runs_dir/seed_$seed"
    prompts_dir="$seed_root/prompts"
    batch_dir="$seed_root/batch_requests"
    results_dir="$seed_root/api_results"
    summary_dir="$seed_root/summary"
    logs_dir="$seed_root/logs"
    mkdir -p -- "$prompts_dir" "$batch_dir" "$results_dir" "$summary_dir" "$logs_dir"

    exec {lock_fd}> "$seed_root/.pipeline.lock"
    flock -n "$lock_fd" || die "Another pipeline is already using $seed_root"
    stem="eval_prompts.$dataset_tag.n$sample_size.seed$seed"
    prompt_file="$prompts_dir/$stem.jsonl"
    manifest_file="$prompts_dir/$stem.manifest.json"
    preview_file="$prompts_dir/$stem.preview.csv"

    printf '=== Sampling seed %s: prompt generation ===\n' "$seed"
    if [[ -e "$prompt_file" || -e "$manifest_file" || -e "$preview_file" ]]; then
        [[ -f "$prompt_file" && -f "$manifest_file" && -f "$preview_file" ]] || die "Partial prompts in $prompts_dir: choose a fresh --runs-dir/--seeds"
        verify_existing_prompts "$manifest_file" "$prompt_file" "$sample_size" "$seed"
    else
        run_logged "$logs_dir/generate_prompts.log" "$python_bin" -u generate_eval_prompts.py \
            --datasets "${datasets[@]}" --sample-size "$sample_size" --seed "$seed" --output-dir "$prompts_dir"
    fi

    batch_files=(
        "$batch_dir/ours_simcse_codet5.jsonl"
        "$batch_dir/cot_3shot_comment2code.jsonl"
        "$batch_dir/baseline_3shot_comment2code.jsonl"
        "$batch_dir/dome.jsonl"
    )
    existing=0
    for request_file in "${batch_files[@]}"; do
        if [[ -e "$request_file" ]]; then ((existing+=1)); fi
    done
    printf '=== Sampling seed %s: split requests ===\n' "$seed"
    if ((existing == 0)); then
        run_logged "$logs_dir/split_prompts.log" "$python_bin" -u split_prompts_to_batch_jsonl.py \
            --input "$prompt_file" --output-dir "$batch_dir" --model "$model"
    elif ((existing != ${#batch_files[@]})); then
        die "Partial batch request files in $batch_dir: choose a fresh --runs-dir/--seeds"
    fi
    verify_existing_requests "$prompt_file" "$batch_dir"

    if [[ "$prepare_only" == true ]]; then
        printf 'Prepared %s (NO API requests).\n' "$seed_root"
    else
        verify_or_save_api_config "$seed_root/judge_config.json" "$api_base" "$results_dir"
        printf '=== Sampling seed %s: call %s ===\n' "$seed" "$model"
        run_logged "$logs_dir/call_proxy_api.log" "$python_bin" -u call_proxy_api.py \
            "${batch_files[@]}" --api-base "$api_base" --model "$model" \
            --output-dir "$results_dir" --sleep-seconds "$sleep_seconds" \
            --timeout "$timeout" --max-requests "$max_requests" \
            --concurrency "$concurrency" --retry-errors

        success_files=()
        for request_file in "${batch_files[@]}"; do
            success_file="$results_dir/$(basename "${request_file%.jsonl}")_success.jsonl"
            if [[ -f "$success_file" ]]; then success_files+=("$success_file"); fi
        done
        ((${#success_files[@]} > 0)) || die "No successful API results in $results_dir"
        printf '=== Sampling seed %s: summarize ===\n' "$seed"
        "$python_bin" -u summarize_overall_scores.py "${success_files[@]}" \
            --parse-failure-output "$logs_dir/parse_failures.txt" \
            | tee "$summary_dir/summary.tsv" "$logs_dir/summary.log"
        if ((max_requests == 0)); then
            check_complete "$batch_dir" "$results_dir"
        else
            printf 'Pilot run (--max-requests %s): summary covers only available successes.\n' "$max_requests"
        fi
        printf 'Results: %s\n' "$summary_dir/summary.tsv"
    fi

    flock -u "$lock_fd"
    exec {lock_fd}>&-
done


# ./run_eval_pipeline.sh  --seeds 54 --sample-size 150  --concurrency 16 --sleep-seconds 0.1