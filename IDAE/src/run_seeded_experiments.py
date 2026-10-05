#!/usr/bin/env python3
"""Run the decoder training matrix for multiple ablations, seeds and examples.

Each trial is an independent subprocess. If it exits with a CUDA OOM, the
same trial is restarted with the next smaller batch size. Logs and a JSONL
summary are written under ``experiment_logs`` so failed attempts remain
available for inspection.

With no ablation option, the script runs the three retrieval ablations needed
for the reviewer experiment: ``token``, ``semantic`` and ``rand_ex``. Use
``--ablation`` to run one method, or ``--ablations`` to provide a comma-
separated list.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from ablation_utils import nearest_examples_filename


DEFAULT_SEEDS = (42, 13, 100)
DEFAULT_NUM_EXAMPLES = (0, 3, 5)
DEFAULT_BATCH_SIZES = (32, 16, 8, 4)
DEFAULT_ABLATIONS = ("token", "semantic", "rand_ex")
SUPPORTED_ABLATIONS = ("main", "contrastive", "rand_ex", "token", "semantic")
OOM_RE = re.compile(
    r"out of memory|outofmemory|cuda.*memory|cublas_status_alloc_failed|" \
    r"cudnn_status_alloc_failed|cuda error:.*alloc|\boom\b",
    re.IGNORECASE,
)


def _parse_int_list(value, name):
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc
    if not result:
        raise ValueError(f"{name} must contain at least one integer")
    return result


def _parse_ablation_list(value, name="--ablations"):
    """Parse and validate a comma-separated ablation list."""
    result = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not result:
        raise ValueError(f"{name} must contain at least one ablation")
    unsupported = sorted(set(result) - set(SUPPORTED_ABLATIONS))
    if unsupported:
        choices = ", ".join(SUPPORTED_ABLATIONS)
        raise ValueError(
            f"Unsupported ablation(s): {', '.join(unsupported)}. "
            f"Expected one of: {choices}"
        )
    # Avoid accidentally running the same configuration twice when a list
    # contains duplicate names.
    return tuple(dict.fromkeys(result))


def _default_list(env_name, fallback):
    value = os.getenv(env_name)
    return ",".join(str(item) for item in fallback) if not value else value


def _dataset_names(value):
    names = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not names:
        raise ValueError("--datasets must contain at least one dataset")
    supported = {"tlcodesum", "funcom"}
    unknown = sorted(set(names) - supported)
    if unknown:
        raise ValueError(
            f"Unsupported dataset(s): {', '.join(unknown)}. "
            "Expected tlcodesum and/or funcom"
        )
    return tuple(dict.fromkeys(names))


def _check_retrieval_files(script_dir, datasets, ablations, num_examples_values):
    """Fail early when a retrieval ablation has no training exemplar file.

    ``codet5_dec.py`` first looks for a dataset-specific file and then falls
    back to the ``all`` file.  Check the same two locations before launching a
    potentially long training process.  NUM_EXAMPLES=0 does not require a
    retrieval file because the decoder explicitly skips loading it.
    """
    if not any(value > 0 for value in num_examples_values):
        return

    missing = []
    for ablation in ablations:
        if ablation not in SUPPORTED_ABLATIONS:
            continue
        for dataset in datasets:
            dataset_path = script_dir / nearest_examples_filename(
                "train", "codebert", dataset, ablation
            )
            all_path = script_dir / nearest_examples_filename(
                "train", "codebert", "all", ablation
            )
            if not dataset_path.is_file() and not all_path.is_file():
                missing.append(
                    f"{ablation}/{dataset}: {dataset_path.name} "
                    f"or {all_path.name}"
                )

    if missing:
        details = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Retrieval exemplar files are missing. Build the retrieval files "
            "before training, or use --skip-retrieval-check to bypass this "
            "preflight check:\n  - " + details
        )


def _run_attempt(script_dir, env, log_path):
    command = [sys.executable, "-u", "codet5_dec.py"]
    start = time.monotonic()
    saw_oom = False
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        log_file.write("Environment overrides:\n")
        for key in sorted(env):
            if key.startswith("IDAE_") or key in {
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            }:
                log_file.write(f"{key}={env[key]}\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=script_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
            if OOM_RE.search(line):
                saw_oom = True
        return_code = process.wait()

    return {
        "return_code": return_code,
        "oom": saw_oom,
        "duration_seconds": round(time.monotonic() - start, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        default=_default_list("IDAE_SEEDS", DEFAULT_SEEDS),
        help="Comma-separated random seeds (default: 42,13,100)",
    )
    parser.add_argument(
        "--num-examples",
        default=_default_list("IDAE_NUM_EXAMPLES_LIST", DEFAULT_NUM_EXAMPLES),
        help="Comma-separated exemplar counts (default: 0,3,5)",
    )
    parser.add_argument(
        "--batch-sizes",
        default=_default_list("IDAE_BATCH_SIZES", DEFAULT_BATCH_SIZES),
        help="Batch sizes tried in order after OOM (default: 32,16,8,4)",
    )
    parser.add_argument(
        "--datasets",
        default=os.getenv("IDAE_DATASETS", "tlcodesum,funcom"),
        help="Datasets passed to codet5_dec.py (default: tlcodesum,funcom)",
    )
    parser.add_argument(
        "--ablation",
        choices=SUPPORTED_ABLATIONS,
        default=None,
        help=(
            "Run one ablation (backward-compatible shortcut). If omitted, "
            "use --ablations or the default token,semantic,rand_ex."
        ),
    )
    parser.add_argument(
        "--ablations",
        default=None,
        help=(
            "Comma-separated ablations to run when --ablation is not given "
            "(default: token,semantic,rand_ex)"
        ),
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value; otherwise inherit the shell",
    )
    parser.add_argument("--max-len", type=int, default=int(os.getenv("IDAE_MAX_LEN", 512)))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=int(os.getenv("IDAE_VALIDATION_LIMIT", 100)),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("IDAE_NUM_WORKERS", 4)),
    )
    parser.add_argument(
        "--log-dir",
        default=os.getenv("IDAE_EXPERIMENT_LOG_DIR", "experiment_logs"),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the matrix after a non-OOM trial failure",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the trial matrix without starting training",
    )
    parser.add_argument(
        "--skip-retrieval-check",
        action="store_true",
        help="Do not verify token/semantic/random training exemplar files before launch",
    )
    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds, "--seeds")
    num_examples_values = _parse_int_list(args.num_examples, "--num-examples")
    batch_sizes = _parse_int_list(args.batch_sizes, "--batch-sizes")
    if args.ablation:
        # Explicit --ablation has the highest priority.
        ablations = (args.ablation,)
    elif args.ablations is not None:
        # Explicit --ablations must override a stale IDAE_ABLATION in the
        # parent shell.
        ablations = _parse_ablation_list(args.ablations)
    elif os.getenv("IDAE_ABLATION"):
        # Preserve the old one-method environment-variable interface.
        ablations = _parse_ablation_list(
            os.environ["IDAE_ABLATION"], name="IDAE_ABLATION"
        )
    else:
        ablations = _parse_ablation_list(
            _default_list("IDAE_ABLATIONS", DEFAULT_ABLATIONS)
        )
    if any(size <= 0 for size in batch_sizes):
        raise ValueError("batch sizes must be positive")
    if args.epochs != 1:
        raise ValueError("This reviewer experiment requires --epochs 1")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.validation_limit < 0:
        raise ValueError("--validation-limit must be non-negative")
    datasets = _dataset_names(args.datasets)

    script_dir = Path(__file__).resolve().parent
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = script_dir / log_dir
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "summary.jsonl"

    if not args.dry_run and not args.skip_retrieval_check:
        _check_retrieval_files(
            script_dir=script_dir,
            datasets=datasets,
            ablations=ablations,
            num_examples_values=num_examples_values,
        )

    base_env = os.environ.copy()
    base_env.update({
        "IDAE_DATASETS": args.datasets,
        "IDAE_EPOCHS": str(args.epochs),
        "IDAE_MAX_LEN": str(args.max_len),
        "IDAE_NUM_WORKERS": str(args.workers),
        "IDAE_VALIDATION_LIMIT": str(args.validation_limit),
        "PYTHONUNBUFFERED": "1",
    })
    if args.gpus is not None:
        base_env["CUDA_VISIBLE_DEVICES"] = args.gpus
    base_env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    trials = [
        (ablation, seed, num_examples)
        for ablation in ablations
        for seed in seeds
        for num_examples in num_examples_values
    ]
    print(f"Planned trials: {len(trials)}")
    print(f"Ablations: {ablations}")
    print(f"Seeds: {seeds}; NUM_EXAMPLES: {num_examples_values}")
    print(f"Datasets: {args.datasets}; batch retry sequence: {batch_sizes}")
    print(f"Summary: {summary_path}")

    if args.dry_run:
        for trial_index, (ablation, seed, num_examples) in enumerate(trials, start=1):
            print(
                f"[{trial_index}/{len(trials)}] ablation={ablation}, seed={seed}, "
                f"NUM_EXAMPLES={num_examples}, epochs={args.epochs}"
            )
        return

    with summary_path.open("a", encoding="utf-8") as summary_file:
        for trial_index, (ablation, seed, num_examples) in enumerate(trials, start=1):
            print(
                f"\n[{trial_index}/{len(trials)}] ablation={ablation}, seed={seed}, "
                f"NUM_EXAMPLES={num_examples}",
                flush=True,
            )
            trial_done = False
            last_result = None
            for attempt, batch_size in enumerate(batch_sizes, start=1):
                env = base_env.copy()
                env.update({
                    "IDAE_ABLATION": ablation,
                    "IDAE_SEED": str(seed),
                    "IDAE_NUM_EXAMPLES": str(num_examples),
                    "IDAE_BATCH_SIZE": str(batch_size),
                })
                log_path = log_dir / (
                    f"{ablation}.seed{seed}.num_examples{num_examples}."
                    f"attempt{attempt}.batch{batch_size}.log"
                )
                print(f"Starting attempt {attempt}: batch_size={batch_size}", flush=True)
                result = _run_attempt(script_dir, env, log_path)
                record = {
                    "ablation": ablation,
                    "seed": seed,
                    "num_examples": num_examples,
                    "batch_size": batch_size,
                    "attempt": attempt,
                    "log": str(log_path),
                    **result,
                }
                summary_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary_file.flush()
                last_result = record

                if result["return_code"] == 0:
                    print("Trial completed successfully", flush=True)
                    trial_done = True
                    break
                if result["oom"] and attempt < len(batch_sizes):
                    next_batch = batch_sizes[attempt]
                    print(
                        f"CUDA OOM detected; retrying with batch_size={next_batch}",
                        flush=True,
                    )
                    time.sleep(5)
                    continue
                print(
                    f"Trial failed with return code {result['return_code']}; "
                    f"see {log_path}",
                    flush=True,
                )
                break

            if not trial_done and args.stop_on_error:
                raise SystemExit(
                    f"Stopping after failed trial: {last_result['log'] if last_result else 'unknown'}"
                )


if __name__ == "__main__":
    main()

