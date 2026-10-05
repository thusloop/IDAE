import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Dict, List

from summarize_overall_scores import (
    METRIC_NAMES,
    extract_message_content,
    extract_metric_value,
    parse_content_json,
)


DEFAULT_GLOB = "*_success.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample shared custom_id values across result files and compare metric averages."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing *_success.jsonl files.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        required=True,
        help="Number of shared custom_id values to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save sampled custom_id values and summary as JSON.",
    )
    return parser.parse_args()


def load_metric_records(path: Path) -> Dict[str, Dict[str, float]]:
    records_by_custom_id: Dict[str, Dict[str, float]] = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            custom_id = str(record.get("custom_id", "")).strip()
            if not custom_id:
                continue

            content = extract_message_content(record)
            if not content:
                continue
            parsed = parse_content_json(content)
            if parsed is None:
                continue

            metric_values = {}
            missing_metric = False
            for metric_name in METRIC_NAMES:
                metric_value = extract_metric_value(parsed, metric_name)
                if metric_value is None:
                    missing_metric = True
                    break
                metric_values[metric_name] = metric_value

            if missing_metric:
                continue
            records_by_custom_id[custom_id] = metric_values

    return records_by_custom_id


def method_name_from_path(path: Path) -> str:
    return path.name.replace("_success.jsonl", "")


def discover_paths(root: Path) -> List[Path]:
    return sorted(root.glob(DEFAULT_GLOB))


def summarize_sampled_metrics(
    sampled_ids: List[str],
    metrics_by_method: Dict[str, Dict[str, Dict[str, float]]],
) -> List[dict]:
    rows = []
    for method, record_map in metrics_by_method.items():
        averages = {}
        for metric_name in METRIC_NAMES:
            values = [record_map[custom_id][metric_name] for custom_id in sampled_ids]
            averages[metric_name] = mean(values)
        rows.append(
            {
                "method": method,
                **averages,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    paths = discover_paths(args.root)
    if not paths:
        raise SystemExit(f"No *_success.jsonl files found under {args.root}")

    metrics_by_method = {
        method_name_from_path(path): load_metric_records(path)
        for path in paths
    }

    shared_custom_ids = None
    for record_map in metrics_by_method.values():
        current_ids = set(record_map.keys())
        shared_custom_ids = current_ids if shared_custom_ids is None else (shared_custom_ids & current_ids)

    shared_custom_ids = sorted(shared_custom_ids or [])
    if not shared_custom_ids:
        raise SystemExit("No shared custom_id values found across the result files.")

    if args.sample_size > len(shared_custom_ids):
        raise SystemExit(
            f"Requested sample_size={args.sample_size}, but only {len(shared_custom_ids)} shared custom_id values are available."
        )

    rng = random.Random(args.seed)
    sampled_ids = sorted(rng.sample(shared_custom_ids, args.sample_size), key=lambda x: int(x))
    summary_rows = summarize_sampled_metrics(sampled_ids=sampled_ids, metrics_by_method=metrics_by_method)

    print("method\tavg_accuracy\tavg_intent_consistency\tavg_adequacy\tavg_naturalness\tavg_overall_score")
    for row in summary_rows:
        print(
            f"{row['method']}\t{row['accuracy']:.4f}\t{row['intent_consistency']:.4f}\t{row['adequacy']:.4f}\t{row['naturalness']:.4f}\t{row['overall_score']:.4f}"
        )

    if args.output_json is not None:
        payload = {
            "root": str(args.root),
            "seed": args.seed,
            "sample_size": args.sample_size,
            "shared_custom_id_count": len(shared_custom_ids),
            "sampled_custom_ids": sampled_ids,
            "summary": summary_rows,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
