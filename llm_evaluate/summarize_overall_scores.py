import argparse
import ast
import json
import re
from pathlib import Path
from statistics import mean
from typing import List, Optional, TextIO


DEFAULT_GLOB = "*_success.jsonl"
# METRIC_NAMES = [
#     "groundedness",
#     "intent_consistency",
#     "adequacy",
#     "naturalness",
#     "uncertainty_appropriateness",
#     "specificity",
#     "conciseness",
#     "non_redundancy",
#     "coverage",
#     "terminology_quality",
#     "readability",
#     "overall_score",
# ]
METRIC_NAMES = [
    "accuracy",
    "adequacy",
    "naturalness",
    "overall_score",
]
# METRIC_NAMES = [
#     "accuracy",
#     "adequacy",
#     "naturalness",
#     "fluency",
#     "conciseness",
#     "readability",
#     "overall_score",
# ]

def metric_header(metric_name: str) -> str:
    return f"avg_{metric_name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize average overall_score from LLM evaluation result files."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Result files to summarize. If omitted, all *_success.jsonl under llm_evaluate are used.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory used when auto-discovering *_success.jsonl files.",
    )
    parser.add_argument(
        "--parse-failure-output",
        type=Path,
        default=None,
        help="Optional path to write raw response text for records that fail JSON parsing.",
    )
    return parser.parse_args()


def discover_input_paths(args: argparse.Namespace) -> List[Path]:
    if args.inputs:
        return [Path(path) for path in args.inputs]
    return sorted(args.root.glob(DEFAULT_GLOB))


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_json_candidate(text: str) -> str:
    text = strip_code_fences(text)
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0).strip() if match else text


def parse_content_json(content: str) -> Optional[dict]:
    candidate = extract_json_candidate(content)

    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(candidate)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    repaired = candidate
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    repaired = re.sub(r",\s*}", "}", repaired)
    repaired = re.sub(r",\s*]", "]", repaired)

    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(repaired)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    return None


def extract_message_content(record: dict) -> Optional[str]:
    try:
        return record["response"]["body"]["choices"][0]["message"]["content"]
    except Exception:
        return None


def extract_metric_value(parsed_content: dict, metric_name: str) -> Optional[float]:
    value = parsed_content.get(metric_name)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def write_parse_failure(
    failure_file: TextIO,
    path: Path,
    record: dict,
    line_number: int,
    reason: str,
    content: Optional[str],
) -> None:
    method = path.name.replace("_success.jsonl", "")
    custom_id = record.get("custom_id", "")
    response_body = (record.get("response") or {}).get("body")
    record_error = record.get("error")
    failure_file.write("=" * 80 + "\n")
    failure_file.write(f"method: {method}\n")
    failure_file.write(f"file: {path}\n")
    failure_file.write(f"line: {line_number}\n")
    failure_file.write(f"custom_id: {custom_id}\n")
    failure_file.write(f"reason: {reason}\n")
    failure_file.write("raw_content:\n")
    failure_file.write((content if content is not None else "<missing content>") + "\n")
    failure_file.write("raw_response_body:\n")
    failure_file.write(json.dumps(response_body, ensure_ascii=False, indent=2) + "\n")
    if record_error is not None:
        failure_file.write("record_error:\n")
        failure_file.write(json.dumps(record_error, ensure_ascii=False, indent=2) + "\n")
    failure_file.write("\n")


def summarize_file(path: Path, failure_file: Optional[TextIO] = None) -> dict:
    metric_scores = {metric_name: [] for metric_name in METRIC_NAMES}
    parse_failures = 0
    score_failures = 0
    total = 0

    with path.open("r", encoding="utf-8") as file:
        count = 0
        for line_number, line in enumerate(file, start=1):
            count += 1
            # if count > 300:
            #     break
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            content = extract_message_content(record)
            if not content:
                parse_failures += 1
                if failure_file is not None:
                    write_parse_failure(
                        failure_file=failure_file,
                        path=path,
                        record=record,
                        line_number=line_number,
                        reason="missing message content",
                        content=content,
                    )
                continue
            parsed = parse_content_json(content)
            if parsed is None:
                parse_failures += 1
                if failure_file is not None:
                    write_parse_failure(
                        failure_file=failure_file,
                        path=path,
                        record=record,
                        line_number=line_number,
                        reason="content is not valid JSON after extraction/repair",
                        content=content,
                    )
                continue

            record_has_missing_metric = False
            extracted = {}
            for metric_name in METRIC_NAMES:
                metric_value = extract_metric_value(parsed, metric_name)
                if metric_value is None:
                    record_has_missing_metric = True
                    break
                extracted[metric_name] = metric_value

            if record_has_missing_metric:
                score_failures += 1
                continue

            for metric_name, metric_value in extracted.items():
                metric_scores[metric_name].append(metric_value)

    return {
        "method": path.name.replace("_success.jsonl", ""),
        "file": str(path),
        "total_records": total,
        "parsed_records": len(metric_scores[METRIC_NAMES[0]]) if METRIC_NAMES else 0,
        "parse_failures": parse_failures,
        "score_failures": score_failures,
        "average_scores": {
            metric_name: (mean(values) if values else None)
            for metric_name, values in metric_scores.items()
        },
    }


def main() -> None:
    args = parse_args()
    paths = discover_input_paths(args)
    if not paths:
        raise SystemExit("No input files found.")

    if args.parse_failure_output is not None:
        args.parse_failure_output.parent.mkdir(parents=True, exist_ok=True)
        with args.parse_failure_output.open("w", encoding="utf-8") as failure_file:
            summaries = [summarize_file(path, failure_file=failure_file) for path in paths]
    else:
        summaries = [summarize_file(path) for path in paths]

    header_columns = ["method"] + [metric_header(metric_name) for metric_name in METRIC_NAMES]
    header_columns += ["parsed/total", "parse_failures", "score_failures"]
    print("\t".join(header_columns))

    for item in summaries:
        averages = item["average_scores"]
        parsed_total = f"{item['parsed_records']}/{item['total_records']}"
        row = [item["method"]]
        for metric_name in METRIC_NAMES:
            metric_value = averages.get(metric_name)
            row.append("NA" if metric_value is None else f"{metric_value:.4f}")
        row.extend([parsed_total, str(item["parse_failures"]), str(item["score_failures"])])
        print("\t".join(row))


if __name__ == "__main__":
    main()
