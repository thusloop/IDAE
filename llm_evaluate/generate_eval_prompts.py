import argparse
import csv
import hashlib
import json
import random
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from prompt import prompt as PROMPT_TEMPLATE


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated_prompts" / "main_num3_funcom_seed42_tlcodesum_seed13"
DEFAULT_DATASETS = ("tlcodesum", "funcom")
DATA_DIR = ROOT_DIR / "data"
KUMIC_WORKBOOKS = {
    "funcom": DATA_DIR / "funcom-test.xlsx",
    "tlcodesum": DATA_DIR / "tlcodesum-test.xlsx",
}
DEFAULT_OURS = {
    "funcom": DATA_DIR / "predictions_label_num3_funcom.main.seed42.jsonl",
    "tlcodesum": DATA_DIR / "predictions_label_num3_tlcodesum.main.seed13.jsonl",
}
BASELINE_NAMES = ("dome", "baseline_3shot_comment2code", "cot_3shot_comment2code")
XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM evaluation prompts for comment-quality assessment."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
        help="Datasets to include.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=150,
        help="Number of code samples to draw per dataset. Must be >= number of intents when coverage is required.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the generated prompt files.",
    )
    parser.add_argument(
        "--ours-name",
        default="ours_simcse_codet5",
        help="System name to use for the paper model in metadata.",
    )
    for dataset in DEFAULT_DATASETS:
        parser.add_argument(
            f"--ours-{dataset}",
            type=Path,
            default=DEFAULT_OURS[dataset],
            help=f"Predictions for {dataset}; default: {DEFAULT_OURS[dataset]}",
        )
    parser.add_argument(
        "--ensure-all-intents",
        action="store_true",
        default=True,
        help="Ensure every sampled dataset contains at least one sample for each intent.",
    )
    parser.add_argument(
        "--no-ensure-all-intents",
        dest="ensure_all_intents",
        action="store_false",
        help="Disable intent coverage constraints.",
    )
    return parser.parse_args()


def get_dataset_paths(dataset: str, ours_path: Optional[Path] = None) -> Dict[str, Path]:
    return {
        "xlsx": KUMIC_WORKBOOKS[dataset],
        "test": DATA_DIR / f"{dataset}.test",
        "ours": ours_path if ours_path is not None else DEFAULT_OURS[dataset],
    }


def load_ours_predictions(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))
    return records


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def normalize_cell(value) -> str:
    return "" if value is None else str(value).strip()


def read_workbook_rows(path: Path) -> Iterator[Dict[str, str]]:
    """Read KUMIC XLSX without requiring pandas/openpyxl in the training env."""
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            with archive.open("xl/sharedStrings.xml") as stream:
                for _, element in ET.iterparse(stream, events=("end",)):
                    if element.tag == XML_NS + "si":
                        strings.append("".join(node.text or "" for node in element.iter(XML_NS + "t")))
                        element.clear()
        with archive.open("xl/worksheets/sheet1.xml") as stream:
            header = None
            for _, row in ET.iterparse(stream, events=("end",)):
                if row.tag != XML_NS + "row":
                    continue
                columns = {}
                for cell in row:
                    column = "".join(c for c in cell.get("r", "") if c.isalpha())
                    value = cell.find(XML_NS + "v")
                    inline = cell.find(XML_NS + "is")
                    if value is not None:
                        text = strings[int(value.text)] if cell.get("t") == "s" else value.text or ""
                    elif inline is not None:
                        text = "".join(node.text or "" for node in inline.iter(XML_NS + "t"))
                    else:
                        text = ""
                    columns[column] = text
                row.clear()
                if header is None:
                    header = {key: value.strip() for key, value in columns.items()}
                    needed = {"ids", "intent", "ground_truth", *BASELINE_NAMES}
                    if not needed <= set(header.values()):
                        raise ValueError(f"{path}: missing workbook columns: {needed - set(header.values())}")
                else:
                    yield {name: normalize_cell(columns.get(key)) for key, name in header.items()}


def load_test_records(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))
    return records


def render_prompt(code: str, intent: str, comment: str) -> str:
    return (
        PROMPT_TEMPLATE.replace("{code}", code)
        .replace("{intent}", intent)
        .replace("{comment}", comment)
    )


def load_merged_records(dataset: str, ours_name: str, ours_path: Optional[Path] = None) -> List[dict]:
    paths = get_dataset_paths(dataset, ours_path)
    test_records = load_test_records(paths["test"])
    ours_records = load_ours_predictions(paths["ours"])

    if len(test_records) != len(ours_records):
        raise ValueError(
            f"Row count mismatch for {dataset}: test={len(test_records)} vs ours={len(ours_records)}"
        )

    test_by_id: Dict[str, Tuple[int, dict]] = {}
    for test_idx, test_record in enumerate(test_records):
        key = str(test_record["id"]).strip()
        if key in test_by_id:
            raise ValueError(f"{dataset}: duplicate test ID {key}")
        test_by_id[key] = (test_idx, test_record)
        ours = ours_records[test_idx]
        if normalize_text(ours.get("ground_truth", "")) != normalize_text(test_record.get("comment", "")) or normalize_text(ours.get("label", "")) != normalize_text(test_record.get("label", "")):
            raise ValueError(f"{dataset}: {paths['ours']} does not align with {paths['test']} at line {test_idx + 1}")

    merged_records = []
    verification_mismatch_count = 0
    seen_ids = set()
    for row_idx, row in enumerate(read_workbook_rows(paths["xlsx"])):
        source_id = row["ids"]
        if source_id not in test_by_id or source_id in seen_ids:
            raise ValueError(f"{dataset}: missing/duplicate workbook ID {source_id} at row {row_idx + 2}")
        seen_ids.add(source_id)
        test_idx, test_record = test_by_id[source_id]
        ours = ours_records[test_idx]

        ground_truth_match = normalize_text(row["ground_truth"]) == normalize_text(test_record.get("comment", ""))
        intent_match = normalize_text(row["intent"]) == normalize_text(test_record.get("label", ""))
        if not intent_match:
            raise ValueError(f"{dataset}: workbook and .test label disagree for ID {source_id}")
        if not ground_truth_match:
            verification_mismatch_count += 1

        merged_records.append(
            {
                "dataset": dataset,
                "row_index": row_idx,
                "ids": source_id,
                "intent": normalize_cell(row["intent"]),
                "ground_truth": normalize_cell(test_record.get("comment", "")),
                "code": normalize_cell(test_record.get("raw_code", "")),
                "comments": {
                    "dome": normalize_cell(row["dome"]),
                    "baseline_3shot_comment2code": normalize_cell(row["baseline_3shot_comment2code"]),
                    "cot_3shot_comment2code": normalize_cell(row["cot_3shot_comment2code"]),
                    ours_name: normalize_cell(ours.get("prediction", "")),
                },
                "alignment_check": {
                    "matched_by": "id_via_test_split",
                    "test_row_index": test_idx,
                    "ground_truth_match_with_test": ground_truth_match,
                    "intent_match_with_test": intent_match,
                    "ours_ground_truth_match_with_test": normalize_text(test_record.get("comment", "")) == normalize_text(ours.get("ground_truth", "")),
                    "ours_intent_match_with_test": normalize_text(test_record.get("label", "")) == normalize_text(ours.get("label", "")),
                },
            }
        )

    if len(seen_ids) != len(test_records):
        raise ValueError(f"{dataset}: workbook contains {len(seen_ids)} distinct IDs but .test contains {len(test_records)} rows")
    if verification_mismatch_count:
        print(
            f"[INFO] {dataset}: {verification_mismatch_count} workbook references differ from .test (e.g. verb inflection); prompt metadata uses .test reference."
        )

    return merged_records


def sample_records(records: List[dict], sample_size: int, seed: int, ensure_all_intents: bool) -> List[dict]:
    if sample_size <= 0:
        raise ValueError("--sample-size must be a positive integer.")

    rng = random.Random(seed)
    by_intent: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        by_intent[record["intent"]].append(record)

    intents = sorted(by_intent)
    if ensure_all_intents and sample_size < len(intents):
        raise ValueError(
            f"--sample-size={sample_size} is smaller than the number of intents={len(intents)}."
        )

    sampled: List[dict] = []
    used_keys = set()

    if ensure_all_intents:
        for intent in intents:
            choice = rng.choice(by_intent[intent])
            sampled.append(choice)
            used_keys.add((choice["dataset"], choice["row_index"]))

    remaining = sample_size - len(sampled)
    if remaining <= 0:
        return sampled

    pool = [record for record in records if (record["dataset"], record["row_index"]) not in used_keys]
    if remaining > len(pool):
        raise ValueError(
            f"Requested sample size {sample_size}, but only {len(records)} records are available."
        )

    sampled.extend(rng.sample(pool, remaining))
    sampled.sort(key=lambda item: (item["intent"], item["row_index"]))
    return sampled


def build_prompt_records(records: List[dict]) -> List[dict]:
    prompt_records = []
    sample_id = 0
    for record in records:
        for system_name, comment in record["comments"].items():
            sample_id += 1
            prompt_text = render_prompt(
                code=record["code"],
                intent=record["intent"],
                comment=comment,
            )
            prompt_records.append(
                {
                    "prompt_id": sample_id,
                    "dataset": record["dataset"],
                    "row_index": record["row_index"],
                    "source_id": int(record["ids"]),
                    "intent": record["intent"],
                    "system_name": system_name,
                    "ground_truth": record["ground_truth"],
                    "comment": comment,
                    "prompt": prompt_text,
                }
            )
    return prompt_records


def write_outputs(
    output_dir: Path,
    datasets: List[str],
    sample_size: int,
    seed: int,
    sampled_by_dataset: Dict[str, List[dict]],
    prompt_records: List[dict],
    ours_paths: Dict[str, Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_tag = "-".join(datasets)
    stem = f"eval_prompts.{dataset_tag}.n{sample_size}.seed{seed}"

    prompts_path = output_dir / f"{stem}.jsonl"
    manifest_path = output_dir / f"{stem}.manifest.json"
    preview_path = output_dir / f"{stem}.preview.csv"
    existing = [str(path) for path in (prompts_path, manifest_path, preview_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing prompts: {existing}. Choose a new --output-dir or --seed.")
    with prompts_path.open("w", encoding="utf-8") as file:
        for record in prompt_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "datasets": datasets,
        "sample_size_per_dataset": sample_size,
        "seed": seed,
        "num_code_samples": sum(len(records) for records in sampled_by_dataset.values()),
        "num_prompt_records": len(prompt_records),
        "intent_distribution": {
            dataset: dict(Counter(record["intent"] for record in records))
            for dataset, records in sampled_by_dataset.items()
        },
        "systems": sorted({record["system_name"] for record in prompt_records}),
        "prompt_template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest(),
        "ours_prediction_files": {dataset: str(ours_paths[dataset].resolve()) for dataset in datasets},
        "kumic_workbooks": {dataset: str(KUMIC_WORKBOOKS[dataset].resolve()) for dataset in datasets},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    preview_rows = []
    for record in prompt_records[:20]:
        preview_rows.append(
            {
                "prompt_id": record["prompt_id"],
                "dataset": record["dataset"],
                "intent": record["intent"],
                "system_name": record["system_name"],
                "row_index": record["row_index"],
                "source_id": record["source_id"],
            }
        )
    with preview_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=("prompt_id", "dataset", "intent", "system_name", "row_index", "source_id"))
        writer.writeheader()
        writer.writerows(preview_rows)

    print(f"Saved prompts to: {prompts_path}")
    print(f"Saved manifest to: {manifest_path}")
    print(f"Saved preview to: {preview_path}")


def main() -> None:
    args = parse_args()
    ours_paths = {dataset: getattr(args, f"ours_{dataset}") for dataset in args.datasets}

    sampled_by_dataset: Dict[str, List[dict]] = {}
    all_prompt_records: List[dict] = []

    for dataset in args.datasets:
        merged_records = load_merged_records(dataset, args.ours_name, ours_paths[dataset])
        sampled_records = sample_records(
            merged_records,
            sample_size=args.sample_size,
            seed=args.seed,
            ensure_all_intents=args.ensure_all_intents,
        )
        sampled_by_dataset[dataset] = sampled_records
        all_prompt_records.extend(build_prompt_records(sampled_records))

    write_outputs(
        output_dir=args.output_dir,
        datasets=args.datasets,
        sample_size=args.sample_size,
        seed=args.seed,
        sampled_by_dataset=sampled_by_dataset,
        prompt_records=all_prompt_records,
        ours_paths=ours_paths,
    )


if __name__ == "__main__":
    main()
