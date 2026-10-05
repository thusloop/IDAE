import argparse
from contextlib import redirect_stdout
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm

from eval.bleu import compute_bleu
from eval.rouge import Rouge
from eval.meteor import Meteor
from ablation_utils import prediction_filename
# nltk.download('punkt')
def calc_text_similarity(ref, hpy):
    ref = str(ref)
    hpy = str(hpy)
    # cc = SmoothingFunction()
    # bleu = nltk.translate.bleu_score.sentence_bleu([ref], hpy, smoothing_function=cc.method4)
    # bleu = nltk_sentence_bleu(hpy, ref)
    bleu = compute_bleu([[ref.split()]], [hpy.split()], smooth=True)[0]
    rouge_score = Rouge().calc_score(hpy.split(), [ref.split()])
    # reference_tokens = [word_tokenize(ref)]
    # hypothesis_tokens = word_tokenize(hpy)
    # meteor = meteor_score(reference_tokens, hypothesis_tokens)
    return bleu, rouge_score

def cal_meteor_new(comment_pred, comment_ref, batch_size=512):
    meteor_calculator = Meteor()
    try:
        meteors = []
        for start in tqdm(range(0, len(comment_pred), batch_size), desc="Computing METEOR", unit="batch"):
            end = min(start + batch_size, len(comment_pred))
            _, batch_scores = meteor_calculator.compute_score(
                list(range(start, end)),
                [text.split() for text in comment_pred[start:end]],
                [[text.split()] for text in comment_ref[start:end]],
            )
            meteors.extend(batch_scores)
        return meteors
    finally:
        meteor_calculator.close()

# nltk.download('averaged_perceptron_tagger')
nltk.data.find("taggers/averaged_perceptron_tagger_eng")
def reformat_comment(comment):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(comment)
    tags = nltk.pos_tag(words)

    for idx, word in enumerate(tags):
        if word[1] == 'VBZ':
            words[idx] = lemmatizer.lemmatize(words[idx], pos='v')
    if tags[0][1] != 'VBD' and tags[0][1] != 'VBN':
        words[0] = lemmatizer.lemmatize(words[0], pos='v')
    return " ".join(words)


def load_label_list(path):
    label_set = set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for v in data.values():
        for item in v:
            lab = item.get("label")
            if lab is not None:
                label_set.add(lab)
    return sorted(label_set)

def evaluate_file(input_file, label_list):
    label_stats = {lab: {"bleu": 0.0, "rouge": 0.0, "meteor": 0.0, "count": 0} for lab in label_list}
    total_bleu, total_rouge, total_meteor, total_count = 0.0, 0.0, 0.0, 0
    print(f"Processing file: {input_file} ...")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        with open(input_file, 'r', encoding='utf-8') as f:
            predictions = []
            ground_truths = []
            labels = []
            bleu_scores = []
            rouge_scores = []
            for i, line in tqdm(enumerate(f), total=total_lines, desc="Processing file", unit="line"):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)

                    prediction = data.get('prediction', '')
                    ground_truth = data.get('ground_truth', '')
                    prediction = reformat_comment(prediction.lower())
                    ground_truth = reformat_comment(ground_truth.lower())
                    lab = data.get('label', None)

                    b, r = calc_text_similarity(ground_truth, prediction)
                    predictions.append(str(prediction))
                    ground_truths.append(str(ground_truth))
                    labels.append(lab)
                    bleu_scores.append(b)
                    rouge_scores.append(r)
                except json.JSONDecodeError:
                    print(f"Invalid JSON on line {i + 1}; skipping.")
                except Exception as e:
                    print(f"Error processing line {i + 1}: {e}")

        if not predictions:
            print("No valid records were processed.")
            return None

        print(f"Computing METEOR for {len(predictions)} records; this may take a while.", flush=True)
        meteor_scores = cal_meteor_new(predictions, ground_truths)
        for i in range(len(predictions)):
            b = bleu_scores[i]
            r = rouge_scores[i]
            m = meteor_scores[i]
            lab = labels[i]

            total_bleu += b
            total_rouge += r
            total_meteor += m
            total_count += 1

            if lab not in label_stats:
                label_stats[lab] = {"bleu": 0.0, "rouge": 0.0, "meteor": 0.0, "count": 0}
            label_stats[lab]["bleu"] += b
            label_stats[lab]["rouge"] += r
            label_stats[lab]["meteor"] += m
            label_stats[lab]["count"] += 1

        if total_count == 0:
            print("No valid records were processed.")
            return None

        print("-" * 30)
        for lab in label_list:
            s = label_stats.get(lab)
            if not s or s["count"] == 0:
                print(f"Label {lab}: no samples")
                continue
            avg_b = s["bleu"] / s["count"]
            avg_r = s["rouge"] / s["count"]
            avg_m = s["meteor"] / s["count"]
            print(f"Label {lab}: BLEU {avg_b:.6f}, Rouge-L {avg_r:.6f}, METEOR {avg_m:.6f} (n={s['count']})")

        avg_bleu = total_bleu / total_count
        avg_rouge = total_rouge / total_count
        avg_meteor = total_meteor / total_count
        print("-" * 30)
        print(f"Finished processing {total_count} records.")
        print(f"Average BLEU:    {avg_bleu:.6f}")
        print(f"Average Rouge-L: {avg_rouge:.6f}")
        print(f"Average METEOR:  {avg_meteor:.6f}")
        print("-" * 30)
        return {
            "file": input_file,
            "bleu_sum": total_bleu,
            "rouge_sum": total_rouge,
            "meteor_sum": total_meteor,
            "count": total_count,
            "avg_bleu": avg_bleu,
            "avg_rouge": avg_rouge,
            "avg_meteor": avg_meteor,
            "label_stats": label_stats,
        }

    except FileNotFoundError:
        print(f"Error: file not found: {input_file}")
        return None

def _parse_ids(value):
    try:
        values = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Provide comma-separated integers, for example 0,3,5") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required")
    return values


def _line_count(path):
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for _ in stream)


def _discover_groups(data_dir, ablation, num_examples, seeds):
    # All ablations, including contrastive, are explicit in prediction names:
    # predictions_label_num3_funcom.contrastive.seed42.jsonl
    suffix = f".{ablation}"
    pattern = re.compile(
        rf"^predictions_label_num(\d+)_(funcom|tlcodesum){re.escape(suffix)}"
        r"(?:\.seed(\d+))?\.jsonl$"
    )
    available = set()
    for path in data_dir.glob("predictions_label_num*.jsonl"):
        match = pattern.fullmatch(path.name)
        if match:
            number = int(match.group(1))
            seed = int(match.group(3)) if match.group(3) is not None else None
            available.add((number, seed))

    if not available and (num_examples is None or seeds is None):
        raise ValueError(f"No prediction files matching the requested configuration were found in {data_dir}")

    numbers = num_examples if num_examples is not None else {number for number, _ in available}
    selected_seeds = seeds if seeds is not None else {seed for _, seed in available}
    candidates = {(number, seed) for number in numbers for seed in selected_seeds}
    if not candidates:
        raise ValueError(f"No prediction files matching the requested configuration were found in {data_dir}")

    expected = {dataset: _line_count(data_dir / f"{dataset}.test") for dataset in ("funcom", "tlcodesum")}
    groups = []
    skipped = []
    for number, seed in sorted(candidates, key=lambda item: (item[0], -1 if item[1] is None else item[1])):
        name = f"{ablation}.num{number}" + (f".seed{seed}" if seed is not None else "")
        paths = {
            dataset: data_dir / prediction_filename(number, dataset, ablation, seed=seed)
            for dataset in ("funcom", "tlcodesum")
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            message = f"Skipping {name}: missing paired files {', '.join(missing)}"
            skipped.append(message)
            print(message, file=sys.stderr)
            continue
        incomplete = []
        for dataset, path in paths.items():
            actual = _line_count(path)
            if actual != expected[dataset]:
                incomplete.append(f"{dataset} {actual}/{expected[dataset]} lines")
        if incomplete:
            message = f"Skipping {name}: predictions do not cover the complete test set ({', '.join(incomplete)})"
            skipped.append(message)
            print(message, file=sys.stderr)
            continue
        groups.append((name, paths))
    if not groups:
        raise ValueError("No complete FunCom/TLCodesum prediction pairs were found")
    return groups, skipped


def _collect_labels(paths):
    labels = set()
    for path in paths.values():
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    try:
                        labels.add(json.loads(line).get("label"))
                    except json.JSONDecodeError:
                        pass
    return sorted(labels, key=str)


def _evaluate_group(name, paths):
    label_list = _collect_labels(paths)
    print(f"\n{'=' * 50}\nConfiguration: {name}\n{'=' * 50}")
    results = {}
    for dataset in ("funcom", "tlcodesum"):
        print(f"\nDataset: {dataset}")
        result = evaluate_file(str(paths[dataset]), label_list)
        if result is None:
            raise ValueError(f"The {dataset} file for {name} produced no valid evaluation results")
        lines = _line_count(paths[dataset])
        if result["count"] != lines:
            raise ValueError(
                f"The {dataset} file for {name} processed only {result['count']}/{lines} lines successfully; "
                "check the skip messages printed above"
            )
        results[dataset] = result

    count = sum(result["count"] for result in results.values())
    totals = {
        metric: sum(result[f"{metric}_sum"] for result in results.values())
        for metric in ("bleu", "rouge", "meteor")
    }
    print("\nAverage results grouped by intent label across both datasets:")
    for lab in label_list:
        stats = {
            metric: sum(result["label_stats"][lab][metric] for result in results.values())
            for metric in ("bleu", "rouge", "meteor", "count")
        }
        if stats["count"]:
            print(
                f"Label {lab}: BLEU {stats['bleu'] / stats['count']:.6f}, "
                f"Rouge-L {stats['rouge'] / stats['count']:.6f}, "
                f"METEOR {stats['meteor'] / stats['count']:.6f} (n={stats['count']})"
            )
    print(f"Overall average across both datasets ({count} records).")
    print(f"Overall Average BLEU:    {totals['bleu'] / count:.6f}")
    print(f"Overall Average Rouge-L: {totals['rouge'] / count:.6f}")
    print(f"Overall Average METEOR:  {totals['meteor'] / count:.6f}")
    return {"name": name, "count": count, **{metric: totals[metric] / count for metric in totals}}


class _Tee:
    def __init__(self, terminal, report):
        self.terminal = terminal
        self.report = report

    def write(self, text):
        self.terminal.write(text)
        self.report.write(text)

    def flush(self):
        self.terminal.flush()
        self.report.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pair FunCom/TLCodesum prediction files by configuration, compute metrics, and save a TXT report")
    default_data_dir = Path(__file__).resolve().parents[2] / "data"
    parser.add_argument("--data-dir", type=Path, default=default_data_dir, help="Directory in which to discover prediction files")
    parser.add_argument(
        "--ablation",
        choices=("main", "contrastive", "rand_ex", "token", "semantic"),
        default="main",
    )
    parser.add_argument("--num-examples", type=_parse_ids, help="Evaluate only the specified shot counts, for example 0,3,5")
    parser.add_argument("--seeds", type=_parse_ids, help="Evaluate only the specified seeds, for example 42,13,100")
    parser.add_argument(
        "--group", nargs=3, action="append", metavar=("NAME", "FUNCOM_JSONL", "TLCODESUM_JSONL"),
        help="Explicitly provide an input group; may be repeated and disables automatic discovery",
    )
    parser.add_argument("--output", type=Path, help="Output TXT path; defaults to a timestamped file in the data directory")
    parser.add_argument("--list-groups", action="store_true", help="List complete configuration groups without computing metrics")
    args = parser.parse_args(argv)

    if args.group:
        groups = []
        skipped = []
        for name, funcom, tlcodesum in args.group:
            paths = {"funcom": Path(funcom).resolve(), "tlcodesum": Path(tlcodesum).resolve()}
            for path in paths.values():
                if not path.is_file():
                    parser.error(f"Input file not found: {path}")
            if any(existing_name == name for existing_name, _ in groups):
                parser.error(f"Duplicate group name: {name}")
            groups.append((name, paths))
    else:
        try:
            groups, skipped = _discover_groups(args.data_dir.resolve(), args.ablation, args.num_examples, args.seeds)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))

    for name, paths in groups:
        print(f"{name}: funcom={paths['funcom']}, tlcodesum={paths['tlcodesum']}")
    if args.list_groups:
        return

    output = args.output or args.data_dir / f"cal_pre_results_{datetime.now():%Y%m%d_%H%M%S}.txt"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    try:
        with output.open("x", encoding="utf-8") as report:
            with redirect_stdout(_Tee(sys.stdout, report)):
                print(f"Number of groups evaluated: {len(groups)}; report file: {output}")
                print("The report is complete only when it ends with 'END OF REPORT'.")
                if skipped:
                    print(f"Skipped {len(skipped)} missing or incomplete configurations:")
                    for message in skipped:
                        print(message)
                summaries = [_evaluate_group(name, paths) for name, paths in groups]
                print(f"\n{'=' * 50}\nSummary of all configurations\n{'=' * 50}")
                for row in summaries:
                    print(
                        f"{row['name']}: BLEU={row['bleu']:.6f}, "
                        f"Rouge-L={row['rouge']:.6f}, METEOR={row['meteor']:.6f} (n={row['count']})"
                    )
                print(f"Total elapsed time: {time.time() - start_time:.2f} seconds")
                print("END OF REPORT")
    except FileExistsError:
        parser.error(f"Output file already exists and will not be overwritten: {output}; specify a new --output")


if __name__ == "__main__":
    main()