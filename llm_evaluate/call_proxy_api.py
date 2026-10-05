import argparse
import copy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set
from urllib import error, request


DEFAULT_API_BASE = "https://xxx"
DEFAULT_TIMEOUT = 120
DEFAULT_SLEEP_SECONDS = 0.2
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/135.0.0.0 Safari/537.36"
)
DEFAULT_INPUTS = [
    Path(__file__).resolve().parent / "ours_simcse_codet5.jsonl",
    #Path(__file__).resolve().parent / "cot_3shot_comment2code.jsonl",
    # Path(__file__).resolve().parent / "baseline_3shot_comment2code.jsonl",
    # Path(__file__).resolve().parent / "dome.jsonl",
]


def render_progress(current: int, total: int, success: int, failure: int, width: int = 30) -> str:
    if total <= 0:
        total = 1
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} success={success} error={failure}"


def print_progress(current: int, total: int, success: int, failure: int) -> None:
    print("\r" + render_progress(current, total, success, failure), end="", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send OpenAI-compatible chat completion requests to a proxy API and save results as jsonl."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=DEFAULT_INPUTS,
        help="Input request jsonl files. Defaults to ours_simcse_codet5.jsonl and cot_3shot_comment2code.jsonl.",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help="Proxy API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("IDAE_LLM_API_KEY", ""),
        help="Bearer token; defaults to IDAE_LLM_API_KEY environment variable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to write *_success.jsonl and *_error.jsonl.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Pause between submitting requests; with concurrency this is not a strict rate limit.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of in-flight API requests per input file (default: 1).",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Maximum number of requests to send for each input file. 0 means no limit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result files instead of resuming.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="On resume, retry failed request IDs; successful IDs are still skipped.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4",
        help="Override body.model in the input requests. If empty, keep the original model.",
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.max_requests < 0:
        parser.error("--max-requests must be non-negative")
    if args.sleep_seconds < 0:
        parser.error("--sleep-seconds must be non-negative")
    return args


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_processed_custom_ids(paths: Iterable[Path]) -> Set[str]:
    processed: Set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                custom_id = record.get("custom_id")
                if custom_id is not None:
                    processed.add(str(custom_id))
    return processed


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_target_url(api_base: str, relative_url: str) -> str:
    base = api_base.rstrip("/")
    suffix = relative_url if relative_url.startswith("/") else f"/{relative_url}"
    return f"{base}{suffix}"


def extract_assistant_content(response_body: Optional[Dict]) -> str:
    """Return accumulated assistant text from our normalized streaming body."""
    if not isinstance(response_body, dict):
        return ""
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def empty_response_error(response_body: Optional[Dict]) -> Dict:
    return {
        "type": "EmptyResponse",
        "message": "HTTP request succeeded, but the LLM returned no assistant content.",
        "response_body": response_body,
    }


def send_request(
    target_url: str,
    api_key: str,
    body: Dict,
    timeout: int,
) -> Dict:
    request_body = dict(body)
    request_body["stream"] = True
    data = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    http_request = request.Request(
        url=target_url,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            response_body = parse_streaming_response(response)
            request_id = response.headers.get("x-request-id") or response_body.get("id")
            if not extract_assistant_content(response_body).strip():
                return {
                    "ok": False,
                    "status_code": response.status,
                    "request_id": request_id,
                    "body": response_body,
                    "error": empty_response_error(response_body),
                }
            return {
                "ok": True,
                "status_code": response.status,
                "request_id": request_id,
                "body": response_body,
                "error": None,
            }
    except error.HTTPError as exc:
        raw_text = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed = {"raw_text": raw_text}
        return {
            "ok": False,
            "status_code": exc.code,
            "request_id": exc.headers.get("x-request-id"),
            "body": parsed,
            "error": {
                "type": "HTTPError",
                "message": str(exc),
                "response_body": parsed,
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": None,
            "request_id": None,
            "body": None,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
        }


def extract_text_from_delta_value(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        parts.append(text_value)
                elif isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "".join(parts)
    return ""


def iter_sse_payloads(response) -> Iterable[str]:
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload:
            yield payload


def parse_streaming_response(response) -> Dict:
    response_id: Optional[str] = None
    object_name: Optional[str] = None
    created: Optional[int] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    content_parts: List[str] = []
    role: Optional[str] = None
    chunks: List[dict] = []

    for payload in iter_sse_payloads(response):
        if payload == "[DONE]":
            break

        chunk = json.loads(payload)
        chunks.append(chunk)

        response_id = chunk.get("id", response_id)
        object_name = chunk.get("object", object_name)
        created = chunk.get("created", created)
        model = chunk.get("model", model)
        if chunk.get("usage") is not None:
            usage = chunk["usage"]

        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            role = delta.get("role", role)

            delta_content = extract_text_from_delta_value(delta.get("content"))
            if delta_content:
                content_parts.append(delta_content)

            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

    return {
        "id": response_id,
        "object": object_name or "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": role or "assistant",
                    "content": "".join(content_parts),
                },
                "finish_reason": finish_reason or "stop",
            }
        ],
        "usage": usage,
        "stream_chunks": chunks,
    }


def build_output_record(request_record: dict, response_payload: Dict) -> Dict:
    return {
        "id": str(uuid.uuid4()),
        "custom_id": str(request_record.get("custom_id", "")),
        "response": {
            "status_code": response_payload["status_code"],
            "request_id": response_payload["request_id"],
            "body": response_payload["body"],
        }
        if response_payload["ok"]
        else None,
        "error": None if response_payload["ok"] else response_payload["error"],
    }


def append_jsonl(path: Path, record: dict) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def repair_empty_success_records(success_path: Path, error_path: Path) -> int:
    """Move old HTTP-200/empty-content records out of success on resume.

    Older runs classified every HTTP 200 response as success. Keep a one-time
    backup, append normalized failure records, and atomically rewrite success.
    """
    if not success_path.is_file():
        return 0

    kept_lines: List[str] = []
    moved: List[dict] = []
    with success_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Do not silently destroy an unrelated malformed line.
                kept_lines.append(line if line.endswith("\n") else line + "\n")
                continue
            response = record.get("response") or {}
            body = response.get("body") if isinstance(response, dict) else None
            if response and not extract_assistant_content(body).strip():
                failed = copy.deepcopy(record)
                failed["response"] = None
                failed["error"] = {
                    **empty_response_error(body),
                    "reclassified_from_success": True,
                    "original_status_code": response.get("status_code"),
                    "original_request_id": response.get("request_id"),
                }
                moved.append(failed)
            else:
                kept_lines.append(json.dumps(record, ensure_ascii=False) + "\n")

    if not moved:
        return 0

    backup_path = success_path.with_name(success_path.name + ".before_empty_repair")
    if not backup_path.exists():
        shutil.copy2(success_path, backup_path)

    ensure_parent_dir(error_path)
    with error_path.open("a", encoding="utf-8") as file:
        for record in moved:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    temporary_path = success_path.with_name(success_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        file.writelines(kept_lines)
    os.replace(temporary_path, success_path)
    print(
        f"{success_path.name}: moved {len(moved)} empty response(s) to {error_path.name}; "
        f"backup={backup_path.name}"
    )
    return len(moved)


def process_input_file(
    input_path: Path,
    output_dir: Path,
    api_base: str,
    api_key: str,
    model_override: str,
    timeout: int,
    sleep_seconds: float,
    max_requests: int,
    overwrite: bool,
    retry_errors: bool = False,
    concurrency: int = 1,
) -> None:
    request_records = load_jsonl(input_path)
    success_path = output_dir / f"{input_path.stem}_success.jsonl"
    error_path = output_dir / f"{input_path.stem}_error.jsonl"

    if overwrite:
        if success_path.exists():
            success_path.unlink()
        if error_path.exists():
            error_path.unlink()

    repair_empty_success_records(success_path, error_path)

    processed_ids = load_processed_custom_ids([success_path] if retry_errors else [success_path, error_path])
    pending_records = [
        record
        for record in request_records
        if str(record.get("custom_id", "")) not in processed_ids
    ]
    if max_requests > 0:
        pending_records = pending_records[:max_requests]

    sent_count = 0
    skipped_count = sum(
        str(record.get("custom_id", "")) in processed_ids
        for record in request_records
    )
    success_count = 0
    error_count = 0
    total_to_send = len(pending_records)

    if total_to_send == 0:
        print(f"{input_path.name}: no pending requests, skipped={len(request_records)}")
        print(f"  success file: {success_path}")
        print(f"  error file:   {error_path}")
        return

    print(f"{input_path.name}: sending {total_to_send} requests...")
    print_progress(0, total_to_send, 0, 0)

    def request_one(request_record: dict) -> Dict:
        relative_url = request_record.get("url", "/v1/chat/completions")
        target_url = build_target_url(api_base, relative_url)
        request_body = dict(request_record["body"])
        if model_override:
            request_body["model"] = model_override
        return send_request(
            target_url=target_url,
            api_key=api_key,
            body=request_body,
            timeout=timeout,
        )

    # Only the network calls run in worker threads. JSONL writes and progress
    # accounting stay in the main thread, preventing interleaved/corrupt lines.
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        in_flight = {}
        next_index = 0

        def submit_one(record: dict) -> None:
            nonlocal next_index
            future = executor.submit(request_one, record)
            in_flight[future] = record
            next_index += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        while next_index < len(pending_records) and len(in_flight) < concurrency:
            submit_one(pending_records[next_index])

        while in_flight:
            completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in completed:
                request_record = in_flight.pop(future)
                try:
                    response_payload = future.result()
                except Exception as exc:  # defensive: send_request normally catches these
                    response_payload = {
                        "ok": False,
                        "status_code": None,
                        "request_id": None,
                        "body": None,
                        "error": {
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                        },
                    }
                output_record = build_output_record(request_record, response_payload)

                if response_payload["ok"]:
                    append_jsonl(success_path, output_record)
                    success_count += 1
                else:
                    append_jsonl(error_path, output_record)
                    error_count += 1

                sent_count += 1
                print_progress(sent_count, total_to_send, success_count, error_count)

                if next_index < len(pending_records):
                    submit_one(pending_records[next_index])

    print()
    print(
        f"{input_path.name}: sent={sent_count}, success={success_count}, error={error_count}, skipped={skipped_count}"
    )
    print(f"  success file: {success_path}")
    print(f"  error file:   {error_path}")


def main() -> None:
    args = parse_args()
    for input_arg in args.inputs:
        process_input_file(
            input_path=Path(input_arg),
            output_dir=args.output_dir,
            api_base=args.api_base,
            api_key=args.api_key,
            model_override=args.model,
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
            max_requests=args.max_requests,
            overwrite=args.overwrite,
            retry_errors=args.retry_errors,
            concurrency=args.concurrency,
        )


if __name__ == "__main__":
    main()
