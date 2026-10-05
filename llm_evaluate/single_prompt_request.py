import json
from urllib import error, request


API_BASE = "https://xxx/"
API_KEY = "sk-xxx"
MODEL = "xxx"
TEMPERATURE = 0
TIMEOUT = 120
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/135.0.0.0 Safari/537.36"
)

# Edit this prompt directly when needed.
PROMPT = """hi"""


def build_target_url(api_base: str, relative_url: str) -> str:
    base = api_base.rstrip("/")
    suffix = relative_url if relative_url.startswith("/") else f"/{relative_url}"
    return f"{base}{suffix}"


def main() -> None:
    url = build_target_url(API_BASE, "/v1/chat/completions")
    body = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    http_request = request.Request(
        url=url,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=TIMEOUT) as response:
            raw_text = response.read().decode("utf-8")
            parsed = json.loads(raw_text)
            print("status_code:", response.status)
            print("model:", parsed.get("model"))
            print("raw_response:")
            print(json.dumps(parsed, ensure_ascii=False, indent=2))

            choices = parsed.get("choices") or []
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                print("\nassistant_content:")
                print(content)
    except error.HTTPError as exc:
        raw_text = exc.read().decode("utf-8", errors="replace")
        print("status_code:", exc.code)
        print("error_response:")
        print(raw_text)
    except Exception as exc:
        print(f"{exc.__class__.__name__}: {exc}")


if __name__ == "__main__":
    main()
