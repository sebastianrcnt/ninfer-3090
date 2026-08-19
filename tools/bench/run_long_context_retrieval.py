#!/usr/bin/env python3
"""Run a reproducible mixed-language long-context retrieval probe.

The probe uses the Anthropic-compatible token-count endpoint to size the prompt
with the model's embedded tokenizer, then sends that exact prompt for greedy,
thinking-disabled generation.  Four needles cover Korean, English, code, and
arithmetic retrieval and are distributed through the context.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


MODEL = "qwen3.8-27b-ninfer"
FILLER = "이 문장은 긴 문맥의 길이를 채우기 위한 무관한 기록입니다. "
NEEDLES = (
    "기록 A의 비밀 값은 해오라기-7319 입니다.",
    "The secret value for record B is cobalt-orbit-4821.",
    "Record C stores this exact Python expression: x*x+1",
    "기록 D의 산술 답은 1591 입니다.",
)
EXPECTED = ("해오라기-7319", "cobalt-orbit-4821", "x*x+1", "1591")


def post_json(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def make_prompt(repeats: int) -> str:
    sections = []
    for needle in NEEDLES:
        sections.append(FILLER * repeats)
        sections.append(f"\n{needle}\n")
    return "".join(sections) + (
        "\n위 긴 기록에서 A, B, C, D의 값을 찾으세요. 설명 없이 반드시 다음 네 줄 형식으로 "
        "정확히 답하세요:\nA: <값>\nB: <값>\nC: <값>\nD: <값>"
    )


def request_body(prompt: str) -> dict[str, object]:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "thinking": {"type": "disabled"},
    }


def count_tokens(base_url: str, repeats: int) -> int:
    result = post_json(
        f"{base_url}/v1/messages/count_tokens",
        request_body(make_prompt(repeats)),
        timeout=120,
    )
    return int(result["input_tokens"])


def find_repeats(base_url: str, target_tokens: int) -> tuple[int, int]:
    low, high = 0, max(1, target_tokens // 16)
    while count_tokens(base_url, high) <= target_tokens:
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        if count_tokens(base_url, middle) <= target_tokens:
            low = middle
        else:
            high = middle
    return low, count_tokens(base_url, low)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8090")
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=86400)
    args = parser.parse_args()

    repeats, input_tokens = find_repeats(args.base_url, args.target_tokens)
    prompt = make_prompt(repeats)
    payload = request_body(prompt)
    payload.update({"max_tokens": args.max_tokens, "temperature": 0, "stream": False})

    started = time.monotonic()
    result = post_json(f"{args.base_url}/v1/messages", payload, timeout=args.timeout)
    elapsed = time.monotonic() - started
    answer = "".join(
        str(block.get("text", ""))
        for block in result.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )
    missing = [value for value in EXPECTED if value not in answer]
    usage = result.get("usage", {})
    summary = {
        "target_tokens": args.target_tokens,
        "input_tokens": input_tokens,
        "output_tokens": usage.get("output_tokens"),
        "repeats_per_section": repeats,
        "elapsed_seconds": round(elapsed, 3),
        "all_needles_found": not missing,
        "missing": missing,
        "answer": answer,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
