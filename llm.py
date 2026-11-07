"""
Step 2a: the swappable model layer.

Every AI call in the project goes through call_llm() / call_llm_json(). Nothing else in the
code knows which provider is being used, so switching models means changing two settings:

    LLM_PROVIDER = gemini | groq | openai | anthropic | mock
    LLM_MODEL    = the model name for that provider

Set them as environment variables (or edit the defaults below). API keys are read from
GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY.

Two extras that matter in practice:
  * Caching: every response is saved to disk, keyed by (provider, model, prompt). Re-running
    the pipeline costs nothing and gives identical results, and the cache doubles as a log.
  * Retries: free tiers rate-limit hard (HTTP 429), so we wait and retry instead of crashing.
"""
import os
import re
import json
import time
import hashlib
from pathlib import Path

import requests

DEFAULT_MODELS = {
    "gemini": "gemini-flash-latest",       # alias that always points to the current Flash model
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
    "mock": "mock",
}

CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", "llm_cache"))
TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))   # seconds to wait for one reply
MAX_OUTPUT_TOKENS = 4000   # enough for our JSON answer; keeps each request under free-tier caps


def current_config():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]
    return provider, model


# ---------------------------------------------------------------- provider adapters
def _gemini(prompt, model, system):
    key = os.environ["GEMINI_API_KEY"].strip()
    # Key goes in a header, never in the URL, so it can't leak into error messages or logs
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": key}
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    r = requests.post(url, json=body, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _openai_compatible(prompt, model, system, base_url, key_env):
    headers = {"Authorization": f"Bearer {os.environ[key_env]}"}
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": messages, "temperature": 0.2,
            "max_tokens": MAX_OUTPUT_TOKENS, "response_format": {"type": "json_object"}}
    r = requests.post(f"{base_url}/chat/completions", json=body, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _anthropic(prompt, model, system):
    headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
               "anthropic-version": "2023-06-01", "content-type": "application/json"}
    body = {"model": model, "max_tokens": 4000, "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}]}
    if system:
        body["system"] = system
    r = requests.post("https://api.anthropic.com/v1/messages", json=body, headers=headers,
                      timeout=TIMEOUT)
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"])


def _mock(prompt, model, system):
    """Offline stand-in for testing the plumbing without an API key. NOT a real promo finder:
    it marks exclaimed dialogue as hooks and the last few beats as cliffhangers."""
    lines = [l for l in prompt.splitlines() if re.match(r"^E\d+-B\d+", l)]
    hooks = [l.split()[0] for l in lines if "!" in l and "CAMERA" not in l][:3]
    cliffs = [l.split()[0] for l in lines[-6:] if "CAMERA" not in l][-2:]
    return json.dumps({
        "hooks": [{"beat_id": h, "type": "shocking_line", "score": 5, "why": "mock"} for h in hooks],
        "cliffhangers": [{"beat_id": c, "open_question": "mock", "score": 5, "why": "mock"}
                         for c in cliffs],
    })


ADAPTERS = {
    "gemini": _gemini,
    "groq": lambda p, m, s: _openai_compatible(p, m, s, "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai": lambda p, m, s: _openai_compatible(p, m, s, "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": _anthropic,
    "mock": _mock,
}


# ---------------------------------------------------------------- public functions
def call_llm(prompt, system=None, use_cache=True, max_retries=5):
    provider, model = current_config()
    key = hashlib.sha256(f"{provider}|{model}|{system}|{prompt}".encode()).hexdigest()[:20]
    cache_file = CACHE_DIR / provider / f"{key}.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())["response"]

    wait = 10
    for attempt in range(max_retries):
        try:
            text = ADAPTERS[provider](prompt, model, system)
            break
        except (requests.Timeout, requests.ConnectionError) as e:
            # Network hiccup or a slow model reply: wait and try again, same as a 429
            if attempt < max_retries - 1:
                print(f"  [{provider}] {type(e).__name__}, waiting {wait}s...")
                time.sleep(wait)
                wait = min(wait * 2, 120)
                continue
            raise
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            # Groq rejects answers that are not valid JSON (usually truncated). The partial text
            # is in the error body, so hand it back and let the JSON repair step deal with it.
            if code == 400 and e.response is not None:
                try:
                    err = e.response.json().get("error", {})
                except ValueError:
                    err = {}
                if err.get("code") == "json_validate_failed" and err.get("failed_generation"):
                    text = err["failed_generation"]
                    break
            if code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                # Use the wait time the provider asks for, if it sends one
                header = e.response.headers.get("retry-after") if e.response is not None else None
                try:
                    pause = min(float(header) + 1, 120) if header else wait
                except ValueError:
                    pause = wait
                print(f"  [{provider}] HTTP {code} (rate limit/busy), waiting {pause:.0f}s...")
                time.sleep(pause)
                wait = min(wait * 2, 120)
                continue
            raise

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"provider": provider, "model": model,
                                      "response": text}, indent=1))
    return text


def repair_truncated(text):
    """Rescue a JSON object that was cut off mid-way: drop the unfinished trailing item and
    close the brackets that are still open. Returns None if it cannot be salvaged."""
    # Candidate cut points: after any closing bracket, and before any item that may be unfinished
    cuts = [i for i in range(len(text), 0, -1) if text[i - 1] in "}]"]
    cuts += [i for i in range(len(text), 0, -1) if text[i - 1] in "[{,"]
    cuts = cuts or [len(text)]
    for cut in cuts:
        head = text[:cut].rstrip(", [{" if text[:cut].rstrip()[-1:] in ",[{" else "")
        stack = []
        in_str = esc = False
        for ch in head:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str and ch in "{[":
                stack.append(ch)
            elif not in_str and ch in "}]":
                if stack:
                    stack.pop()
        if in_str:
            continue
        closed = head + "".join("}" if c == "{" else "]" for c in reversed(stack))
        try:
            return json.loads(closed)
        except json.JSONDecodeError:
            continue
    # Nothing closed cleanly (e.g. cut off inside a string): drop the unfinished item entirely
    last = max(text.rfind("},"), text.rfind("],"), text.rfind("["))
    if last > 0:
        trimmed = text[:last + 1].rstrip(",[")
        if trimmed and trimmed != text:
            return repair_truncated(trimmed)
    return None


def parse_json(text):
    """Models sometimes wrap JSON in ``` fences, add a sentence, or get cut off mid-answer."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        salvaged = repair_truncated(text)
        if salvaged is not None:
            return salvaged
        raise


def call_llm_json(prompt, system=None):
    """call_llm + JSON parsing, with one repair attempt if the model returns broken JSON."""
    text = call_llm(prompt, system)
    try:
        return parse_json(text)
    except json.JSONDecodeError:
        fixed = call_llm(prompt + "\n\nYour previous answer was not valid JSON. "
                         "Return ONLY the JSON object, nothing else.", system)
        return parse_json(fixed)
