"""
Classify each noisy dataset using GPT or Gemini.

Input   : Dataset/noisy/hate_speech_400_p_*.csv  (produced by add_noise.py)
Output  : Dataset/results/<model>/hate_speech_400_p_*.csv
            same CSV with one extra column: "pred_<model>"

Usage
-----
  python get_llm_responses.py --model gpt
  python get_llm_responses.py --model gemini
  python get_llm_responses.py --model both

API keys
--------
Create a .env file in the same directory:
  OPENAI_API_KEY=sk-...
  GEMINI_API_KEY=AIza...

Robustness
----------
- Batches that fail are recursively split in half until individual items.
- Rate-limit errors trigger an exponential back-off before retrying.
- Files that already have an output are skipped (safe to resume).
"""

import argparse
import glob
import json
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================
NOISY_DIR  = "./Dataset/noisy"
OUTPUT_DIR = "./Dataset/results"

GPT_MODEL    = "gpt-3.5-turbo"   # e.g. "gpt-4o", "gpt-3.5-turbo"
GEMINI_MODEL = "gemini-2.5-flash"

BATCH_SIZE           = 50
SLEEP_BETWEEN_CALLS  = 0.2   # seconds between batch calls
MAX_RETRIES          = 3
# ==============================================================================

_SYSTEM_PROMPT = "You are a content moderation classifier for hate speech. Output JSON only."

_RULE_HINT = (
    "Binary hate-speech classification.\n"
    "Label=1: text clearly contains hate / derogatory / abusive / harassing / "
    "bullying / dehumanizing / threatening / violent content toward a person or group "
    "(explicit or strongly implied) — including slurs, targeted insults, dehumanization, "
    "calls for exclusion or discrimination, or encouragement of harm.\n"
    "Label=0: unclear / ambiguous, non-targeted profanity, general rudeness without a "
    "clear target, sarcasm without clear abuse, or mere discussion / quoting without endorsement.\n"
    "Tie-break: choose 0 if unsure.\n"
)


def _is_rate_limit(e):
    s = str(e).lower()
    return "429" in s or "rate limit" in s or "quota" in s


# ==============================================================================
# GPT
# ==============================================================================

def _gpt_prompt(texts):
    n = len(texts)
    lines = [
        _RULE_HINT,
        f'Return JSON only: {{"predictions": [0 or 1, ...]}}',
        f"predictions length MUST be {n}, in input order.",
        "",
        "Inputs:",
    ]
    for i, t in enumerate(texts):
        lines.append(f"{i}: {str(t).replace(chr(10), ' ')}")
    return "\n".join(lines)


def _call_gpt(client, model, texts):
    n = len(texts)
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _gpt_prompt(texts)},
        ],
        text={"format": {"type": "json_object"}},
        temperature=0.0,
        max_output_tokens=2000,
    )
    if getattr(resp, "status", None) == "incomplete":
        raise RuntimeError(f"Incomplete GPT response: {resp.incomplete_details}")
    preds = json.loads(resp.output_text).get("predictions")
    if not isinstance(preds, list) or len(preds) != n:
        raise ValueError(f"Expected {n} predictions, got {preds!r}")
    if not all(p in (0, 1) for p in preds):
        raise ValueError(f"Non-binary prediction values: {preds}")
    return [int(p) for p in preds]


def _classify_gpt(client, model, texts):
    """Classify with GPT; splits batch recursively on failure."""
    if not texts:
        return []
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return _call_gpt(client, model, texts)
        except Exception as e:
            last_err = e
            wait = (2 ** attempt) if _is_rate_limit(e) else 0.5
            time.sleep(wait)
    if len(texts) == 1:
        print(f"\n  Single item failed (defaulting to 0): {last_err}")
        return [0]
    mid = len(texts) // 2
    return (_classify_gpt(client, model, texts[:mid]) +
            _classify_gpt(client, model, texts[mid:]))


def run_gpt():
    from openai import OpenAI

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment / .env")

    client = OpenAI(api_key=api_key)
    tag = re.sub(r"[^a-z0-9]+", "_", GPT_MODEL.lower()).strip("_")
    out_dir = os.path.join(OUTPUT_DIR, f"gpt_{tag}")
    pred_col = f"pred_gpt_{tag}"
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(NOISY_DIR, "hate_speech_400_p_*.csv")))
    if not files:
        raise FileNotFoundError(f"No noisy files in {NOISY_DIR}. Run add_noise.py first.")

    print(f"=== GPT ({GPT_MODEL}) — {len(files)} files ===")
    for file_path in tqdm(files, desc="Files"):
        out_path = os.path.join(out_dir, os.path.basename(file_path))
        if os.path.exists(out_path):
            tqdm.write(f"  Skip (exists): {os.path.basename(out_path)}")
            continue

        df = pd.read_csv(file_path, encoding="utf-8-sig")
        preds = []
        for start in tqdm(range(0, len(df), BATCH_SIZE), desc="  Batches", leave=False):
            preds.extend(_classify_gpt(client, GPT_MODEL, df["text"].iloc[start:start + BATCH_SIZE].tolist()))
            time.sleep(SLEEP_BETWEEN_CALLS)

        df[pred_col] = preds
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        tqdm.write(f"  Saved: {os.path.basename(out_path)}")

    print(f"GPT results -> {out_dir}/\n")


# ==============================================================================
# Gemini
# ==============================================================================

def _gemini_prompt(texts):
    n = len(texts)
    lines = [
        _RULE_HINT,
        f'Return JSON only: {{"predictions": {{"0": 0, "1": 1, ...}}}}',
        f'Keys are string IDs "0".."{n-1}". Missing key defaults to 0.',
        "",
        "Inputs:",
    ]
    for i, t in enumerate(texts):
        lines.append(f"{i}: {str(t).replace(chr(10), ' ')}")
    return "\n".join(lines)


def _parse_gemini(data, n):
    mapping = data.get("predictions", {})
    if not isinstance(mapping, dict):
        raise ValueError("'predictions' must be a dict of string-ID -> 0/1")
    return [int(mapping.get(str(i), 0)) for i in range(n)]


def _text_from_gemini_response(resp):
    """Concatenate only text parts, skipping thought_signature and similar."""
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        return ""
    parts = getattr(getattr(cands[0], "content", None), "parts", None) or []
    return "".join(getattr(p, "text", None) or "" for p in parts).strip()


def _call_gemini(client, model, safety, texts):
    from google.genai import types

    n = len(texts)
    resp = client.models.generate_content(
        model=model,
        contents=_gemini_prompt(texts),
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            safety_settings=safety,
        ),
    )
    text = _text_from_gemini_response(resp)
    if not text:
        raise RuntimeError("Empty response from Gemini")
    cleaned = re.sub(r"```json\s*|```\s*$", "", text).strip()
    return _parse_gemini(json.loads(cleaned), n)


def _classify_gemini(client, model, safety, texts):
    """Classify with Gemini; splits batch recursively on failure."""
    if not texts:
        return []
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return _call_gemini(client, model, safety, texts)
        except Exception as e:
            last_err = e
            wait = (2 ** (attempt + 1)) if _is_rate_limit(e) else 0.5
            time.sleep(wait)
    if len(texts) == 1:
        print(f"\n  Single item failed (defaulting to 0): {last_err}")
        return [0]
    mid = len(texts) // 2
    return (_classify_gemini(client, model, safety, texts[:mid]) +
            _classify_gemini(client, model, safety, texts[mid:]))


def run_gemini():
    from google import genai
    from google.genai import types

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment / .env")

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60000))
    safety = [
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]

    tag = re.sub(r"[^a-z0-9]+", "_", GEMINI_MODEL.lower()).strip("_")
    out_dir = os.path.join(OUTPUT_DIR, f"gemini_{tag}")
    pred_col = f"pred_gemini_{tag}"
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(NOISY_DIR, "hate_speech_400_p_*.csv")))
    if not files:
        raise FileNotFoundError(f"No noisy files in {NOISY_DIR}. Run add_noise.py first.")

    print(f"=== Gemini ({GEMINI_MODEL}) — {len(files)} files ===")
    for file_path in tqdm(files, desc="Files"):
        out_path = os.path.join(out_dir, os.path.basename(file_path))
        if os.path.exists(out_path):
            tqdm.write(f"  Skip (exists): {os.path.basename(out_path)}")
            continue

        df = pd.read_csv(file_path, encoding="utf-8-sig")
        preds = []
        for start in tqdm(range(0, len(df), BATCH_SIZE), desc="  Batches", leave=False):
            preds.extend(_classify_gemini(client, GEMINI_MODEL, safety, df["text"].iloc[start:start + BATCH_SIZE].tolist()))
            time.sleep(SLEEP_BETWEEN_CALLS)

        df[pred_col] = preds
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        tqdm.write(f"  Saved: {os.path.basename(out_path)}")

    print(f"Gemini results -> {out_dir}/\n")


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Get LLM responses for each noisy dataset")
    parser.add_argument(
        "--model", choices=["gpt", "gemini", "both"], default="gpt",
        help="Which LLM(s) to use (default: gpt)"
    )
    args = parser.parse_args()

    if args.model in ("gpt", "both"):
        run_gpt()
    if args.model in ("gemini", "both"):
        run_gemini()


if __name__ == "__main__":
    main()
