"""
Classify each noisy dataset using GPT or Gemini.

Input   : Dataset/noisy/hate_speech_*_p_*.csv  (produced by add_noise.py)
Output  : Dataset/results/<model>/hate_speech_*_p_*.csv
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
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================
NOISY_DIR  = "./Dataset/text_noise"
OUTPUT_DIR = "./Dataset/results"

GPT_MODEL    = "gpt-3.5-turbo"   # e.g. "gpt-4o", "gpt-3.5-turbo"
GEMINI_MODEL = "gemini-2.5-flash"

MAX_WORKERS          = 10    # parallel API calls per file
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

def _gpt_prompt_single(text):
    return (
        f"{_RULE_HINT}"
        f'Return JSON only: {{"label": 0 or 1}}\n\n'
        f"Text: {str(text).replace(chr(10), ' ')}"
    )


def _call_gpt_single(client, model, text):
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _gpt_prompt_single(text)},
        ],
        text={"format": {"type": "json_object"}},
        temperature=0.0,
        max_output_tokens=16,
    )
    if getattr(resp, "status", None) == "incomplete":
        raise RuntimeError(f"Incomplete GPT response: {resp.incomplete_details}")
    label = json.loads(resp.output_text).get("label")
    if label not in (0, 1):
        raise ValueError(f"Non-binary label: {label!r}")
    return int(label)


def _classify_gpt_single(client, model, text):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return _call_gpt_single(client, model, text)
        except Exception as e:
            last_err = e
            wait = (2 ** attempt) if _is_rate_limit(e) else 0.5
            time.sleep(wait)
    print(f"\n  Item failed (defaulting to 0): {last_err}")
    return 0


def run_gpt(model=None, input_dir=None, results_dir=None):
    from openai import OpenAI

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment / .env")

    input_dir   = input_dir   or NOISY_DIR
    results_dir = results_dir or OUTPUT_DIR
    model = model or GPT_MODEL
    client = OpenAI(api_key=api_key)
    tag = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    out_dir = os.path.join(results_dir, f"gpt_{tag}")
    pred_col = f"pred_gpt_{tag}"
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "hate_speech_*.csv")))
    if not files:
        raise FileNotFoundError(f"No files in {input_dir}.")

    print(f"=== GPT ({model}) — {len(files)} files, {MAX_WORKERS} parallel workers ===")
    for file_path in tqdm(files, desc="Files"):
        out_path = os.path.join(out_dir, os.path.basename(file_path))
        if os.path.exists(out_path):
            tqdm.write(f"  Skip (exists): {os.path.basename(out_path)}")
            continue

        df = pd.read_csv(file_path, encoding="utf-8-sig")
        texts = df["text"].tolist()
        preds = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(_classify_gpt_single, client, model, text): i
                for i, text in enumerate(texts)
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="  Rows", leave=False):
                preds[futures[fut]] = fut.result()

        df[pred_col] = preds
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        tqdm.write(f"  Saved: {os.path.basename(out_path)}")

    print(f"GPT results -> {out_dir}/\n")


# ==============================================================================
# Gemini
# ==============================================================================

def _gemini_prompt_single(text):
    return (
        f"{_RULE_HINT}"
        f'Return JSON only: {{"label": 0 or 1}}\n\n'
        f"Text: {str(text).replace(chr(10), ' ')}"
    )


def _text_from_gemini_response(resp):
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        return ""
    parts = getattr(getattr(cands[0], "content", None), "parts", None) or []
    return "".join(getattr(p, "text", None) or "" for p in parts).strip()


def _call_gemini_single(client, model, safety, text):
    from google.genai import types

    resp = client.models.generate_content(
        model=model,
        contents=_gemini_prompt_single(text),
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            safety_settings=safety,
        ),
    )
    raw = _text_from_gemini_response(resp)
    if not raw:
        raise RuntimeError("Empty response from Gemini")
    cleaned = re.sub(r"```json\s*|```\s*$", "", raw).strip()
    label = json.loads(cleaned).get("label")
    if label not in (0, 1):
        raise ValueError(f"Non-binary label: {label!r}")
    return int(label)


def _classify_gemini_single(client, model, safety, text):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return _call_gemini_single(client, model, safety, text)
        except Exception as e:
            last_err = e
            wait = (2 ** (attempt + 1)) if _is_rate_limit(e) else 0.5
            time.sleep(wait)
    print(f"\n  Item failed (defaulting to 0): {last_err}")
    return 0


def run_gemini(model=None, input_dir=None, results_dir=None):
    from google import genai
    from google.genai import types

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment / .env")

    input_dir   = input_dir   or NOISY_DIR
    results_dir = results_dir or OUTPUT_DIR
    model = model or GEMINI_MODEL
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60000))
    safety = [
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]

    tag = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    out_dir = os.path.join(results_dir, f"gemini_{tag}")
    pred_col = f"pred_gemini_{tag}"
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "hate_speech_*.csv")))
    if not files:
        raise FileNotFoundError(f"No files in {input_dir}.")

    print(f"=== Gemini ({model}) — {len(files)} files, {MAX_WORKERS} parallel workers ===")
    for file_path in tqdm(files, desc="Files"):
        out_path = os.path.join(out_dir, os.path.basename(file_path))
        if os.path.exists(out_path):
            tqdm.write(f"  Skip (exists): {os.path.basename(out_path)}")
            continue

        df = pd.read_csv(file_path, encoding="utf-8-sig")
        texts = df["text"].tolist()
        preds = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(_classify_gemini_single, client, model, safety, text): i
                for i, text in enumerate(texts)
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="  Rows", leave=False):
                preds[futures[fut]] = fut.result()

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
    parser.add_argument("--gpt-model",    default=None, help=f"GPT model ID (default: {GPT_MODEL})")
    parser.add_argument("--gemini-model", default=None, help=f"Gemini model ID (default: {GEMINI_MODEL})")
    parser.add_argument("--input-dir",    default=None, help=f"Input CSV dir (default: {NOISY_DIR})")
    parser.add_argument("--results-dir",  default=None, help=f"Results output dir (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    if args.model in ("gpt", "both"):
        run_gpt(model=args.gpt_model, input_dir=args.input_dir, results_dir=args.results_dir)
    if args.model in ("gemini", "both"):
        run_gemini(model=args.gemini_model, input_dir=args.input_dir, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
