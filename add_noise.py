"""
Generate noisy versions of the hate-speech sample at 11 noise levels.

Input   : Dataset/hate_speech_binary_400.csv   (produced by download_data.py)
Output  : Dataset/noisy/hate_speech_1000_p_00.csv  (p = 0.0, no noise)
          Dataset/noisy/hate_speech_1000_p_01.csv  (p = 0.1)
          ...
          Dataset/noisy/hate_speech_1000_p_10.csv  (p = 1.0, max noise)

Noise model
-----------
p  : probability that each word token is perturbed
R  : when a word is chosen, ceil(R * word_length) characters are edited
     (applies to insert / replace / delete; swap always changes 2 chars)

Operation weights (must sum to 1):
  swap    — randomly exchange two characters within the word
  insert  — insert k random characters at random positions
  replace — replace k characters with random ones
  delete  — delete k characters (keeping at least 1)

Seeds are derived per (noise_level, row_index) so each file is fully
reproducible independently of how many files have already been generated.
"""

import math
import os
import re
import string

import numpy as np
import pandas as pd

# ==============================================================================
# CONFIG
# ==============================================================================
INPUT_PATH = "./Dataset/hate_speech_binary_400.csv"
OUTPUT_DIR = "./Dataset/noisy"
SEED       = 42

# Per-word perturbation probability steps: p = k/10 for k in 0..10
# (generates 11 files)

# Character-edit fraction when a word is perturbed
R = 1.0

# Operation distribution: swap / insert / replace / delete
WEIGHTS = [0.60, 0.10, 0.15, 0.15]
# ==============================================================================

_OPS      = ["swap", "insert", "replace", "delete"]
_ALPHABET = list(string.ascii_letters + string.digits)
_TOKENIZER = re.compile(r"\w+|\s+|[^\w\s]")


def _swap(word, rng):
    if len(word) < 2:
        return word
    i, j = rng.choice(len(word), size=2, replace=False)
    chars = list(word)
    chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)


def _insert(word, rng, k):
    for _ in range(k):
        pos = rng.integers(0, len(word) + 1)
        word = word[:pos] + rng.choice(_ALPHABET) + word[pos:]
    return word


def _replace(word, rng, k):
    for _ in range(k):
        pos = rng.integers(0, len(word))
        word = word[:pos] + rng.choice(_ALPHABET) + word[pos + 1:]
    return word


def _delete(word, rng, k):
    k = min(k, len(word) - 1)
    for _ in range(k):
        if len(word) <= 1:
            break
        pos = rng.integers(0, len(word))
        word = word[:pos] + word[pos + 1:]
    return word


def _perturb(word, rng):
    op = rng.choice(_OPS, p=WEIGHTS)
    if op == "swap":
        return _swap(word, rng)
    k = max(1, math.ceil(R * len(word)))
    if op == "insert":
        return _insert(word, rng, k)
    if op == "replace":
        return _replace(word, rng, k)
    return _delete(word, rng, k)


def add_noise(text, rng, p):
    """Perturb each word token independently with probability p."""
    if not text or (isinstance(text, float) and math.isnan(text)):
        return ""
    tokens = _TOKENIZER.findall(str(text))
    return "".join(
        _perturb(tok, rng) if tok and tok[0].isalnum() and rng.random() < p else tok
        for tok in tokens
    )


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}\nRun download_data.py first.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        df0 = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df0 = pd.read_csv(INPUT_PATH, encoding="latin1")

    n = len(df0)
    print(f"Loaded {n} rows from {INPUT_PATH}")
    print(f"Noise weights (swap/insert/replace/delete): {WEIGHTS},  R={R}")
    print(f"Generating 11 files to {OUTPUT_DIR}/\n")

    for k in range(11):
        p = k / 10.0
        df = df0.copy(deep=True)

        # Each (noise_level, row) gets its own reproducible seed
        for i in range(n):
            rng = np.random.default_rng(SEED + k * 1000 + i)
            df.at[i, "text"] = add_noise(df.at[i, "text"], rng, p)

        out_path = os.path.join(OUTPUT_DIR, f"hate_speech_400_p_{k:02d}.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  p={p:.1f}  ->  {os.path.basename(out_path)}")

    print(f"\nDone. 11 files written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
