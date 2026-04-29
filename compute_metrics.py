"""
Compute summary statistics for all LLM result files.

Input   : Dataset/results/<model>/hate_speech_400_p_*.csv
            (produced by get_llm_responses.py; must contain a "label" column
            and a prediction column named "pred_<model>")

Output  : Dataset/results/<model>/summary_<model>.csv
            one row per noise level, columns:
              noise_p  — noise probability (0.0 to 1.0)
              P1a      — sensitivity   P(pred=1 | label=1)
              P1b      — specificity   P(pred=0 | label=0)
              Pc       — accuracy      P(Y=1)*P1a + P(Y=0)*P1b
              IYA_nats — mutual information I(Y;A) in nats

Usage
-----
  python compute_metrics.py --model gpt
  python compute_metrics.py --model gemini
  python compute_metrics.py --model both
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

# ==============================================================================
# CONFIG
# ==============================================================================
RESULTS_DIR = "./Dataset/results"
TRUTH_COL   = "label"   # ground-truth column created by download_data.py

# These must match the values used in get_llm_responses.py
GPT_MODEL    = "gpt-3.5-turbo"
GEMINI_MODEL = "gemini-2.5-flash"
# ==============================================================================


def _entropy(probs):
    """Shannon entropy in nats (natural log); treats 0*log(0) = 0."""
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _mutual_info(p1, p1a, p1b):
    """
    I(Y;A) in nats for binary Y and A.
    p1  = P(Y=1)
    p1a = P(A=1 | Y=1)  [sensitivity]
    p1b = P(A=0 | Y=0)  [specificity]
    """
    p0  = 1.0 - p1
    p0a = 1.0 - p1a   # P(A=0 | Y=1)
    p0b = 1.0 - p1b   # P(A=1 | Y=0)

    pA1 = p1 * p1a + p0 * p0b   # marginal P(A=1)
    pA0 = 1.0 - pA1

    H_A         = _entropy([pA0, pA1])
    H_A_given_Y = p1 * _entropy([p0a, p1a]) + p0 * _entropy([p1b, p0b])
    return float(H_A - H_A_given_Y)


def _noise_level(filename):
    m = re.search(r"_p_(\d+)", filename)
    return int(m.group(1)) / 10.0 if m else -1.0


def compute_summary(model_dir, pred_col, model_label):
    files = sorted(glob.glob(os.path.join(model_dir, "hate_speech_400_p_*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No result files found in {model_dir}.\n"
            f"Run get_llm_responses.py first."
        )

    rows = []
    for file_path in files:
        fname = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin1")

        if TRUTH_COL not in df.columns:
            print(f"  Skipping {fname}: '{TRUTH_COL}' column missing")
            continue
        if pred_col not in df.columns:
            print(f"  Skipping {fname}: '{pred_col}' column missing")
            continue

        pred  = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
        truth = df[TRUTH_COL]

        p1  = float(truth.mean())
        p0  = 1.0 - p1
        p1a = float(pred[truth == 1].mean()) if (truth == 1).sum() > 0 else 0.0
        p1b = float(1.0 - pred[truth == 0].mean()) if (truth == 0).sum() > 0 else 0.0
        pc  = p1 * p1a + p0 * p1b
        iya = _mutual_info(p1, p1a, p1b)

        rows.append({
            "noise_p":  _noise_level(fname),
            "filename": fname,
            "P1a":      round(p1a, 6),
            "P1b":      round(p1b, 6),
            "Pc":       round(pc,  6),
            "IYA_nats": round(iya, 6),
        })

    summary = pd.DataFrame(rows).sort_values("noise_p").reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f" {model_label}")
    print(f"{'='*60}")
    print(f"  {'noise_p':>8}  {'P1a':>8}  {'P1b':>8}  {'Pc':>8}  {'I(Y;A)':>10}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    for _, row in summary.iterrows():
        print(f"  {row.noise_p:>8.1f}  {row.P1a:>8.4f}  {row.P1b:>8.4f}  {row.Pc:>8.4f}  {row.IYA_nats:>10.6f}")

    out_path = os.path.join(model_dir, f"summary_{model_label}.csv")
    summary.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}\n")
    return summary


def _make_tag(model_name):
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def main():
    parser = argparse.ArgumentParser(description="Compute P1a / P1b / Pc / I(Y;A) across noise levels")
    parser.add_argument(
        "--model", choices=["gpt", "gemini", "both"], default="gpt",
        help="Which model results to summarise (default: gpt)"
    )
    args = parser.parse_args()

    if args.model in ("gpt", "both"):
        tag = _make_tag(GPT_MODEL)
        compute_summary(
            model_dir   = os.path.join(RESULTS_DIR, f"gpt_{tag}"),
            pred_col    = f"pred_gpt_{tag}",
            model_label = f"gpt_{tag}",
        )

    if args.model in ("gemini", "both"):
        tag = _make_tag(GEMINI_MODEL)
        compute_summary(
            model_dir   = os.path.join(RESULTS_DIR, f"gemini_{tag}"),
            pred_col    = f"pred_gemini_{tag}",
            model_label = f"gemini_{tag}",
        )


if __name__ == "__main__":
    main()
