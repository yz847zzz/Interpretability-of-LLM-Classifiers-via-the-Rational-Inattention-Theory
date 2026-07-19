"""
Compute summary statistics for audio-noise LLM result files.

Input   : Dataset/results/audio_<noise>/<model>/hate_speech_*_snr_*.csv
Output  : Dataset/results/audio_<noise>/<model>/summary_<model>.csv
            one row per SNR level, columns:
              noise_p  — SNR in dB (30 → clean, -20 → noisiest)
              P1a, P2a, P1b, P2b, Pc, IYA_nats  (same definition as compute_metrics.py)

Usage
-----
  python compute_metrics_audio.py --model both --results-dir ./Dataset/results/audio_babble
  python compute_metrics_audio.py --model gpt  --gpt-model gpt-5.4-nano --results-dir ./Dataset/results/audio_babble
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
RESULTS_DIR  = "./Dataset/results/audio_babble"
TRUTH_COL    = "label"

GPT_MODEL    = "gpt-3.5-turbo"
GEMINI_MODEL = "gemini-2.5-flash"
# ==============================================================================


def _entropy(probs):
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _mutual_info(p1, p1a, p1b):
    p0  = 1.0 - p1
    p0a = 1.0 - p1a
    p0b = 1.0 - p1b
    pA1 = p1 * p1a + p0 * p0b
    pA0 = 1.0 - pA1
    H_A         = _entropy([pA0, pA1])
    H_A_given_Y = p1 * _entropy([p0a, p1a]) + p0 * _entropy([p1b, p0b])
    return float(H_A - H_A_given_Y)


def _snr_from_filename(filename):
    """
    hate_speech_100_p_00.csv      → NSR = 0.0   (noise-free baseline)
    hate_speech_100_snr_p30db.csv → NSR = 0.001
    hate_speech_100_snr_m20db.csv → NSR = 100.0
    NSR = N/S = 10^(-SNR_dB/10)
    """
    # noise-free baseline from text_noise pipeline
    if re.search(r"_p_00\.csv$", filename):
        return 0.0
    m = re.search(r"snr_([mp])(\d+)db", filename)
    if not m:
        return None
    sign = 1 if m.group(1) == "p" else -1
    snr_db = sign * int(m.group(2))
    return 10 ** (-snr_db / 10)   # NSR = N/S power ratio


def compute_summary(model_dir, pred_col, model_label):
    files = sorted(
        glob.glob(os.path.join(model_dir, "hate_speech_*snr*.csv")) +
        glob.glob(os.path.join(model_dir, "hate_speech_*_p_00.csv"))
    )
    if not files:
        raise FileNotFoundError(
            f"No audio-noise result files found in {model_dir}.\n"
            f"Run get_llm_responses.py --input-dir first."
        )

    rows = []
    for file_path in files:
        fname = os.path.basename(file_path)
        snr   = _snr_from_filename(fname)
        if snr is None:
            print(f"  Skipping {fname}: cannot parse SNR")
            continue

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

        p_s2 = float(truth.mean())
        p_s1 = 1.0 - p_s2

        p1a = float(1.0 - pred[truth == 0].mean()) if (truth == 0).sum() > 0 else 0.0
        p1b = float(pred[truth == 0].mean())        if (truth == 0).sum() > 0 else 0.0
        p2a = float(1.0 - pred[truth == 1].mean()) if (truth == 1).sum() > 0 else 0.0
        p2b = float(pred[truth == 1].mean())        if (truth == 1).sum() > 0 else 0.0

        pc  = p_s2 * p2b + p_s1 * p1a
        iya = _mutual_info(p_s2, p2b, p1a)

        rows.append({
            "noise_p":  snr,        # SNR in dB (higher = cleaner)
            "filename": fname,
            "P1a":      round(p1a, 6),
            "P2a":      round(p2a, 6),
            "P1b":      round(p1b, 6),
            "P2b":      round(p2b, 6),
            "Pc":       round(pc,  6),
            "IYA_nats": round(iya, 6),
        })

    # sort descending by SNR: 30 → -20  (clean → noisy, analogous to p 0→1)
    summary = pd.DataFrame(rows).sort_values("noise_p", ascending=True).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f" {model_label}  (noise_p = SNR in dB)")
    print(f"{'='*60}")
    print(f"  {'NSR(N/S)':>10}  {'P1a':>8}  {'P2a':>8}  {'P1b':>8}  {'P2b':>8}  {'Pc':>8}  {'I(Y;A)':>10}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    for _, row in summary.iterrows():
        print(f"  {row.noise_p:>10.4f}  {row.P1a:>8.4f}  {row.P2a:>8.4f}  "
              f"{row.P1b:>8.4f}  {row.P2b:>8.4f}  {row.Pc:>8.4f}  {row.IYA_nats:>10.6f}")

    out_path = os.path.join(model_dir, f"summary_{model_label}.csv")
    summary.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}\n")
    return summary


def _make_tag(model_name):
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def main():
    parser = argparse.ArgumentParser(
        description="Compute P1a/P2a/Pc/I(Y;A) for audio-noise LLM results"
    )
    parser.add_argument("--model", choices=["gpt", "gemini", "both"], default="both")
    parser.add_argument("--gpt-model",    default=None, help=f"GPT model (default: {GPT_MODEL})")
    parser.add_argument("--gemini-model", default=None, help=f"Gemini model (default: {GEMINI_MODEL})")
    parser.add_argument("--results-dir",  default=None, help=f"Results dir (default: {RESULTS_DIR})")
    args = parser.parse_args()

    results_dir = args.results_dir or RESULTS_DIR

    if args.model in ("gpt", "both"):
        tag = _make_tag(args.gpt_model or GPT_MODEL)
        compute_summary(
            model_dir   = os.path.join(results_dir, f"gpt_{tag}"),
            pred_col    = f"pred_gpt_{tag}",
            model_label = f"gpt_{tag}",
        )

    if args.model in ("gemini", "both"):
        tag = _make_tag(args.gemini_model or GEMINI_MODEL)
        compute_summary(
            model_dir   = os.path.join(results_dir, f"gemini_{tag}"),
            pred_col    = f"pred_gemini_{tag}",
            model_label = f"gemini_{tag}",
        )


if __name__ == "__main__":
    main()
