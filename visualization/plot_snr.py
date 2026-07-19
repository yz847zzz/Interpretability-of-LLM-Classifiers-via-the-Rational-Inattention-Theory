"""
Plot Pc vs SNR (S/N power ratio) from fitted RI audio model results.

Reads:
  {results_dir}/ri_fit/fitted_params.csv   — per-model x, shared alpha, beta
  {results_dir}/{model_dir}/summary_*.csv  — observed Pc at each NSR level

Outputs:
  {results_dir}/ri_fit/SNR_Plot/
      {model}_fit.png          — empirical + fitted Pc vs SNR per model
      all_models_fit.png       — all models overlaid

Usage:
    python plot_snr.py --results-dir ./Dataset/results/audio_babble
    python plot_snr.py --results-dir ./Dataset/results/audio_cafe
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# RI model functions (same as fit_ri_model_audio.py)
_EPS = 1e-10

def _pa(x, q):
    ex  = np.exp(np.clip(x, -500, 500))
    num = (1 + q) * ex - (1 - q)
    den = 2 * ex - 2
    pa  = np.where(np.abs(den) < 1e-12, 0.5, num / den)
    return np.clip(pa, 0.5, 1.0 - _EPS)

def ri_pc(x, q):
    pa = _pa(x, q)
    ex = np.exp(np.clip(x, -500, 500))
    p1a = np.clip((pa * ex) / (pa * ex + (1 - pa)), 0.0, 1.0)
    p2b = np.clip(((1 - pa) * ex) / (pa + (1 - pa) * ex), 0.0, 1.0)
    return 0.5 * (1 - q) * p1a + 0.5 * (1 - q) * p2b + 0.5 * q

def q_of_nsr(nsr, alpha, beta):
    return np.clip(alpha * np.power(np.maximum(nsr, 0.0), beta), 0.0, 1.0)

def pc_from_snr(snr_ratio, x, alpha, beta):
    """snr_ratio = S/N; convert to NSR then compute Pc."""
    nsr = np.where(snr_ratio > 0, 1.0 / snr_ratio, 0.0)
    return ri_pc(x, q_of_nsr(nsr, alpha, beta))


def main():
    parser = argparse.ArgumentParser(description="Plot Pc vs SNR from audio RI fit")
    parser.add_argument("--results-dir", required=True,
                        help="Results dir containing ri_fit/ and model subdirs")
    args = parser.parse_args()

    ri_dir     = os.path.join(args.results_dir, "ri_fit")
    out_dir    = os.path.join(ri_dir, "SNR_Plot")
    noise_type = os.path.basename(args.results_dir).replace("audio_", "")
    os.makedirs(out_dir, exist_ok=True)

    # Load fitted params
    params_path = os.path.join(ri_dir, "fitted_params.csv")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"No fitted_params.csv in {ri_dir}. Run fit_ri_model_audio.py first.")
    params_df = pd.read_csv(params_path)
    alpha = float(params_df["alpha"].iloc[0])
    beta  = float(params_df["beta"].iloc[0])

    # SNR range for fitted curve — extend to 10^4
    snr_fine = np.logspace(-2, 4, 500)   # 0.01 .. 10000

    MODEL_COLORS = {
        "GPT-3.5-turbo":         "black",
        "GPT-5.4-nano":          "red",
        "Gemini-2.5-Flash":      "goldenrod",
        "Gemini-2.5-Flash-Lite": "green",
    }

    model_data = []
    for _, row in params_df.iterrows():
        label    = row["model"]
        x        = float(row["x_r_over_lambda"])
        dir_name = label.lower().replace(" ", "_").replace(".", "_").replace("-", "_")

        # find summary CSV
        candidates = (
            glob.glob(os.path.join(args.results_dir, f"*{dir_name}*", "summary_*.csv")) +
            glob.glob(os.path.join(args.results_dir, f"*{dir_name.replace('__','_')}*", "summary_*.csv"))
        )
        # try a looser match
        if not candidates:
            for d in os.listdir(args.results_dir):
                tag = d.replace("gpt_", "").replace("gemini_", "")
                if tag in dir_name or dir_name.replace("gpt_","") in tag:
                    candidates = glob.glob(os.path.join(args.results_dir, d, "summary_*.csv"))
                    if candidates:
                        break

        if not candidates:
            print(f"  WARNING: no summary CSV found for '{label}' — skipping.")
            continue

        df = pd.read_csv(candidates[0])
        df["noise_p"] = pd.to_numeric(df["noise_p"], errors="coerce")
        df = df.dropna(subset=["noise_p", "Pc"])

        # Convert NSR → SNR: skip NSR=0 (infinite SNR)
        obs = df[df["noise_p"] > 0].copy()
        obs["snr_ratio"] = 1.0 / obs["noise_p"]

        model_data.append({
            "label":     label,
            "x":         x,
            "lambda":    float(row["lambda"]),
            "R2":        float(row["R2_Pc"]),
            "obs_snr":   obs["snr_ratio"].values,
            "obs_Pc":    obs["Pc"].values,
            "fit_Pc":    pc_from_snr(snr_fine, x, alpha, beta),
        })

    if not model_data:
        raise RuntimeError("No model data loaded.")

    # Per-model plots
    for m in model_data:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(m["obs_snr"], m["obs_Pc"], color="red", zorder=3,
                   label=r"$\hat{P}^{(Y)}_{correct}$ (Observed)")
        ax.plot(snr_fine, m["fit_Pc"], color="blue", linewidth=2,
                label=f"$P^{{(Y)}}_{{correct}}$ (Fitted, x={m['x']:.3f}, λ={m['lambda']:.3f})")
        ax.set_xscale("log")
        ax.set_xlabel("SNR")
        ax.set_ylabel(r"$P^{(Y)}_{correct}$")
        ax.set_title(f"{m['label']}   α={alpha:.4f}, β={beta:.4f}   R²={m['R2']:.4f}")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        fig.tight_layout()
        fname = os.path.join(out_dir,
                             f"{m['label'].replace('-','').replace(' ','_')}_fit.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Plot -> {fname}")

    # All models overlaid
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in model_data:
        c = MODEL_COLORS.get(m["label"], "gray")
        ax.scatter(m["obs_snr"], m["obs_Pc"], color=c, zorder=3, s=40)
        ax.plot(snr_fine, m["fit_Pc"], color=c, linewidth=2, label=m["label"])
    ax.set_xscale("log")
    ax.set_xlabel("SNR")
    ax.set_ylabel(r"$P^{(Y)}_{correct}$")
    ax.set_title(f"RI fits — {noise_type}   α={alpha:.4f}, β={beta:.4f}")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fname = os.path.join(out_dir, "all_models_fit.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Plot -> {fname}")

    # regression lines only (no scatter)
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in model_data:
        c = MODEL_COLORS.get(m["label"], "gray")
        ax.plot(snr_fine, m["fit_Pc"], color=c, linewidth=2, label=m["label"])
    ax.set_xscale("log")
    ax.set_xlabel("SNR")
    ax.set_ylabel(r"$P^{(Y)}_{correct}$")
    ax.set_title(f"RI fits — {noise_type}   α={alpha:.4f}, β={beta:.4f}")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fname = os.path.join(out_dir, "all_models_fit_regression_only.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Plot -> {fname}")

    # q vs SNR
    q_fine = q_of_nsr(1.0 / snr_fine, alpha, beta)   # NSR = 1/SNR
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(snr_fine, q_fine, color="darkorange", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("SNR")
    ax.set_ylabel("hate speech hiding rate")
    ax.set_title(f"Noise mapping  q(SNR) = min(α·(1/SNR)^β, 1)\nα={alpha:.4f}, β={beta:.4f}")
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fname = os.path.join(out_dir, "q_vs_SNR.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Plot -> {fname}")

    print(f"\nDone. Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
