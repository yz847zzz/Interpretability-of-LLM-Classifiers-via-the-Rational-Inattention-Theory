"""
Plot bootstrap distributions of all RI parameters (lambda, alpha, beta).

Reads:
    Dataset/results/bootstrap/bootstrap_raw.csv
    Dataset/results/bootstrap/original_estimates.csv

Output:
    Dataset/results/bootstrap/dist_lambda.png   — lambda per model × noise
    Dataset/results/bootstrap/dist_alpha_beta.png — shared params across noise

Usage:
    python plot_bootstrap_dist.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "./Dataset/results"
NOISE_TYPES = ["babble", "cafe", "white"]
NOISE_COLORS = {"babble": "steelblue", "cafe": "darkorange", "white": "seagreen"}

MODEL_LABELS = {
    "lambda_GPT-3.5-turbo":         "GPT-3.5-turbo",
    "lambda_GPT-5.4-nano":          "GPT-5.4-nano",
    "lambda_Gemini-2.5-Flash":      "Gemini-2.5-Flash",
    "lambda_Gemini-2.5-Flash-Lite": "Gemini-2.5-Flash-Lite",
}


def main():
    out_dir = os.path.join(RESULTS_DIR, "bootstrap")

    raw  = pd.read_csv(os.path.join(out_dir, "bootstrap_raw.csv"))
    orig = pd.read_csv(os.path.join(out_dir, "original_estimates.csv"))

    def get_est(noise, param):
        s = orig[(orig["noise"] == noise) & (orig["param"] == param)]["estimate"]
        return float(s.iloc[0]) if not s.empty else None

    # ── Figure 1: lambda distributions ──────────────────────────────────────
    lambda_cols = list(MODEL_LABELS.keys())
    fig, axes = plt.subplots(len(lambda_cols), len(NOISE_TYPES),
                             figsize=(12, 9), sharey=False)

    for row_i, col in enumerate(lambda_cols):
        for col_j, noise in enumerate(NOISE_TYPES):
            ax = axes[row_i, col_j]
            vals = raw[raw["noise"] == noise][col].dropna().values
            est  = get_est(noise, col)

            ax.hist(vals, bins=40, color=NOISE_COLORS[noise],
                    alpha=0.75, edgecolor="white", linewidth=0.4)
            if est is not None:
                ax.axvline(est, color="red", linewidth=2,
                           label=f"est = {est:.4f}")
            ax.axvline(np.percentile(vals, 2.5),  color="gray",
                       linewidth=1, linestyle="--", alpha=0.7)
            ax.axvline(np.percentile(vals, 97.5), color="gray",
                       linewidth=1, linestyle="--", alpha=0.7, label="95% CI")

            if row_i == 0:
                ax.set_title(noise.capitalize(), fontsize=10, fontweight="bold")
            if col_j == 0:
                ax.set_ylabel(MODEL_LABELS[col], fontsize=8)
            ax.set_xlabel("λ", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.tick_params(labelsize=7)

    fig.suptitle("Bootstrap distributions of λ  (B=1000, red = original estimate, dashed = 95% CI)",
                 fontsize=11)
    fig.tight_layout()
    fname = os.path.join(out_dir, "dist_lambda.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved -> {fname}")

    # ── Figure 2: alpha & beta distributions ────────────────────────────────
    fig, axes = plt.subplots(2, len(NOISE_TYPES), figsize=(12, 6), sharey=False)

    for col_j, noise in enumerate(NOISE_TYPES):
        for row_i, param in enumerate(["alpha", "beta"]):
            ax = axes[row_i, col_j]
            vals = raw[raw["noise"] == noise][param].dropna().values
            est  = get_est(noise, param)

            ax.hist(vals, bins=40, color=NOISE_COLORS[noise],
                    alpha=0.75, edgecolor="white", linewidth=0.4)
            if est is not None:
                ax.axvline(est, color="red", linewidth=2,
                           label=f"est = {est:.4f}")
            ax.axvline(np.percentile(vals, 2.5),  color="gray",
                       linewidth=1, linestyle="--", alpha=0.7)
            ax.axvline(np.percentile(vals, 97.5), color="gray",
                       linewidth=1, linestyle="--", alpha=0.7, label="95% CI")

            if row_i == 0:
                ax.set_title(noise.capitalize(), fontsize=10, fontweight="bold")
            if col_j == 0:
                ax.set_ylabel("α" if param == "alpha" else "β", fontsize=10)
            ax.set_xlabel("α" if param == "alpha" else "β", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.tick_params(labelsize=7)

    fig.suptitle("Bootstrap distributions of α and β  (B=1000, shared params)",
                 fontsize=11)
    fig.tight_layout()
    fname = os.path.join(out_dir, "dist_alpha_beta.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved -> {fname}")

    # ── Figure 3: overlay — all noise conditions per parameter ───────────────
    all_params = lambda_cols + ["alpha", "beta"]
    param_labels = {**MODEL_LABELS, "alpha": "α (shared)", "beta": "β (shared)"}

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes_flat = axes.flatten()

    for i, param in enumerate(all_params):
        ax = axes_flat[i]
        for noise in NOISE_TYPES:
            vals = raw[raw["noise"] == noise][param].dropna().values
            est  = get_est(noise, param)
            ax.hist(vals, bins=40, color=NOISE_COLORS[noise],
                    alpha=0.5, edgecolor="none", label=noise)
            if est is not None:
                ax.axvline(est, color=NOISE_COLORS[noise],
                           linewidth=2, linestyle="--")
        ax.set_title(param_labels.get(param, param), fontsize=9)
        ax.set_xlabel("value", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Bootstrap distributions — all parameters overlaid by noise type\n"
                 "(dashed lines = original estimates)", fontsize=11)
    fig.tight_layout()
    fname = os.path.join(out_dir, "dist_all_overlay.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved -> {fname}")

    print("\nDone.")


if __name__ == "__main__":
    main()
