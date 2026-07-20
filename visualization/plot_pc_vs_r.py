"""
Plot Pc vs reward r for each LLM at fixed NSR = 0.2.

For each LLM i:
  - lambda_i is estimated from data (lambda = r_i / x_i, r_i = token price)
  - As r varies: x(r) = r / lambda_i
  - Pc(r) = ri_pc( x(r), q(NSR) )

q is computed from average alpha and beta across noise conditions.

Output: Dataset/results/pc_vs_r_nsr02.png
Usage : python visualization/plot_pc_vs_r.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SUMM_CSV = "./Dataset/results/bootstrap/bootstrap_summary.csv"
OUT_PNG   = "./Dataset/results/pc_vs_r_snr6db.png"
SNR_FIXED = 6.0  # dB  →  NSR = 10^(−SNR/10) → q ≈ 0.10

MODELS = [
    "GPT-3.5-turbo",
    "GPT-5.4-nano",
    "Gemini-2.5-Flash",
    "Gemini-2.5-Flash-Lite",
]
PRICE_MAP = {
    "GPT-3.5-turbo":         0.50,
    "GPT-5.4-nano":          0.20,
    "Gemini-2.5-Flash":      0.30,
    "Gemini-2.5-Flash-Lite": 0.10,
}
COLORS = {
    "GPT-3.5-turbo":         "#2196F3",   # blue
    "GPT-5.4-nano":          "#FF9800",   # orange
    "Gemini-2.5-Flash":      "#4CAF50",   # green
    "Gemini-2.5-Flash-Lite": "#9C27B0",   # purple
}
LLM_LABELS = {
    "GPT-3.5-turbo":         "GPT-3.5-turbo  (r=$0.50)",
    "GPT-5.4-nano":          "GPT-5.4-nano   (r=$0.20)",
    "Gemini-2.5-Flash":      "Gemini-2.5-Flash  (r=$0.30)",
    "Gemini-2.5-Flash-Lite": "Gemini-2.5-Flash-Lite  (r=$0.10)",
}

_EPS = 1e-10


# ---------------------------------------------------------------------------
# RI model (same as bootstrap_ri.py)
# ---------------------------------------------------------------------------

def _pa(x, q):
    ex  = np.exp(np.clip(x, -500, 500))
    num = (1 + q) * ex - (1 - q)
    den = 2 * ex - 2
    pa  = np.where(np.abs(den) < 1e-12, 0.5, num / den)
    return np.clip(pa, 0.5, 1.0 - _EPS)


def ri_pc(x, q):
    pa  = _pa(x, q)
    ex  = np.exp(np.clip(x, -500, 500))
    p1a = np.clip((pa * ex) / (pa * ex + (1 - pa)), 0.0, 1.0)
    p2b = np.clip(((1 - pa) * ex) / (pa + (1 - pa) * ex), 0.0, 1.0)
    return 0.5 * (1 - q) * p1a + 0.5 * (1 - q) * p2b + 0.5 * q


def q_of_nsr(nsr, alpha, beta):
    return float(np.clip(alpha * (nsr ** beta), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Load parameters
# ---------------------------------------------------------------------------

def load_params(summ_csv):
    df = pd.read_csv(summ_csv)

    # average lambda per LLM across noise conditions
    lam = {}
    for m in MODELS:
        rows = df[df["param"] == f"lambda_{m}"]
        lam[m] = rows["estimate"].mean()

    # average alpha and beta
    alpha = df[df["param"] == "alpha"]["estimate"].mean()
    beta  = df[df["param"] == "beta"]["estimate"].mean()

    return lam, alpha, beta


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_pc_vs_r(lam, alpha, beta, snr_db=SNR_FIXED, out_path=OUT_PNG):
    nsr = 10 ** (-snr_db / 10)
    q   = q_of_nsr(nsr, alpha, beta)
    print(f"SNR = {snr_db} dB  →  NSR = {nsr:.4f},  q = {q:.4f}")

    r_vals = np.linspace(0.001, 1.0, 500)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for m in MODELS:
        lam_i = lam[m]
        r_i   = PRICE_MAP[m]
        color = COLORS[m]

        x_vals  = r_vals / lam_i
        pc_vals = ri_pc(x_vals, q)

        ax.plot(r_vals, pc_vals, color=color, linewidth=2, label=LLM_LABELS[m])

        x_op  = r_i / lam_i
        pc_op = ri_pc(x_op, q)
        ax.scatter([r_i], [pc_op], color=color, s=70, zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.axvline(r_i, color=color, linewidth=0.8, linestyle=":", alpha=0.5)

        print(f"  {m:<28}  lambda={lam_i:.4f}  Pc at r={r_i:.2f}: {pc_op:.4f}")

    ax.set_xlabel("Reward  r  ($/1M input tokens)", fontsize=11)
    ax.set_ylabel("Correct action probability  Pc", fontsize=11)
    ax.set_title(
        f"LLM Classification Accuracy vs Reward  (SNR = {snr_db} dB,  q ≈ {q:.2f})",
        fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.48, 1.01)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6,
               label="Chance (Pc = 0.5)")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lam, alpha, beta = load_params(SUMM_CSV)
    print("Average lambda per LLM:")
    for m in MODELS:
        print(f"  {m:<28}  lambda = {lam[m]:.4f}  (r = ${PRICE_MAP[m]:.2f})")
    print(f"\nAverage alpha = {alpha:.4f},  beta = {beta:.4f}")
    plot_pc_vs_r(lam, alpha, beta, snr_db=SNR_FIXED)
