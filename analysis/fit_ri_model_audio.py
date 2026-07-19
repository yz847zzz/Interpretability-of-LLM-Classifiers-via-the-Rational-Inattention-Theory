"""
Fit the extended RI model to audio-noise LLM classification results.

Noise-to-channel mapping (audio version):
    q(NSR) = min( alpha * NSR^beta, 1 )    where NSR = N/S power ratio

    Same functional form as text-noise model q(p') = min(alpha*p'^beta, 1).
    alpha — scale parameter (shared across LLMs)
    beta  — shape/convexity parameter (shared across LLMs)

The RI channel model and NIAS test are identical to fit_ri_model.py.
Only the q-mapping is replaced.

Input  : Dataset/results/audio_<noise>/<model>/summary_<model>.csv
           (produced by compute_metrics_audio.py; noise_p column = SNR in dB)
Output : Dataset/results/audio_<noise>/ri_fit/
           fitted_params.csv, shared_noise_params.csv, nias_test.csv, *.png

Usage
-----
    python fit_ri_model_audio.py --results-dir ./Dataset/results/audio_babble
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binomtest

# ==============================================================================
# CONFIG
# ==============================================================================
RESULTS_DIR = "./Dataset/results/audio_babble"
OUTPUT_DIR  = "./Dataset/results/audio_babble/ri_fit"

MODELS = [
    ("GPT-3.5-turbo",         "gpt_gpt_3_5_turbo"),
    ("GPT-5.4-nano",          "gpt_gpt_5_4_nano"),
    ("Gemini-2.5-Flash",      "gemini_gemini_2_5_flash"),
    ("Gemini-2.5-Flash-Lite", "gemini_gemini_2_5_flash_lite"),
]

# Optimisation bounds  (same structure as text-noise fit_ri_model.py)
LB_X     = 1e-6;  UB_X     = 100.0
LB_ALPHA = 0.01;  UB_ALPHA = 50.0   # scale α
LB_BETA  = 0.1;   UB_BETA  = 3.0    # shape β
# ==============================================================================

_EPS = 1e-10


# ------------------------------------------------------------------------------
# RI channel model  (same as fit_ri_model.py)
# ------------------------------------------------------------------------------

def _pa(x, q):
    ex  = np.exp(np.clip(x, -500, 500))
    num = (1 + q) * ex - (1 - q)
    den = 2 * ex - 2
    pa  = np.where(np.abs(den) < 1e-12, 0.5, num / den)
    return np.clip(pa, 0.5, 1.0 - _EPS)


def ri_p1a(x, q):
    pa = _pa(x, q)
    ex = np.exp(np.clip(x, -500, 500))
    return np.clip((pa * ex) / (pa * ex + (1 - pa)), 0.0, 1.0)


def ri_p2b(x, q):
    pa = _pa(x, q)
    ex = np.exp(np.clip(x, -500, 500))
    return np.clip(((1 - pa) * ex) / (pa + (1 - pa) * ex), 0.0, 1.0)


def ri_pc(x, q):
    return 0.5 * (1 - q) * ri_p1a(x, q) + 0.5 * (1 - q) * ri_p2b(x, q) + 0.5 * q


# ------------------------------------------------------------------------------
# Audio q-mapping
# ------------------------------------------------------------------------------

def q_of_nsr(nsr, alpha, beta):
    """q(NSR) = min( alpha * NSR^beta, 1 )
    Same functional form as text-noise: q(p') = min(alpha*p'^beta, 1).
    NSR = N/S power ratio: 0=clean, large=noisy.
    NSR=0 → q=0 naturally.
    """
    return np.clip(alpha * np.power(np.maximum(nsr, 0.0), beta), 0.0, 1.0)


def pc_model_snr(x, nsr, alpha, beta):
    return ri_pc(x, q_of_nsr(nsr, alpha, beta))


# ------------------------------------------------------------------------------
# NIAS test
# ------------------------------------------------------------------------------

def nias_test(summary_df, n_samples=100, alpha=0.05):
    """
    NIAS condition: P(A=a|Y=1) >= (P(A=a|Y=2) + 2) / 3

    A violation is declared only when BOTH hold:
      1. P(A=a|Y=1) < constraint  (wrong direction)
      2. One-sided binomial test p-value < alpha  (statistically significant)
    Otherwise: "no significant violation".
    """
    n_per_class = n_samples // 2
    rows = []
    for _, row in summary_df.iterrows():
        p_a_given_y1 = row["P1a"]
        p_a_given_y2 = row["P2a"]
        if np.isnan(p_a_given_y1) or np.isnan(p_a_given_y2):
            continue
        constraint = (p_a_given_y2 + 2.0) / 3.0
        k = int(round(p_a_given_y1 * n_per_class))
        try:
            # one-sided: H0 p >= constraint, H1 p < constraint
            pval = binomtest(k, n_per_class, constraint, alternative="less").pvalue
        except Exception:
            pval = float("nan")
        # violation only when significantly below constraint
        violated = (p_a_given_y1 < constraint) and (not np.isnan(pval)) and (pval < alpha)
        rows.append({
            "NSR":         row["noise_p"],
            "P(A=a|Y=1)":  round(p_a_given_y1, 4),
            "P(A=a|Y=2)":  round(p_a_given_y2, 4),
            "Constraint":  round(constraint, 4),
            "NIAS_holds":  not violated,
            "p_value":     round(pval, 6) if not np.isnan(pval) else float("nan"),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def load_summary(dir_name, label, results_dir):
    pattern = os.path.join(results_dir, dir_name, f"summary_{dir_name}.csv")
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No summary CSV for '{label}' at {pattern}\n"
            "Run compute_metrics_audio.py first."
        )
    df = pd.read_csv(files[0])
    df = df.dropna(subset=["noise_p", "Pc", "P1a", "P2a"]).copy()
    df["noise_p"] = pd.to_numeric(df["noise_p"], errors="coerce")
    df = df.dropna(subset=["noise_p"])
    # noise_p stores SNR ratio; add snr_db column for fitting
    df["snr_db"] = 10 * np.log10(df["noise_p"].clip(lower=1e-10))
    df = df.sort_values("noise_p", ascending=True).reset_index(drop=True)
    return df


# ------------------------------------------------------------------------------
# Joint fitting
# ------------------------------------------------------------------------------

def _joint_sse(params, datasets):
    """params = [x_0, ..., x_N, eta, zeta]"""
    n     = len(datasets)
    xs    = params[:n]
    alpha = params[n]
    beta  = params[n + 1]
    total = 0.0
    for i, (nsr_obs, Pc_obs) in enumerate(datasets):
        Pc_pred = pc_model_snr(xs[i], nsr_obs, alpha, beta)
        total  += np.sum((Pc_pred - Pc_obs) ** 2)
    return total


def fit_joint(models_data):
    n        = len(models_data)
    datasets = [(df["noise_p"].values, df["Pc"].values) for _, df in models_data]

    lb  = np.array([LB_X] * n + [LB_ALPHA, LB_BETA])
    ub  = np.array([UB_X] * n + [UB_ALPHA, UB_BETA])
    mid = (lb + ub) / 2.0

    x0_candidates = [
        np.clip(lb + 1e-3, lb + 1e-9, ub - 1e-9),
        np.clip(ub - 1e-3, lb + 1e-9, ub - 1e-9),
        mid,
    ]

    best_x, best_fval = None, np.inf
    for k, x0 in enumerate(x0_candidates):
        res = minimize(
            _joint_sse, x0, args=(datasets,),
            method="L-BFGS-B", bounds=list(zip(lb, ub)),
            options={"maxiter": 10000, "ftol": 1e-14, "gtol": 1e-10},
        )
        print(f"  Start {k+1}: SSE={res.fun:.6g}  success={res.success}")
        if res.fun < best_fval:
            best_x, best_fval = res.x, res.fun

    alpha = best_x[n]
    beta  = best_x[n + 1]

    results = {"alpha": alpha, "beta": beta, "joint_SSE": best_fval, "models": {}}

    for i, (label, df) in enumerate(models_data):
        x       = best_x[i]
        nsr_obs = df["noise_p"].values
        Pc_obs  = df["Pc"].values
        Pc_pred = pc_model_snr(x, nsr_obs, alpha, beta)
        sse     = np.sum((Pc_pred - Pc_obs) ** 2)
        sst     = np.sum((Pc_obs - Pc_obs.mean()) ** 2)
        r2      = 1.0 - sse / max(sst, 1e-15)
        results["models"][label] = {
            "x":       x,
            "lambda":  1.0 / x,
            "SSE":     sse,
            "MSSE":    sse / len(nsr_obs),
            "R2":      r2,
            "df":      df,
            "Pc_pred": Pc_pred,
        }

    return results


# ------------------------------------------------------------------------------
# Save & plot
# ------------------------------------------------------------------------------

def save_params(results, output_dir):
    alpha = results["alpha"]
    beta  = results["beta"]

    rows = []
    for label, m in results["models"].items():
        rows.append({
            "model":           label,
            "x_r_over_lambda": round(m["x"],      6),
            "lambda":          round(m["lambda"],  6),
            "alpha":           round(alpha, 6),
            "beta":            round(beta,  6),
            "SSE_Pc":          round(m["SSE"],      8),
            "MSSE_Pc":         round(m["MSSE"],     8),
            "R2_Pc":           round(m["R2"],       6),
        })
    df_model = pd.DataFrame(rows)
    path = os.path.join(output_dir, "fitted_params.csv")
    df_model.to_csv(path, index=False)
    print(f"\n  Fitted params  -> {path}")
    print(df_model.to_string(index=False))

    key_snr_db = ["∞", 40, 30, 20, 15, 10, 5, 4, 3, 2, 1, 0, -1, -2, -3, -5, -6, -7, -8, -10, -15, -20]
    key_nsr    = [0.0] + [round(10 ** (-s / 10), 6) for s in key_snr_db[1:]]
    shared_rows = [{
        "alpha": round(alpha, 6),
        "beta":  round(beta,  6),
        **{f"q(NSR={r})": round(float(q_of_nsr(r, alpha, beta)), 6) for r in key_nsr},
    }]
    path2 = os.path.join(output_dir, "shared_noise_params.csv")
    pd.DataFrame(shared_rows).to_csv(path2, index=False)
    print(f"  Shared params  -> {path2}")

    print(f"\n  q(NSR) = min(alpha * NSR^beta, 1)")
    print(f"  alpha={alpha:.6f},  beta={beta:.6f}")
    print(f"  {'SNR(dB)':>8}  {'NSR':>8}  {'q':>8}")
    for snr_db, nsr in zip(key_snr_db, key_nsr):
        print(f"  {str(snr_db):>8}  {nsr:>8.4f}  {float(q_of_nsr(nsr, alpha, beta)):>8.4f}")


def save_nias_csv(nias_results, output_dir):
    rows = []
    for label, ndf in nias_results.items():
        ndf = ndf.copy(); ndf.insert(0, "model", label); rows.append(ndf)
    out  = pd.concat(rows, ignore_index=True)
    path = os.path.join(output_dir, "nias_test.csv")
    out.to_csv(path, index=False)
    print(f"\n  NIAS test      -> {path}")
    return out


MODEL_COLORS = {
    "GPT-3.5-turbo":         "black",
    "GPT-5.4-nano":          "red",
    "Gemini-2.5-Flash":      "goldenrod",
    "Gemini-2.5-Flash-Lite": "green",
}


def plot_fits(results, output_dir, noise_type="audio"):
    alpha    = results["alpha"]
    beta     = results["beta"]
    nsr_fine = np.concatenate([[0.0], np.logspace(-3, 2, 400)])

    for label, m in results["models"].items():
        x      = m["x"]
        df     = m["df"]
        Pc_fit = pc_model_snr(x, nsr_fine, alpha, beta)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["noise_p"], df["Pc"], color="red", zorder=3,
                   label=r"$\hat{P}^{(Y)}_{correct}$ (Observed)")
        ax.plot(nsr_fine, Pc_fit, color="blue", linewidth=2,
                label=f"$P^{{(Y)}}_{{correct}}$ (Fitted, x={x:.3f}, λ={m['lambda']:.3f})")
        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_xlabel("NSR")
        ax.set_ylabel(r"$P^{(Y)}_{correct}$")
        ax.set_title(f"{label}   α={alpha:.4f}, β={beta:.4f}")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8); ax.grid(True, which="both", ls="--", alpha=0.4)
        fig.tight_layout()
        fname = os.path.join(output_dir, f"{label.replace('-','').replace(' ','_')}_fit.png")
        fig.savefig(fname, dpi=150); plt.close(fig)
        print(f"  Plot -> {fname}")

    # all models — empirical + fitted
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, m in results["models"].items():
        c = MODEL_COLORS.get(label, "gray")
        ax.scatter(m["df"]["noise_p"], m["df"]["Pc"], color=c, zorder=3, s=40)
        ax.plot(nsr_fine, pc_model_snr(m["x"], nsr_fine, alpha, beta),
                color=c, linewidth=2, label=label)
    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_xlabel("NSR")
    ax.set_ylabel(r"$P^{(Y)}_{correct}$")
    ax.set_title(f"RI fits — {noise_type}   α={alpha:.4f}, β={beta:.4f}")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fname = os.path.join(output_dir, "all_models_fit.png")
    fig.savefig(fname, dpi=150); plt.close(fig)
    print(f"  Plot -> {fname}")

    # regression lines only (no scatter)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, m in results["models"].items():
        c = MODEL_COLORS.get(label, "gray")
        ax.plot(nsr_fine, pc_model_snr(m["x"], nsr_fine, alpha, beta),
                color=c, linewidth=2, label=label)
    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_xlabel("NSR")
    ax.set_ylabel(r"$P^{(Y)}_{correct}$")
    ax.set_title(f"RI fits — {noise_type}   α={alpha:.4f}, β={beta:.4f}")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fname = os.path.join(output_dir, "all_models_fit_regression_only.png")
    fig.savefig(fname, dpi=150); plt.close(fig)
    print(f"  Plot -> {fname}")

    # q vs NSR
    q_fine = q_of_nsr(nsr_fine, alpha, beta)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(nsr_fine, q_fine, color="darkorange", linewidth=2)
    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_xlabel("NSR")
    ax.set_ylabel("hate speech hiding rate")
    ax.set_title(f"Noise mapping  q(NSR) = min(α·NSR^β, 1)\nα={alpha:.4f}, β={beta:.4f}")
    ax.set_ylim(0, 1.05); ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fname = os.path.join(output_dir, "q_vs_NSR.png")
    fig.savefig(fname, dpi=150); plt.close(fig)
    print(f"  Plot -> {fname}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    global OUTPUT_DIR, RESULTS_DIR
    parser = argparse.ArgumentParser(description="Fit audio RI model (SNR-based q)")
    parser.add_argument("--results-dir", default=None,
                        help=f"Results dir (default: {RESULTS_DIR})")
    args = parser.parse_args()

    results_dir = args.results_dir or RESULTS_DIR
    output_dir  = os.path.join(results_dir, "ri_fit")
    os.makedirs(output_dir, exist_ok=True)

    models_data  = []
    nias_results = {}

    for label, dir_name in MODELS:
        try:
            df = load_summary(dir_name, label, results_dir)
            models_data.append((label, df))
            nias_results[label] = nias_test(df)
            print(f"Loaded {label}: {len(df)} SNR levels")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")

    if not models_data:
        raise RuntimeError("No model data found. Run compute_metrics_audio.py first.")

    print("\n=== NIAS Test ===")
    nias_all = save_nias_csv(nias_results, output_dir)
    for label in nias_results:
        sub = nias_all[nias_all["model"] == label]
        print(f"  {label}: {sub['NIAS_holds'].sum()}/{len(sub)} SNR levels satisfy NIAS")

    print("\n=== Fitting RI model (audio) ===")
    print("Params: x per model (shared α, β)\n")
    results = fit_joint(models_data)

    print(f"\n{'='*55}")
    print(f"  α (alpha) = {results['alpha']:.6f}")
    print(f"  β (beta)  = {results['beta']:.6f}")
    print(f"  Joint SSE = {results['joint_SSE']:.8f}")
    for label, m in results["models"].items():
        print(f"  {label:25s}  x={m['x']:.4f}  λ={m['lambda']:.4f}  R²={m['R2']:.4f}")
    print(f"{'='*55}")

    noise_type = os.path.basename(results_dir).replace("audio_", "")
    save_params(results, output_dir)
    plot_fits(results, output_dir, noise_type=noise_type)


if __name__ == "__main__":
    main()
