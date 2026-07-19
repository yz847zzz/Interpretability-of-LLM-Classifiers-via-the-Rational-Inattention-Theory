"""
Fit the extended Rational Inattention (RI) model to observed LLM classification
metrics, and run the NIAS (No Improving Action Switches) test.

Reference
---------
Y. Zhao, A. Abdi, "Interpretability of LLM Classifiers via the Rational
Inattention Theory with Application to Hate Speech Detection,"
ACL Student Research Workshop, 2026.

Model (Eqs. 4-6 in the paper)
------------------------------
Noise-to-channel mapping  (Eq. 11):
    q(p'; alpha, beta) = min(alpha * p'^beta, 1)

Optimal attention probability  (Eq. 5):
    Pa = clip( ((1+q)*exp(x) - (1-q)) / (2*(exp(x) - 1)),  0.5, 1 )

Sensitivity / Specificity  (Eq. 4):
    P1a = Pa*exp(x)  / (Pa*exp(x) + (1-Pa))
    P2b = (1-Pa)*exp(x) / (Pa + (1-Pa)*exp(x))

Accuracy  (Eq. 6):
    Pc(q, x) = 0.5*(1-q)*P1a + 0.5*(1-q)*P2b + 0.5*q

Parameters estimated  (Eq. 12):
    x_i = (r/lambda)_i  — reward-to-cost ratio, one per LLM
    alpha, beta          — shared noise-mapping parameters

lambda is recovered afterwards by setting r = 1:
    lambda_i = r / x_i = 1 / x_i

NIAS test (Eq. 9):
    P(A=a | Y=1) >= (P(A=a | Y=2) + 2) / 3

Input   : Dataset/results/<model>/summary_<model>.csv
Output  : Dataset/results/ri_fit/fitted_params.csv   — x, lambda, alpha, beta
          Dataset/results/ri_fit/<model>_fit.png     — observed vs fitted Pc
          Dataset/results/ri_fit/nias_test.csv       — NIAS test per noise level
"""

import glob
import os
from scipy.stats import binomtest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ==============================================================================
# CONFIG
# ==============================================================================
RESULTS_DIR = "./Dataset/results"
OUTPUT_DIR  = "./Dataset/results/ri_fit"

# Models to fit jointly — label must match directory name under RESULTS_DIR
# Format: (display_label, directory_name)
MODELS = [
    ("GPT-3.5-turbo",        "gpt_gpt_3_5_turbo"),
    ("GPT-5.4-nano",         "gpt_gpt_5_4_nano"),
    ("Gemini-2.5-Flash",     "gemini_gemini_2_5_flash"),
    ("Gemini-2.5-Flash-Lite","gemini_gemini_2_5_flash_lite"),
]

# Bounds for optimisation:  x = r/lambda in (0, 100],  alpha in (0.01, 10],  beta in (1, 4]
LB_X     = 1e-6
UB_X     = 100.0
LB_ALPHA = 0.01
UB_ALPHA = 10.0
LB_BETA  = 1.0
UB_BETA  = 4.0
# ==============================================================================

_EPS = 1e-10   # numerical floor


# ------------------------------------------------------------------------------
# RI model  — all functions take x = r/lambda directly
# ------------------------------------------------------------------------------

def _pa(x, q):
    """Optimal attention Pa*(x, q), clamped to [0.5, 1-eps]  (Eq. 5)."""
    ex = np.exp(np.clip(x, -500, 500))
    num = (1 + q) * ex - (1 - q)
    den = 2 * ex - 2
    pa = np.where(np.abs(den) < 1e-12, 0.5, num / den)
    return np.clip(pa, 0.5, 1.0 - _EPS)


def ri_p1a(x, q):
    """P(A=a | V=1) — sensitivity for 'not hate' class  (Eq. 4)."""
    pa = _pa(x, q)
    ex = np.exp(np.clip(x, -500, 500))
    return np.clip((pa * ex) / (pa * ex + (1 - pa)), 0.0, 1.0)


def ri_p2b(x, q):
    """P(A=b | V=2) — sensitivity for 'hate' class  (Eq. 4)."""
    pa  = _pa(x, q)
    pb  = 1.0 - pa
    ex  = np.exp(np.clip(x, -500, 500))
    return np.clip((pb * ex) / (pa + pb * ex), 0.0, 1.0)


def ri_pc(x, q):
    """Probability of correct action  (Eq. 6)."""
    return 0.5 * (1 - q) * ri_p1a(x, q) + 0.5 * (1 - q) * ri_p2b(x, q) + 0.5 * q


def q_of_p(p, alpha, beta):
    """Noise-to-channel mapping  (Eq. 11)."""
    return np.clip(alpha * np.power(np.maximum(p, 0.0), beta), 0.0, 1.0)


def pc_model(x, p, alpha, beta):
    return ri_pc(x, q_of_p(p, alpha, beta))


# ------------------------------------------------------------------------------
# NIAS test  (Eq. 9)
# ------------------------------------------------------------------------------

def nias_test(summary_df, n_samples=100, alpha=0.05):
    """
    NIAS condition (Eq. 9):
        P(A=a | Y=1) >= (P(A=a | Y=2) + 2) / 3

    A violation is declared only when BOTH hold:
      1. P(A=a|Y=1) < constraint  (wrong direction)
      2. One-sided binomial p-value < alpha  (statistically significant)
    Otherwise: no significant violation.
    """
    rows = []
    n_per_class = n_samples // 2

    for _, row in summary_df.iterrows():
        p_a_given_y1 = row["P1a"]
        p_a_given_y2 = row["P2a"]
        constraint   = (p_a_given_y2 + 2.0) / 3.0
        k = int(round(p_a_given_y1 * n_per_class))
        try:
            # one-sided: H1 = P(A=a|Y=1) < constraint
            pval = binomtest(k, n_per_class, constraint, alternative="less").pvalue
        except Exception:
            pval = float("nan")
        violated = (p_a_given_y1 < constraint) and (not np.isnan(pval)) and (pval < alpha)
        rows.append({
            "noise_p":    row["noise_p"],
            "P(A=a|Y=1)": round(p_a_given_y1, 4),
            "P(A=a|Y=2)": round(p_a_given_y2, 4),
            "Constraint": round(constraint, 4),
            "NIAS_holds": not violated,
            "p_value":    round(pval, 6) if not np.isnan(pval) else float("nan"),
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def load_summary(dir_name, label, results_dir=None, snr_mode=False):
    base    = results_dir or RESULTS_DIR
    pattern = os.path.join(base, dir_name, f"summary_{dir_name}.csv")
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No summary CSV for '{label}' at {pattern}\n"
            "Run compute_metrics_audio.py first."
        )
    df = pd.read_csv(files[0])
    if snr_mode:
        # SNR axis: higher = cleaner. Normalise to [0,1]: SNR_max→0, SNR_min→1
        snr_max = df["noise_p"].max()
        snr_min = df["noise_p"].min()
        df["noise_p"] = (snr_max - df["noise_p"]) / (snr_max - snr_min)
    df = df.sort_values("noise_p").reset_index(drop=True)
    return df


# ------------------------------------------------------------------------------
# Joint fitting
# ------------------------------------------------------------------------------

def _joint_sse(params, datasets):
    """
    params = [x_0, x_1, ..., x_N, alpha, beta]
    datasets = list of (p_obs, Pc_obs)
    """
    n      = len(datasets)
    xs     = params[:n]
    alpha  = params[n]
    beta   = params[n + 1]
    total  = 0.0
    for i, (p_obs, Pc_obs) in enumerate(datasets):
        Pc_pred = pc_model(xs[i], p_obs, alpha, beta)
        total  += np.sum((Pc_pred - Pc_obs) ** 2)
    return total


def fit_joint(models_data):
    """
    Estimate {x_i, alpha, beta} jointly by minimising SSE over all LLMs' Pc.

    models_data: list of (label, summary_df)
    Returns dict of results.
    """
    n        = len(models_data)
    datasets = [(df["noise_p"].values, df["Pc"].values) for _, df in models_data]

    lb  = np.array([LB_X] * n    + [LB_ALPHA, LB_BETA])
    ub  = np.array([UB_X] * n    + [UB_ALPHA, UB_BETA])
    mid = (lb + ub) / 2.0

    x0_candidates = [
        np.clip(lb * (1 + 1e-3), lb + 1e-9, ub - 1e-9),
        np.clip(ub * (1 - 1e-3), lb + 1e-9, ub - 1e-9),
        mid,
    ]

    bounds     = list(zip(lb, ub))
    best_x     = None
    best_fval  = np.inf

    for k, x0 in enumerate(x0_candidates):
        res = minimize(
            _joint_sse, x0, args=(datasets,),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 10000, "ftol": 1e-14, "gtol": 1e-10},
        )
        print(f"  Start {k+1}: SSE={res.fun:.6g}  success={res.success}")
        if res.fun < best_fval:
            best_x, best_fval = res.x, res.fun

    alpha = best_x[n]
    beta  = best_x[n + 1]

    results = {
        "alpha":      alpha,
        "beta":       beta,
        "joint_SSE":  best_fval,
        "models":     {},
    }

    for i, (label, df) in enumerate(models_data):
        x        = best_x[i]
        lam      = 1.0 / x          # lambda = r / x,  r = 1 assumed
        p_obs    = df["noise_p"].values
        Pc_obs   = df["Pc"].values

        Pc_pred  = pc_model(x, p_obs, alpha, beta)
        sse      = np.sum((Pc_pred - Pc_obs) ** 2)
        sst      = np.sum((Pc_obs - Pc_obs.mean()) ** 2)
        r2       = 1.0 - sse / max(sst, 1e-15)

        results["models"][label] = {
            "x":       x,           # estimated r/lambda
            "lambda":  lam,         # = 1/x  (with r=1)
            "SSE":     sse,
            "MSSE":    sse / len(p_obs),
            "R2":      r2,
            "df":      df,
            "Pc_pred": Pc_pred,
        }

    return results


# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------

def plot_fits(results):
    p_fine = np.linspace(0, 1, 400)
    alpha  = results["alpha"]
    beta   = results["beta"]

    for label, m in results["models"].items():
        x        = m["x"]
        df       = m["df"]
        Pc_fit   = pc_model(x, p_fine, alpha, beta)

        fig, ax  = plt.subplots(figsize=(6, 4))
        ax.scatter(df["noise_p"], df["Pc"], color="red",  zorder=3,
                   label=f"Observed Pc ({label})")
        ax.plot(p_fine, Pc_fit,             color="blue", linewidth=2,
                label=f"RI fit  (x={x:.3f}, lam={m['lambda']:.3f})")
        ax.set_xlabel("Noise level p'")
        ax.set_ylabel("Pc (correct decision probability)")
        ax.set_title(f"{label}   alpha={alpha:.3f}, beta={beta:.3f}")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True)
        fig.tight_layout()

        fname = os.path.join(OUTPUT_DIR, f"{label.replace('-','').replace(' ','_')}_fit.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Plot -> {fname}")

    # All models overlaid
    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, m) in enumerate(results["models"].items()):
        c       = colors[i % len(colors)]
        Pc_fit  = pc_model(m["x"], p_fine, alpha, beta)
        ax.scatter(m["df"]["noise_p"], m["df"]["Pc"], color=c, zorder=3, s=40)
        ax.plot(p_fine, Pc_fit, color=c, linewidth=2, label=label)
    ax.set_xlabel("Noise level p'")
    ax.set_ylabel("Pc")
    ax.set_title(f"Extended RI model fits  (alpha={alpha:.3f}, beta={beta:.3f})")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "all_models_fit.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Plot -> {fname}")


# ------------------------------------------------------------------------------
# Save CSVs
# ------------------------------------------------------------------------------

def save_params(results):
    # Per-model parameters
    rows = []
    for label, m in results["models"].items():
        rows.append({
            "model":           label,
            "x_r_over_lambda": round(m["x"],           6),
            "lambda":          round(m["lambda"],       6),
            "alpha":           round(results["alpha"],  6),
            "beta":            round(results["beta"],   6),
            "SSE_Pc":          round(m["SSE"],          8),
            "MSSE_Pc":         round(m["MSSE"],         8),
            "R2_Pc":           round(m["R2"],           6),
        })
    df_model = pd.DataFrame(rows)
    path_model = os.path.join(OUTPUT_DIR, "fitted_params.csv")
    df_model.to_csv(path_model, index=False)
    print(f"\n  Fitted params (per model) -> {path_model}")
    print(df_model.to_string(index=False))

    # Shared noise-mapping parameters with q values at key noise levels
    alpha = results["alpha"]
    beta  = results["beta"]
    key_p = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    shared_rows = [{
        "alpha":   round(alpha, 6),
        "beta":    round(beta,  6),
        **{f"q(p={p:.1f})": round(float(q_of_p(p, alpha, beta)), 6) for p in key_p},
    }]
    df_shared = pd.DataFrame(shared_rows)
    path_shared = os.path.join(OUTPUT_DIR, "shared_noise_params.csv")
    df_shared.to_csv(path_shared, index=False)
    print(f"\n  Shared noise params       -> {path_shared}")

    print(f"\n  Noise mapping  q(p') = min(alpha * p'^beta, 1)")
    print(f"  alpha = {alpha:.6f},  beta = {beta:.6f}")
    print(f"  {'p':>6}  {'q(p)':>8}")
    print(f"  {'------':>6}  {'--------':>8}")
    for p in key_p:
        q = float(q_of_p(p, alpha, beta))
        print(f"  {p:>6.1f}  {q:>8.4f}")


def save_nias(nias_results):
    rows = []
    for label, ndf in nias_results.items():
        ndf = ndf.copy()
        ndf.insert(0, "model", label)
        rows.append(ndf)
    out = pd.concat(rows, ignore_index=True)
    path = os.path.join(OUTPUT_DIR, "nias_test.csv")
    out.to_csv(path, index=False)
    print(f"\n  NIAS test     -> {path}")
    return out


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    global OUTPUT_DIR, RESULTS_DIR
    import argparse
    parser = argparse.ArgumentParser(description="Fit RI model and run NIAS test")
    parser.add_argument("--results-dir", default=None,
                        help=f"Results base dir (default: {RESULTS_DIR})")
    parser.add_argument("--snr", action="store_true",
                        help="Normalise SNR-axis noise_p to [0,1] before fitting")
    args = parser.parse_args()
    results_dir = args.results_dir or RESULTS_DIR
    output_dir  = os.path.join(results_dir, "ri_fit")
    os.makedirs(output_dir, exist_ok=True)
    OUTPUT_DIR  = output_dir
    RESULTS_DIR = results_dir

    models_data  = []
    nias_results = {}

    for label, dir_name in MODELS:
        try:
            df = load_summary(dir_name, label, results_dir, snr_mode=args.snr)
            models_data.append((label, df))
            nias_results[label] = nias_test(df)
            print(f"Loaded {label}: {len(df)} noise levels")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")

    if not models_data:
        raise RuntimeError("No model data found. Run compute_metrics.py first.")

    # --- NIAS test ---
    print("\n=== NIAS Test (Eq. 9) ===")
    nias_all = save_nias(nias_results)
    for label in nias_results:
        sub = nias_all[nias_all["model"] == label]
        n_pass = sub["NIAS_holds"].sum()
        print(f"  {label}: {n_pass}/{len(sub)} environments satisfy NIAS")
    print(nias_all[["model","noise_p","P(A=a|Y=1)","Constraint","NIAS_holds","p_value"]].to_string(index=False))

    # --- Fit RI model ---
    print(f"\n=== Fitting RI model ===")
    print(f"Estimating: x = r/lambda per model, shared alpha and beta")
    print(f"(lambda recovered as 1/x after fitting, assuming r=1)\n")

    results = fit_joint(models_data)

    print(f"\n{'='*55}")
    print(f"  alpha  (shared) = {results['alpha']:.6f}")
    print(f"  beta   (shared) = {results['beta']:.6f}")
    print(f"  Joint SSE       = {results['joint_SSE']:.8f}")
    for label, m in results["models"].items():
        print(f"  {label:12s}  x={m['x']:.4f}  lambda={m['lambda']:.4f}  "
              f"MSSE={m['MSSE']:.6f}  R2={m['R2']:.4f}")
    print(f"{'='*55}")

    save_params(results)
    plot_fits(results)


if __name__ == "__main__":
    main()
