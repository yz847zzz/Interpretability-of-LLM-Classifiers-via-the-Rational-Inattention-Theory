"""
Fit the Rational Inattention (RI) model to observed LLM classification metrics.

Reads the summary CSVs produced by compute_metrics.py and fits the RI model
parameters to the observed Pc (accuracy) curve across noise levels p = 0..1.

RI model
--------
Noise-to-channel mapping:
    q(p; alpha, beta) = clip(alpha * p^beta, 0, 1)

Optimal attention allocation Pa*(r, lambda, q):
    Pa = ((1+q)*exp(r/lambda) - (1-q)) / (2*exp(r/lambda) - 2)
    Pa = clip(Pa, 0.5, 1)

Sensitivity  P1a(r, lambda, q) = 1 / (1 + ((1-Pa)/Pa) * exp(-r/lambda))
Specificity  P1b(r, lambda, q) = 1 / (1 + (Pa/(1-Pa)) * exp(-r/lambda))
Accuracy     Pc(r, lambda, q)  = 0.5*(1-q)*P1a + 0.5*(1-q)*P1b + 0.5*q

Parameters estimated
--------------------
    r      : per-model information cost threshold
    alpha  : shared noise-mapping scale  (q-p curve)
    beta   : shared noise-mapping shape  (q-p curve)
    lambda : fixed to 1 (set LAMBDA below to change)

Estimation: joint MSSE minimisation over all models with multiple starts.

Input   : Dataset/results/<model>/summary_<model>.csv
Output  : Dataset/results/ri_fit/fitted_params.csv
          Dataset/results/ri_fit/<model>_fit.png
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ==============================================================================
# CONFIG
# ==============================================================================
RESULTS_DIR = "./Dataset/results"
OUTPUT_DIR  = "./Dataset/results/ri_fit"

# Models to fit jointly (must match directory names under RESULTS_DIR)
# Format: (label, directory_name, pred_col_prefix)
MODELS = [
    ("GPT-3.5",  "gpt_gpt_3_5_turbo",      "gpt_gpt_3_5_turbo"),
    ("Gemini",   "gemini_gemini_2_5_flash",  "gemini_gemini_2_5_flash"),
]

LAMBDA = 1.0   # fixed; set to None to treat as a free parameter per model

# Optimisation bounds:  r in (0, 100],  alpha in (0.01, 10],  beta in (1, 4]
LB = np.array([1e-6, 0.01, 1.0])
UB = np.array([1e2,  10.0, 4.0])
# ==============================================================================


# ------------------------------------------------------------------------------
# RI model functions (vectorised, numerically stable)
# ------------------------------------------------------------------------------

def _pa_star(r, lam, q):
    """Optimal attention probability Pa*(r, lambda, q), clamped to [0.5, 1]."""
    e = np.exp(np.clip(r / lam, -500, 500))
    num = (1 + q) * e - (1 - q)
    den = 2 * e - 2
    # den -> 0 when r -> 0; Pa -> 0.5 in that limit
    pa = np.where(np.abs(den) < 1e-12, 0.5, num / den)
    return np.clip(pa, 0.5, 1.0 - np.finfo(float).tiny)


_EPS = 1e-10   # safe floor for pa away from 0/1


def ri_p1a(r, lam, q):
    """Sensitivity P(pred=1 | label=1)."""
    pa = np.clip(_pa_star(r, lam, q), _EPS, 1 - _EPS)
    e  = np.exp(np.clip(-r / lam, -500, 500))
    return np.clip(1.0 / (1.0 + ((1 - pa) / pa) * e), 0.0, 1.0)


def ri_p1b(r, lam, q):
    """Specificity P(pred=0 | label=0)."""
    pa = np.clip(_pa_star(r, lam, q), _EPS, 1 - _EPS)
    e  = np.exp(np.clip(-r / lam, -500, 500))
    return np.clip(1.0 / (1.0 + (pa / (1 - pa)) * e), 0.0, 1.0)


def ri_pc(r, lam, q):
    """Accuracy Pc = 0.5*(1-q)*P1a + 0.5*(1-q)*P1b + 0.5*q."""
    p1a = ri_p1a(r, lam, q)
    p1b = ri_p1b(r, lam, q)
    return 0.5 * (1 - q) * p1a + 0.5 * (1 - q) * p1b + 0.5 * q


def q_of_p(p, alpha, beta):
    """Noise-to-channel mapping q = alpha * p^beta, clipped to [0, 1]."""
    return np.clip(alpha * np.power(np.maximum(p, 0.0), beta), 0.0, 1.0)


def pc_with_p(r, lam, p, alpha, beta):
    return ri_pc(r, lam, q_of_p(p, alpha, beta))


def p1a_with_p(r, lam, p, alpha, beta):
    return ri_p1a(r, lam, q_of_p(p, alpha, beta))


def p1b_with_p(r, lam, p, alpha, beta):
    return ri_p1b(r, lam, q_of_p(p, alpha, beta))


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def load_summary(model_dir_name, model_label):
    pattern = os.path.join(RESULTS_DIR, model_dir_name, f"summary_{model_dir_name}.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No summary CSV found for '{model_label}' at {pattern}\n"
            f"Run compute_metrics.py first."
        )
    df = pd.read_csv(files[0]).sort_values("noise_p").reset_index(drop=True)
    return df["noise_p"].values, df["Pc"].values, df["P1a"].values, df["P1b"].values


# ------------------------------------------------------------------------------
# Fitting
# ------------------------------------------------------------------------------

def _joint_msse(x, lam, datasets):
    """
    x = [r_0, r_1, ..., r_N, alpha, beta]
    datasets = list of (p_array, Pc_array)
    """
    n = len(datasets)
    r_vals  = x[:n]
    alpha   = x[n]
    beta    = x[n + 1]
    total   = 0.0
    for i, (p_obs, Pc_obs) in enumerate(datasets):
        Pc_pred = pc_with_p(r_vals[i], lam, p_obs, alpha, beta)
        total  += np.mean((Pc_pred - Pc_obs) ** 2)
    return total


def fit_joint(models_data, lam):
    """
    Fit shared (alpha, beta) and per-model r jointly to Pc data.
    models_data: list of (label, p_obs, Pc_obs)
    Returns: dict with r per model, alpha, beta, MSSE, R^2 per model.
    """
    n = len(models_data)
    datasets = [(p, Pc) for _, p, Pc, _, _ in models_data]

    # parameter vector: [r_0, r_1, ..., alpha, beta]
    lb = np.concatenate([[LB[0]] * n, [LB[1], LB[2]]])
    ub = np.concatenate([[UB[0]] * n, [UB[1], UB[2]]])
    mid = (lb + ub) / 2.0

    x0_candidates = [
        lb * (1 + 1e-3),
        ub * (1 - 1e-3),
        mid,
    ]
    # clip inside bounds
    x0_candidates = [np.clip(x, lb + 1e-9, ub - 1e-9) for x in x0_candidates]

    bounds = list(zip(lb, ub))
    obj    = lambda x: _joint_msse(x, lam, datasets)

    best_x, best_fval = None, np.inf
    for k, x0 in enumerate(x0_candidates):
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 10000, "ftol": 1e-14, "gtol": 1e-10})
        print(f"  Start {k+1}: MSSE={res.fun:.6g}  success={res.success}")
        if res.fun < best_fval:
            best_x, best_fval = res.x, res.fun

    r_vals  = best_x[:n]
    alpha   = best_x[n]
    beta    = best_x[n + 1]

    results = {"alpha": alpha, "beta": beta, "lambda": lam, "joint_MSSE": best_fval, "models": {}}
    for i, (label, p_obs, Pc_obs, P1a_obs, P1b_obs) in enumerate(models_data):
        r = r_vals[i]
        Pc_pred  = pc_with_p(r, lam, p_obs, alpha, beta)
        P1a_pred = p1a_with_p(r, lam, p_obs, alpha, beta)
        P1b_pred = p1b_with_p(r, lam, p_obs, alpha, beta)

        sse = np.sum((Pc_pred - Pc_obs) ** 2)
        sst = np.sum((Pc_obs  - Pc_obs.mean()) ** 2)
        r2  = 1 - sse / max(sst, 1e-15)

        results["models"][label] = {
            "r": r, "r_over_lambda": r / lam,
            "MSSE": np.mean((Pc_pred - Pc_obs) ** 2),
            "R2_Pc": r2,
            "p_obs": p_obs,
            "Pc_obs": Pc_obs,   "Pc_pred": Pc_pred,
            "P1a_obs": P1a_obs, "P1a_pred": P1a_pred,
            "P1b_obs": P1b_obs, "P1b_pred": P1b_pred,
        }
    return results


# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------

def plot_fits(results, output_dir):
    p_fine  = np.linspace(0, 1, 400)
    alpha   = results["alpha"]
    beta    = results["beta"]
    lam     = results["lambda"]
    q_fine  = q_of_p(p_fine, alpha, beta)

    for label, m in results["models"].items():
        r = m["r"]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(
            f"{label}   (r={r:.3f}, r/λ={m['r_over_lambda']:.3f}, "
            f"α={alpha:.3f}, β={beta:.3f})",
            fontsize=12
        )

        for ax, key, title in zip(
            axes,
            ["Pc", "P1a", "P1b"],
            ["Pc (accuracy)", "P1a (sensitivity)", "P1b (specificity)"],
        ):
            obs  = m[f"{key}_obs"]
            pred = m[f"{key}_pred"]

            if key == "Pc":
                fine = pc_with_p(r, lam, p_fine, alpha, beta)
            elif key == "P1a":
                fine = p1a_with_p(r, lam, p_fine, alpha, beta)
            else:
                fine = p1b_with_p(r, lam, p_fine, alpha, beta)

            ax.scatter(m["p_obs"], obs,  color="red",  zorder=3, label="Observed")
            ax.plot(p_fine, fine,         color="blue", linewidth=2, label="RI fit")
            ax.set_xlabel("Noise level p'")
            ax.set_ylabel(key)
            ax.set_title(title)
            ax.set_ylim(0, 1)
            ax.grid(True)
            ax.legend(fontsize=8)

        fig.tight_layout()
        fname = os.path.join(output_dir, f"{label.replace('-','').replace(' ','_')}_fit.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Plot saved -> {fname}")

    # q vs p curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(p_fine, q_fine, "k-", linewidth=2)
    ax.set_xlabel("Noise level p'")
    ax.set_ylabel("q(p')")
    ax.set_title(f"Noise mapping  q = α·p^β   (α={alpha:.3f}, β={beta:.3f})")
    ax.set_ylim(0, 1)
    ax.grid(True)
    fig.tight_layout()
    fname = os.path.join(output_dir, "q_vs_p.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {fname}")


# ------------------------------------------------------------------------------
# Save parameters CSV
# ------------------------------------------------------------------------------

def save_params(results, output_dir):
    rows = []
    for label, m in results["models"].items():
        rows.append({
            "model":        label,
            "r":            round(m["r"], 6),
            "r_over_lambda": round(m["r_over_lambda"], 6),
            "alpha":        round(results["alpha"], 6),
            "beta":         round(results["beta"], 6),
            "lambda":       round(results["lambda"], 6),
            "MSSE_Pc":      round(m["MSSE"], 8),
            "R2_Pc":        round(m["R2_Pc"], 6),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "fitted_params.csv")
    df.to_csv(path, index=False)
    print(f"\nFitted parameters saved -> {path}")
    print(df.to_string(index=False))
    return df


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data for each model
    models_data = []
    for label, dir_name, _ in MODELS:
        try:
            p_obs, Pc_obs, P1a_obs, P1b_obs = load_summary(dir_name, label)
            models_data.append((label, p_obs, Pc_obs, P1a_obs, P1b_obs))
            print(f"Loaded {label}: {len(p_obs)} noise levels")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")

    if not models_data:
        raise RuntimeError("No model data found. Run compute_metrics.py first.")

    lam = LAMBDA if LAMBDA is not None else 1.0

    print(f"\nFitting RI model (lambda={lam}, fixed)...")
    print(f"Models: {[m[0] for m in models_data]}")
    print(f"Parameters: r per model + shared alpha, beta\n")

    results = fit_joint(models_data, lam)

    print(f"\n{'='*55}")
    print(f"  alpha (shared) = {results['alpha']:.6f}")
    print(f"  beta  (shared) = {results['beta']:.6f}")
    print(f"  lambda (fixed) = {results['lambda']:.6f}")
    print(f"  Joint MSSE     = {results['joint_MSSE']:.8f}")
    for label, m in results["models"].items():
        print(f"  {label:12s}  r={m['r']:.6f}  r/lam={m['r_over_lambda']:.4f}  "
              f"MSSE={m['MSSE']:.6f}  R2={m['R2_Pc']:.4f}")
    print(f"{'='*55}\n")

    save_params(results, OUTPUT_DIR)
    plot_fits(results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
