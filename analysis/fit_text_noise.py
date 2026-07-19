"""
Fit RI model and bootstrap SE for the text-noise dataset.

Dataset: Dataset/hs_sample_400/
Models:  Gemini-2.0, Gemini-2.5, GPT-3.5-turbo, GPT-5.2
Noise levels: p_00 (0.0) to p_10 (1.0), 400 rows each.

Method (Efron & Tibshirani, 1993):
  1. Load raw per-row CSVs for all 4 models × 11 noise levels
  2. Fit RI model jointly (SSE minimisation, shared alpha/beta, per-model x)
  3. Resample 400 rows with replacement B times
  4. Refit -> record lambda (per model), alpha, beta per bootstrap replicate
  5. SE = std of bootstrap dist; 95% CI = [2.5th, 97.5th percentile]

Pairwise lambda comparison:
  Primary:   bootstrap 95% CI for difference (lambda_A - lambda_B)
             significant when CI excludes 0 (non-parametric, valid for skewed dists)
  Secondary: z-test (reported; may be unreliable — lambda=1/x is nonlinear)

Output:
  Dataset/hs_sample_400/results/
    bootstrap_raw.csv           -- B rows of bootstrap parameter estimates
    bootstrap_summary.csv       -- estimate, SE, 95% CI per parameter
    pairwise_lambda_tests.csv   -- pairwise lambda significance between all model pairs

Usage:
    python fit_text_noise.py
    python fit_text_noise.py --B 200   # quick test
    python fit_text_noise.py --B 1000 --seed 0
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# ==============================================================================
# CONFIG
# ==============================================================================
DATA_DIR = "./Dataset/hs_sample_400"
OUT_DIR  = "./Dataset/hs_sample_400/results"

MODELS = [
    # (display_label, subdir, truth_col, pred_col)
    ("Gemini-2.0",    "gemini_res_2_0",        "hs_state", "hatespeech_LLM_gemini"),
    ("Gemini-2.5",    "gemini_res_2_5",        "hs_state", "hatespeech_LLM_gemini"),
    ("GPT-3.5-turbo", "GPTResult_MIXED_NOISE", "hs_state", "hatespeech_llm"),
    ("GPT-5.2",       "GPTres_gpt_5_2",        "hs_state", "hatespeech_LLM_gpt_5_2"),
]

NOISE_CODES  = [f"p_{i:02d}" for i in range(11)]   # p_00, p_01, ..., p_10
NOISE_LEVELS = [i / 10.0    for i in range(11)]    # 0.0,  0.1,  ..., 1.0

LB_X     = 1e-6;  UB_X     = 100.0
LB_ALPHA = 0.0;   UB_ALPHA = 1.0
LB_BETA  = 0.0;   UB_BETA  = 5.0
# ==============================================================================

_EPS = 1e-10


# ── RI model (identical to fit_ri_model.py) ───────────────────────────────────

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


def q_of_p(p, alpha, beta):
    return np.clip(alpha * np.power(np.maximum(p, 0.0), beta), 0.0, 1.0)


def pc_model(x, p, alpha, beta):
    return ri_pc(x, q_of_p(p, alpha, beta))


def _joint_sse(params, datasets):
    n     = len(datasets)
    xs    = params[:n]
    alpha = params[n]
    beta  = params[n + 1]
    total = 0.0
    for i, (p_obs, Pc_obs) in enumerate(datasets):
        total += np.sum((pc_model(xs[i], p_obs, alpha, beta) - Pc_obs) ** 2)
    return total


def fit_joint(datasets):
    """Minimise SSE jointly. Returns {x: list, alpha: float, beta: float}."""
    n   = len(datasets)
    lb  = np.array([LB_X] * n + [LB_ALPHA, LB_BETA])
    ub  = np.array([UB_X] * n + [UB_ALPHA, UB_BETA])
    mid = (lb + ub) / 2.0
    x0s = [mid, np.clip(lb + 1e-3, lb + 1e-9, ub - 1e-9)]

    best_x, best_val = None, np.inf
    for x0 in x0s:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(_joint_sse, x0, args=(datasets,),
                           method="L-BFGS-B", bounds=list(zip(lb, ub)),
                           options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8})
        if res.fun < best_val:
            best_x, best_val = res.x, res.fun

    return {"x": list(best_x[:n]), "alpha": float(best_x[n]), "beta": float(best_x[n + 1])}


# ── Data loading ──────────────────────────────────────────────────────────────

def compute_pc(truth, pred):
    """Pc = P(Y=hate)*P(pred=hate|Y=hate) + P(Y=no-hate)*P(pred=no-hate|Y=no-hate)."""
    p_s2 = truth.mean()
    p_s1 = 1.0 - p_s2
    mask1 = truth == 1
    mask0 = truth == 0
    p2b = pred[mask1].mean() if mask1.sum() > 0 else 0.0
    p1a = 1.0 - pred[mask0].mean() if mask0.sum() > 0 else 0.0
    return float(p_s2 * p2b + p_s1 * p1a)


def load_model_data(label, subdir, truth_col, pred_col):
    """
    Load row-level CSVs for one model across all noise levels.
    Returns: {noise_p: (truth_array, pred_array)}
    """
    model_dir = os.path.join(DATA_DIR, subdir)
    data = {}
    for code, noise_p in zip(NOISE_CODES, NOISE_LEVELS):
        pattern = os.path.join(model_dir, f"traindataset_{code}_sampled400_*.csv")
        files   = glob.glob(pattern)
        if not files:
            print(f"  WARNING: no file for {label} at noise level {code}")
            continue
        df = pd.read_csv(files[0])
        if truth_col not in df.columns or pred_col not in df.columns:
            print(f"  WARNING: missing columns in {files[0]}")
            continue
        truth = df[truth_col].values.astype(int)
        pred  = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).astype(int).clip(0, 1).values
        data[noise_p] = (truth, pred)
    return data


def load_all_models():
    """Returns {label: {noise_p: (truth_array, pred_array)}}"""
    all_data = {}
    for label, subdir, truth_col, pred_col in MODELS:
        data = load_model_data(label, subdir, truth_col, pred_col)
        if data:
            all_data[label] = data
            print(f"  Loaded {label}: {len(data)} noise levels")
        else:
            print(f"  WARNING: no data loaded for {label}")
    return all_data


def build_datasets(all_data, idx=None):
    """
    Build list of (p_obs, Pc_obs) per model for joint RI fitting.
    idx: row indices for bootstrapping (None = full data).
    """
    datasets = []
    labels   = []
    for label, _, _, _ in MODELS:
        if label not in all_data:
            continue
        p_vals, pc_vals = [], []
        for noise_p, (truth, pred) in sorted(all_data[label].items()):
            t = truth[idx] if idx is not None else truth
            p = pred[idx]  if idx is not None else pred
            p_vals.append(noise_p)
            pc_vals.append(compute_pc(t, p))
        datasets.append((np.array(p_vals), np.array(pc_vals)))
        labels.append(label)
    return datasets, labels


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap(all_data, B, rng):
    n_samples = len(next(iter(next(iter(all_data.values())).values()))[0])
    records   = []

    for b in range(B):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        datasets, labels = build_datasets(all_data, idx=idx)
        if not datasets:
            continue
        try:
            res = fit_joint(datasets)
        except Exception:
            continue

        row = {}
        for i, label in enumerate(labels):
            x = res["x"][i]
            row[f"lambda_{label}"] = 1.0 / x if x > 0 else np.nan
            row[f"x_{label}"]      = x
        row["alpha"] = res["alpha"]
        row["beta"]  = res["beta"]
        records.append(row)

        if (b + 1) % 100 == 0:
            print(f"  Bootstrap: {b+1}/{B} done")

    return pd.DataFrame(records)


# ── Summary + tests ───────────────────────────────────────────────────────────

def summarize(boot_df, orig_estimates):
    """Point estimate = full-data fit; SE and CI from bootstrap distribution."""
    rows = []
    for col in boot_df.columns:
        vals = boot_df[col].dropna().values
        orig = orig_estimates.get(col, np.nan)
        rows.append({
            "param":    col,
            "estimate": round(float(orig), 6),
            "se":       round(float(vals.std(ddof=1)), 6),
            "ci_lo":    round(float(np.percentile(vals, 2.5)),  6),
            "ci_hi":    round(float(np.percentile(vals, 97.5)), 6),
            "b_mean":   round(float(vals.mean()), 6),
        })
    return pd.DataFrame(rows)


def pairwise_lambda(boot_df, orig_estimates):
    """
    All model-pair lambda comparisons.
    Primary test: bootstrap 95% CI for difference (CI excludes 0 → significant).
    Secondary:    z-test (lambda=1/x is nonlinear — distribution may be skewed).
    """
    lambda_cols  = [c for c in boot_df.columns if c.startswith("lambda_")]
    model_labels = [c[len("lambda_"):] for c in lambda_cols]
    pairs = [(model_labels[i], model_labels[j])
             for i in range(len(model_labels))
             for j in range(i + 1, len(model_labels))]

    rows = []
    for ma, mb in pairs:
        col_a = f"lambda_{ma}"
        col_b = f"lambda_{mb}"
        va    = boot_df[col_a].dropna().values
        vb    = boot_df[col_b].dropna().values
        B     = min(len(va), len(vb))

        est_a    = orig_estimates.get(col_a, np.nan)
        est_b    = orig_estimates.get(col_b, np.nan)
        obs_diff = float(est_a - est_b)

        se_a    = float(va.std(ddof=1))
        se_b    = float(vb.std(ddof=1))
        se_diff = np.sqrt(se_a**2 + se_b**2)
        z   = obs_diff / se_diff if se_diff > 0 else np.nan
        p_z = float(2 * norm.sf(abs(z))) if not np.isnan(z) else np.nan

        D      = va[:B] - vb[:B]
        ci_lo  = float(np.percentile(D, 2.5))
        ci_hi  = float(np.percentile(D, 97.5))
        ci_sig = (ci_lo > 0) or (ci_hi < 0)

        rows.append({
            "model_A":    ma,
            "model_B":    mb,
            "lambda_A":   round(est_a,    6),
            "lambda_B":   round(est_b,    6),
            "obs_diff":   round(obs_diff, 6),
            "se_A":       round(se_a,     6),
            "se_B":       round(se_b,     6),
            "z_stat":     round(z,   4) if not np.isnan(z) else np.nan,
            "p_z":        round(p_z, 6) if not np.isnan(p_z) else np.nan,
            "sig_z_0.05": (p_z < 0.05) if not np.isnan(p_z) else False,
            "ci_lo_diff": round(ci_lo, 6),
            "ci_hi_diff": round(ci_hi, 6),
            "sig_ci":     ci_sig,
        })

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fit RI model + bootstrap SE for text-noise dataset")
    parser.add_argument("--B",    type=int, default=1000,
                        help="Bootstrap iterations (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data ...")
    all_data = load_all_models()
    if not all_data:
        raise RuntimeError("No data loaded. Check DATA_DIR and MODELS config.")

    # ── Full-data fit ─────────────────────────────────────────────────────────
    print("\nFitting RI model on full dataset ...")
    datasets, labels = build_datasets(all_data)
    orig_res = fit_joint(datasets)

    orig_estimates = {}
    for i, label in enumerate(labels):
        x = orig_res["x"][i]
        orig_estimates[f"lambda_{label}"] = 1.0 / x if x > 0 else np.nan
        orig_estimates[f"x_{label}"]      = x
    orig_estimates["alpha"] = orig_res["alpha"]
    orig_estimates["beta"]  = orig_res["beta"]

    print("\n" + "="*60)
    print("  Full-data RI fit")
    print("="*60)
    print(f"  alpha (shared) = {orig_res['alpha']:.6f}")
    print(f"  beta  (shared) = {orig_res['beta']:.6f}")
    for i, label in enumerate(labels):
        x   = orig_res["x"][i]
        lam = 1.0 / x if x > 0 else float("nan")
        print(f"  {label:20s}  x = {x:.4f},  lambda = {lam:.4f}")
    print("="*60)

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    print(f"\nBootstrapping (B={args.B}, seed={args.seed}) ...")
    boot_df = bootstrap(all_data, args.B, rng)

    if boot_df.empty:
        print("Bootstrap produced no results.")
        return

    raw_path = os.path.join(OUT_DIR, "bootstrap_raw.csv")
    boot_df.to_csv(raw_path, index=False)
    print(f"\nRaw bootstrap ({len(boot_df)} replicates) -> {raw_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_df   = summarize(boot_df, orig_estimates)
    summary_path = os.path.join(OUT_DIR, "bootstrap_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary -> {summary_path}")

    print("\n" + "="*70)
    print("  RI Parameters  (estimate ± bootstrap SE, 95% CI)")
    print("="*70)
    for _, row in summary_df.iterrows():
        if row["param"].startswith("lambda_") or row["param"] in ("alpha", "beta"):
            print(f"  {row['param']:30s}  {row['estimate']:.4f} ± {row['se']:.4f}"
                  f"   [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]")

    # ── Pairwise lambda tests ─────────────────────────────────────────────────
    pairs_df  = pairwise_lambda(boot_df, orig_estimates)
    pairs_path = os.path.join(OUT_DIR, "pairwise_lambda_tests.csv")
    pairs_df.to_csv(pairs_path, index=False)
    print(f"\nPairwise lambda tests -> {pairs_path}")

    disp = ["model_A","model_B","lambda_A","lambda_B","obs_diff",
            "se_A","se_B","z_stat","p_z","sig_z_0.05",
            "ci_lo_diff","ci_hi_diff","sig_ci"]

    print("\n" + "="*90)
    print("  Pairwise Lambda Comparison  (primary: CI test; secondary: z-test)")
    print("  Note: lambda=1/x is nonlinear — prefer CI test for skewed distributions")
    print("="*90)
    print(pairs_df[disp].to_string(index=False))

    sig_ci = pairs_df[pairs_df["sig_ci"]]
    print()
    if sig_ci.empty:
        print("  No significant lambda differences (CI test) at 5% level.")
    else:
        print("  Significant pairs (CI test — CI excludes 0):")
        print(sig_ci[disp].to_string(index=False))

    sig_z = pairs_df[pairs_df["sig_z_0.05"]]
    if not sig_z.empty:
        print("\n  Significant pairs (z-test, p < 0.05):")
        print(sig_z[disp].to_string(index=False))


if __name__ == "__main__":
    main()
