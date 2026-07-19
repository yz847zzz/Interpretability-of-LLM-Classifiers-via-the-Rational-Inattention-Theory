"""
Bootstrap confidence intervals and pairwise tests for RI model parameters
across audio noise conditions (white, babble, cafe).

Method (Efron & Tibshirani, 1993):
  For each noise condition:
    1. Load raw LLM prediction CSVs (one per SNR level × 4 models)
    2. Resample the 100 observation rows with replacement B times
    3. Recompute Pc from resampled predictions at each SNR level
    4. Refit RI model -> record lambda (per model), alpha, beta
  Then: SE = std of bootstrap distribution, 95% CI = [2.5th, 97.5th percentile]
  Pairwise t-test: H0: lambda_A = lambda_B

Output:
  Dataset/results/bootstrap/
      bootstrap_raw.csv          -- B rows × (noise × param) columns
      bootstrap_summary.csv      -- mean, SE, 95% CI per param per noise
      pairwise_tests.csv         -- t-stat and p-value for each pair

Usage:
    python bootstrap_ri.py --B 1000
    python bootstrap_ri.py --B 200   # quick test
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import ttest_ind

# ==============================================================================
# CONFIG
# ==============================================================================
RESULTS_DIR = "./Dataset/results"
NOISE_TYPES = ["white", "babble", "cafe"]

MODELS = [
    ("GPT-3.5-turbo",         "gpt_gpt_3_5_turbo"),
    ("GPT-5.4-nano",          "gpt_gpt_5_4_nano"),
    ("Gemini-2.5-Flash",      "gemini_gemini_2_5_flash"),
    ("Gemini-2.5-Flash-Lite", "gemini_gemini_2_5_flash_lite"),
]

# Input token price ($/1M tokens, standard tier, July 2026)
# r is used as the reward in the RI model: λ = r / x
PRICE_MAP = {
    "GPT-3.5-turbo":         0.50,
    "GPT-5.4-nano":          0.20,
    "Gemini-2.5-Flash":      0.30,
    "Gemini-2.5-Flash-Lite": 0.10,
}

LB_X     = 1e-6;  UB_X     = 100.0
LB_ALPHA = 0.01;  UB_ALPHA = 50.0
LB_BETA  = 0.1;   UB_BETA  = 3.0
# ==============================================================================

_EPS = 1e-10


# ------------------------------------------------------------------------------
# RI model (same as fit_ri_model_audio.py)
# ------------------------------------------------------------------------------

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
    return np.clip(alpha * np.power(np.maximum(nsr, 0.0), beta), 0.0, 1.0)


def pc_model(x, nsr, alpha, beta):
    return ri_pc(x, q_of_nsr(nsr, alpha, beta))


def _joint_sse(params, datasets):
    n     = len(datasets)
    xs    = params[:n]
    alpha = params[n]
    beta  = params[n + 1]
    total = 0.0
    for i, (nsr_obs, Pc_obs) in enumerate(datasets):
        total += np.sum((pc_model(xs[i], nsr_obs, alpha, beta) - Pc_obs) ** 2)
    return total


def fit_joint(datasets):
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

    return {
        "x":     list(best_x[:n]),
        "alpha": best_x[n],
        "beta":  best_x[n + 1],
    }


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def _snr_from_filename(fname):
    """Return NSR (float) for a given CSV filename."""
    import re
    if re.search(r"_p_00\.csv$", fname):
        return 0.0
    m = re.search(r"snr_([mp])(\d+)db", fname)
    if not m:
        return None
    sign = 1 if m.group(1) == "p" else -1
    return 10 ** (-sign * int(m.group(2)) / 10)


def load_noise_data(noise_type):
    """
    Returns dict:
        { model_label: { nsr: (truth_array, pred_array) } }
    Both arrays have length 100 (or n_samples).
    """
    base = os.path.join(RESULTS_DIR, f"audio_{noise_type}")
    data = {}

    for label, dir_name in MODELS:
        model_dir = os.path.join(base, dir_name)
        csvs = (glob.glob(os.path.join(model_dir, "hate_speech_*snr*.csv")) +
                glob.glob(os.path.join(model_dir, "hate_speech_*_p_00.csv")))
        if not csvs:
            print(f"  WARNING: no CSVs for {label} / {noise_type}")
            continue

        pred_col = f"pred_{dir_name}"
        model_nsrs = {}
        for csv_path in csvs:
            nsr = _snr_from_filename(os.path.basename(csv_path))
            if nsr is None:
                continue
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            if "label" not in df.columns or pred_col not in df.columns:
                continue
            truth = df["label"].values.astype(int)
            pred  = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).astype(int).clip(0, 1).values
            model_nsrs[nsr] = (truth, pred)

        if model_nsrs:
            data[label] = model_nsrs

    return data


def compute_pc(truth, pred):
    p_s2 = truth.mean()
    p_s1 = 1.0 - p_s2
    mask1 = truth == 1
    mask0 = truth == 0
    p2b = pred[mask1].mean() if mask1.sum() > 0 else 0.0
    p1a = 1.0 - pred[mask0].mean() if mask0.sum() > 0 else 0.0
    return float(p_s2 * p2b + p_s1 * p1a)


def build_datasets(noise_data, idx=None):
    """
    Build list of (nsr_array, Pc_array) per model for RI fitting.
    idx: row indices to use (None = all rows).
    """
    datasets = []
    labels   = []
    for label, _ in MODELS:
        if label not in noise_data:
            continue
        nsr_vals, pc_vals = [], []
        for nsr, (truth, pred) in sorted(noise_data[label].items()):
            if idx is not None:
                t, p = truth[idx], pred[idx]
            else:
                t, p = truth, pred
            pc = compute_pc(t, p)
            nsr_vals.append(nsr)
            pc_vals.append(pc)
        datasets.append((np.array(nsr_vals), np.array(pc_vals)))
        labels.append(label)
    return datasets, labels


# ------------------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------------------

def bootstrap_noise(noise_type, B, rng, noise_data=None):
    if noise_data is None:
        noise_data = load_noise_data(noise_type)
    if not noise_data:
        return None

    n_samples = len(next(iter(next(iter(noise_data.values())).values()))[0])

    records = []
    for b in range(B):
        idx     = rng.choice(n_samples, size=n_samples, replace=True)
        datasets, labels = build_datasets(noise_data, idx=idx)
        if not datasets:
            continue
        try:
            res = fit_joint(datasets)
        except Exception:
            continue

        row = {}
        for i, label in enumerate(labels):
            x     = res["x"][i]
            price = PRICE_MAP.get(label, 1.0)
            row[f"lambda_{label}"] = price / x if x > 0 else np.nan
            row[f"x_{label}"]      = x
        row["alpha"] = res["alpha"]
        row["beta"]  = res["beta"]
        records.append(row)

        if (b + 1) % 100 == 0:
            print(f"    {noise_type}: {b+1}/{B} done")

    return pd.DataFrame(records)


# ------------------------------------------------------------------------------
# Summary + pairwise tests
# ------------------------------------------------------------------------------

def summarize(boot_df, noise_type, original_estimates):
    """
    Point estimate = original full-data fit (standard bootstrap practice).
    SE and 95% CI come from bootstrap distribution.
    Ref: Efron & Tibshirani (1993), Ch. 6.
    """
    rows = []
    for col in boot_df.columns:
        vals = boot_df[col].dropna().values
        orig = original_estimates.get(col, np.nan)
        rows.append({
            "noise":    noise_type,
            "param":    col,
            "estimate": round(orig, 6),          # original full-data fit
            "se":       round(vals.std(ddof=1), 6),
            "ci_lo":    round(np.percentile(vals, 2.5),  6),
            "ci_hi":    round(np.percentile(vals, 97.5), 6),
        })
    return pd.DataFrame(rows)


def pairwise_tests(boot_dfs, orig_estimates_all=None):
    """
    Two tests combined:

    1. z-test: z = (est_A - est_B) / sqrt(SE_A^2 + SE_B^2), asymptotic normal.
       Valid when bootstrap distributions are approximately normal.
       Ref: Efron & Tibshirani (1993) bootstrap SE method.

    2. Bootstrap percentile test: CI of D^b = lambda_A^b - lambda_B^b.
       Non-parametric — valid for skewed distributions.
       p = proportion of |D^b| >= |obs_diff|.
       Ref: Efron & Tibshirani (1993) Ch.16; Davison & Hinkley (1997) Ch.4.4

    Compares the SAME parameter across different noise types
    (e.g., lambda_GPT-3.5 in white vs babble vs cafe).
    """
    from scipy.stats import norm
    noises = list(boot_dfs.keys())
    pairs  = [(noises[i], noises[j])
              for i in range(len(noises)) for j in range(i+1, len(noises))]
    rows = []
    cols = [c for c in next(iter(boot_dfs.values())).columns
            if c.startswith("lambda_") or c in ("alpha", "beta")]

    ses = {n: {c: boot_dfs[n][c].dropna().std(ddof=1) for c in cols}
           for n in noises}

    for a, b in pairs:
        for col in cols:
            va = boot_dfs[a][col].dropna().values
            vb = boot_dfs[b][col].dropna().values
            B  = min(len(va), len(vb))

            est_a = orig_estimates_all[a].get(col, np.nan) if orig_estimates_all else np.nan
            est_b = orig_estimates_all[b].get(col, np.nan) if orig_estimates_all else np.nan
            obs_diff = est_a - est_b

            # --- z-test ---
            se_diff = np.sqrt(ses[a][col]**2 + ses[b][col]**2)
            z = obs_diff / se_diff if se_diff > 0 else np.nan
            p_z = float(2 * norm.sf(abs(z))) if not np.isnan(z) else np.nan

            # --- bootstrap percentile test ---
            D     = va[:B] - vb[:B]
            ci_lo = float(np.percentile(D, 2.5))
            ci_hi = float(np.percentile(D, 97.5))
            ci_sig = (ci_lo > 0) or (ci_hi < 0)   # 0 not in CI → significant

            rows.append({
                "noise_A":    a,
                "noise_B":    b,
                "param":      col,
                "est_A":      round(est_a,    6),
                "est_B":      round(est_b,    6),
                "obs_diff":   round(obs_diff, 6),
                # z-test (valid when bootstrap dist is ~normal)
                "z_stat":     round(z,   4) if not np.isnan(z) else np.nan,
                "p_z":        round(p_z, 6) if not np.isnan(p_z) else np.nan,
                "sig_z_0.05": p_z < 0.05 if not np.isnan(p_z) else False,
                # bootstrap 95% CI for the difference (λ_A - λ_B)
                "ci_lo_diff": round(ci_lo, 6),
                "ci_hi_diff": round(ci_hi, 6),
                "sig_ci":     ci_sig,   # True = CI excludes 0
            })
    return pd.DataFrame(rows)


def pairwise_models_within_noise(boot_dfs, orig_estimates_all=None):
    """
    Pairwise comparison of all model-pairs' lambda values WITHIN the same noise type.

    For each noise type, compares every pair of models (A, B):
    1. z-test: z = (lambda_A - lambda_B) / sqrt(SE_A^2 + SE_B^2)
    2. Bootstrap percentile CI for the difference (lambda_A - lambda_B)

    Ref: same as pairwise_tests above.
    """
    from scipy.stats import norm

    lambda_cols = [c for c in next(iter(boot_dfs.values())).columns
                   if c.startswith("lambda_")]
    model_labels = [c[len("lambda_"):] for c in lambda_cols]

    model_pairs = [(model_labels[i], model_labels[j])
                   for i in range(len(model_labels))
                   for j in range(i + 1, len(model_labels))]

    rows = []
    for noise, df in boot_dfs.items():
        orig = orig_estimates_all.get(noise, {}) if orig_estimates_all else {}

        for ma, mb in model_pairs:
            col_a = f"lambda_{ma}"
            col_b = f"lambda_{mb}"
            if col_a not in df.columns or col_b not in df.columns:
                continue

            va = df[col_a].dropna().values
            vb = df[col_b].dropna().values
            B  = min(len(va), len(vb))

            est_a = orig.get(col_a, np.nan)
            est_b = orig.get(col_b, np.nan)
            obs_diff = est_a - est_b

            # --- z-test ---
            se_a = va.std(ddof=1)
            se_b = vb.std(ddof=1)
            se_diff = np.sqrt(se_a**2 + se_b**2)
            z   = obs_diff / se_diff if se_diff > 0 else np.nan
            p_z = float(2 * norm.sf(abs(z))) if not np.isnan(z) else np.nan

            # --- bootstrap percentile CI ---
            D     = va[:B] - vb[:B]
            ci_lo = float(np.percentile(D, 2.5))
            ci_hi = float(np.percentile(D, 97.5))
            ci_sig = (ci_lo > 0) or (ci_hi < 0)

            rows.append({
                "noise":      noise,
                "model_A":    ma,
                "model_B":    mb,
                "lambda_A":   round(est_a,    6),
                "lambda_B":   round(est_b,    6),
                "obs_diff":   round(obs_diff, 6),
                "se_A":       round(se_a,     6),
                "se_B":       round(se_b,     6),
                "z_stat":     round(z,   4) if not np.isnan(z) else np.nan,
                "p_z":        round(p_z, 6) if not np.isnan(p_z) else np.nan,
                "sig_z_0.05": p_z < 0.05 if not np.isnan(p_z) else False,
                "ci_lo_diff": round(ci_lo, 6),
                "ci_hi_diff": round(ci_hi, 6),
                "sig_ci":     ci_sig,
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bootstrap RI parameters across noise conditions")
    parser.add_argument("--B",    type=int, default=1000, help="Bootstrap iterations (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,   help="Random seed")
    args = parser.parse_args()

    out_dir = os.path.join(RESULTS_DIR, "bootstrap")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    boot_dfs  = {}
    summaries = []

    orig_estimates_all = {}

    for noise in NOISE_TYPES:
        print(f"\n[{noise}] Fitting on full dataset ...")
        noise_data = load_noise_data(noise)
        if not noise_data:
            print(f"  Skipped (no data).")
            continue

        # --- original full-data fit ---
        datasets, labels = build_datasets(noise_data, idx=None)
        orig_res = fit_joint(datasets)
        orig_est = {}
        for i, label in enumerate(labels):
            x     = orig_res["x"][i]
            price = PRICE_MAP.get(label, 1.0)
            orig_est[f"lambda_{label}"] = price / x if x > 0 else np.nan
            orig_est[f"x_{label}"]      = x
        orig_est["alpha"] = orig_res["alpha"]
        orig_est["beta"]  = orig_res["beta"]
        orig_estimates_all[noise] = orig_est

        # --- bootstrap ---
        print(f"  Bootstrapping (B={args.B}) ...")
        df = bootstrap_noise(noise, args.B, rng, noise_data)
        if df is None or df.empty:
            continue
        boot_dfs[noise] = df
        summaries.append(summarize(df, noise, orig_est))

    # Save original estimates
    orig_rows = []
    for noise, est in orig_estimates_all.items():
        for k, v in est.items():
            orig_rows.append({"noise": noise, "param": k, "estimate": round(v, 6)})
    orig_path = os.path.join(out_dir, "original_estimates.csv")
    pd.DataFrame(orig_rows).to_csv(orig_path, index=False)
    print(f"\nOriginal estimates -> {orig_path}")

    # Save raw bootstrap samples
    raw_path = os.path.join(out_dir, "bootstrap_raw.csv")
    pd.concat(
        [df.assign(noise=n) for n, df in boot_dfs.items()],
        ignore_index=True
    ).to_csv(raw_path, index=False)
    print(f"\nRaw bootstrap -> {raw_path}")

    # Summary table (mean ± SE, 95% CI)
    summary_df = pd.concat(summaries, ignore_index=True)
    summary_path = os.path.join(out_dir, "bootstrap_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary        -> {summary_path}")

    # Pivot for display: param × noise
    pivot = summary_df[summary_df["param"].str.startswith("lambda_") |
                       summary_df["param"].isin(["alpha","beta"])].copy()
    pivot["value"] = pivot.apply(lambda r: f"{r['estimate']:.4f} ± {r['se']:.4f}", axis=1)
    table = pivot.pivot(index="param", columns="noise", values="value")
    print("\n" + "="*65)
    print("  RI Parameters  (mean ± SE across bootstrap)")
    print("="*65)
    print(table.to_string())

    # Table 1: pairwise tests — same parameter across noise types
    if len(boot_dfs) >= 2:
        pairs_df = pairwise_tests(boot_dfs, orig_estimates_all)
        pairs_path = os.path.join(out_dir, "pairwise_tests.csv")
        pairs_df.to_csv(pairs_path, index=False)
        print(f"\nPairwise tests (across noise types) -> {pairs_path}")
        sig = pairs_df[pairs_df["sig_z_0.05"] | pairs_df["sig_ci"]]
        if sig.empty:
            print("  No significant differences at p < 0.05.")
        else:
            print(f"\n  Significant (p < 0.05 or CI excludes 0):")
            print(sig[["noise_A","noise_B","param","est_A","est_B","obs_diff",
                        "z_stat","p_z","sig_z_0.05","ci_lo_diff","ci_hi_diff","sig_ci"]
                      ].to_string(index=False))

    # Table 2: pairwise model comparisons — lambda within each noise type
    if boot_dfs:
        model_pairs_df = pairwise_models_within_noise(boot_dfs, orig_estimates_all)
        model_pairs_path = os.path.join(out_dir, "pairwise_model_tests.csv")
        model_pairs_df.to_csv(model_pairs_path, index=False)
        print(f"\nModel pairwise tests (within noise type) -> {model_pairs_path}")

        print("\n" + "="*80)
        print("  Lambda pairwise model comparison within each noise type")
        print("="*80)
        display_cols = ["noise","model_A","model_B","lambda_A","lambda_B","obs_diff",
                        "z_stat","p_z","sig_z_0.05","ci_lo_diff","ci_hi_diff","sig_ci"]
        for noise in NOISE_TYPES:
            sub = model_pairs_df[model_pairs_df["noise"] == noise]
            if sub.empty:
                continue
            print(f"\n  -- {noise} noise --")
            print(sub[display_cols].to_string(index=False))

        sig_m = model_pairs_df[model_pairs_df["sig_z_0.05"] | model_pairs_df["sig_ci"]]
        if sig_m.empty:
            print("\n  No significant lambda differences between models at p < 0.05.")
        else:
            print(f"\n  Significant model pairs (p < 0.05 or CI excludes 0):")
            print(sig_m[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
