"""
Recompute all bootstrap-derived outputs using λ = r/x (r = token price).

No bootstrap rerun needed — x values already exist in bootstrap_raw.csv.
This script:
  1. Recomputes λ = price/x for every bootstrap replicate
  2. Updates bootstrap_summary.csv, original_estimates.csv
  3. Recomputes and saves:
       pairwise_model_tests.csv  — within-noise pairwise λ tests (Appendix 1)
       pairwise_tests.csv        — cross-noise same-param tests  (Appendix 2)
  4. Prints Table 1 and saves table1.csv

Usage
-----
    python report/generate_table1.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BOOT_DIR    = "./Dataset/results/bootstrap"
RAW_CSV     = os.path.join(BOOT_DIR, "bootstrap_raw.csv")
ORIG_CSV    = os.path.join(BOOT_DIR, "original_estimates.csv")
SUMM_CSV    = os.path.join(BOOT_DIR, "bootstrap_summary.csv")
PAIR_MODEL  = os.path.join(BOOT_DIR, "pairwise_model_tests.csv")
PAIR_NOISE  = os.path.join(BOOT_DIR, "pairwise_tests.csv")
OUT_TABLE1  = os.path.join(BOOT_DIR, "table1.csv")

MODELS = [
    "GPT-3.5-turbo",
    "GPT-5.4-nano",
    "Gemini-2.5-Flash",
    "Gemini-2.5-Flash-Lite",
]

# r = input token price ($/1M tokens, standard tier, July 2026).
# Justification: all prompts are identical → per-call cost ∝ input price;
# output is a single JSON digit (≤16 tokens), so output cost is negligible.
# r therefore represents the monetary reward the provider charges per decision,
# and λ = r/x gives the true price-adjusted information cost.
PRICE_MAP = {
    "GPT-3.5-turbo":         0.50,
    "GPT-5.4-nano":          0.20,
    "Gemini-2.5-Flash":      0.30,
    "Gemini-2.5-Flash-Lite": 0.10,
}

NOISE_TYPES  = ["white", "babble", "cafe"]
NOISE_LABELS = {"white": "White noise", "babble": "Babble noise", "cafe": "Café noise"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def recompute_lambda_boot(raw_df):
    """Replace lambda_<model> = 1/x with price/x in every replicate."""
    df = raw_df.copy()
    for m in MODELS:
        xcol = f"x_{m}"
        lcol = f"lambda_{m}"
        if xcol not in df.columns:
            print(f"  WARNING: {xcol} missing in bootstrap_raw.csv")
            continue
        df[lcol] = np.where(df[xcol] > 0, PRICE_MAP[m] / df[xcol], np.nan)
    return df


def build_point_estimates(orig_df):
    """
    Return {noise: {param: value}} with lambda recomputed as price/x.
    """
    estimates = {}
    for _, row in orig_df.iterrows():
        estimates.setdefault(row["noise"], {})[row["param"]] = row["estimate"]

    for noise in NOISE_TYPES:
        if noise not in estimates:
            continue
        for m in MODELS:
            xk, lk = f"x_{m}", f"lambda_{m}"
            if xk in estimates[noise]:
                x = estimates[noise][xk]
                estimates[noise][lk] = PRICE_MAP[m] / x if x > 0 else np.nan

    return estimates


def build_summary(boot_df, point_estimates):
    rows = []
    for noise in NOISE_TYPES:
        sub = boot_df[boot_df["noise"] == noise]
        pt  = point_estimates.get(noise, {})
        for param in [c for c in boot_df.columns if c != "noise"]:
            vals = sub[param].dropna().values
            orig = pt.get(param, np.nan)
            rows.append({
                "noise":    noise,
                "param":    param,
                "estimate": round(float(orig),             6),
                "se":       round(float(vals.std(ddof=1)), 6),
                "ci_lo":    round(float(np.percentile(vals, 2.5)),  6),
                "ci_hi":    round(float(np.percentile(vals, 97.5)), 6),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pairwise tests
# ---------------------------------------------------------------------------

def _z_and_ci(va, vb, est_a, est_b):
    se_a = va.std(ddof=1)
    se_b = vb.std(ddof=1)
    se_d = np.sqrt(se_a**2 + se_b**2)
    obs  = est_a - est_b
    z    = obs / se_d if se_d > 0 else np.nan
    p_z  = float(2 * norm.sf(abs(z))) if not np.isnan(z) else np.nan
    B    = min(len(va), len(vb))
    D    = va[:B] - vb[:B]
    ci_lo = float(np.percentile(D, 2.5))
    ci_hi = float(np.percentile(D, 97.5))
    return se_a, se_b, z, p_z, ci_lo, ci_hi


def pairwise_within_noise(boot_dfs, point_estimates):
    """Appendix 1: compare all λ model-pairs within each noise condition."""
    lambda_cols  = [f"lambda_{m}" for m in MODELS]
    model_pairs  = [(MODELS[i], MODELS[j])
                    for i in range(len(MODELS))
                    for j in range(i + 1, len(MODELS))]
    rows = []
    for noise, df in boot_dfs.items():
        orig = point_estimates.get(noise, {})
        for ma, mb in model_pairs:
            ca, cb = f"lambda_{ma}", f"lambda_{mb}"
            if ca not in df.columns or cb not in df.columns:
                continue
            va  = df[ca].dropna().values
            vb  = df[cb].dropna().values
            ea  = orig.get(ca, np.nan)
            eb  = orig.get(cb, np.nan)
            se_a, se_b, z, p_z, ci_lo, ci_hi = _z_and_ci(va, vb, ea, eb)
            rows.append({
                "noise":       noise,
                "model_A":     ma,
                "model_B":     mb,
                "lambda_A":    round(ea,          6),
                "lambda_B":    round(eb,          6),
                "obs_diff":    round(ea - eb,     6),
                "se_A":        round(se_a,        6),
                "se_B":        round(se_b,        6),
                "z_stat":      round(z,    4) if not np.isnan(z)   else np.nan,
                "p_z":         round(p_z,  6) if not np.isnan(p_z) else np.nan,
                "sig_z_0.05":  p_z < 0.05  if not np.isnan(p_z)   else False,
                "ci_lo_diff":  round(ci_lo, 6),
                "ci_hi_diff":  round(ci_hi, 6),
                "sig_ci":      (ci_lo > 0) or (ci_hi < 0),
            })
    return pd.DataFrame(rows)


def pairwise_across_noise(boot_dfs, point_estimates):
    """Appendix 2: compare same parameter across noise conditions."""
    noises = list(boot_dfs.keys())
    noise_pairs = [(noises[i], noises[j])
                   for i in range(len(noises))
                   for j in range(i + 1, len(noises))]

    param_cols = [f"lambda_{m}" for m in MODELS] + ["alpha", "beta"]

    rows = []
    for na, nb in noise_pairs:
        dfa  = boot_dfs[na]
        dfb  = boot_dfs[nb]
        oa   = point_estimates.get(na, {})
        ob   = point_estimates.get(nb, {})

        for param in param_cols:
            if param not in dfa.columns or param not in dfb.columns:
                continue
            va  = dfa[param].dropna().values
            vb  = dfb[param].dropna().values
            ea  = oa.get(param, np.nan)
            eb  = ob.get(param, np.nan)
            se_a, se_b, z, p_z, ci_lo, ci_hi = _z_and_ci(va, vb, ea, eb)
            rows.append({
                "noise_A":    na,
                "noise_B":    nb,
                "param":      param,
                "est_A":      round(ea,       6),
                "est_B":      round(eb,       6),
                "obs_diff":   round(ea - eb,  6),
                "z_stat":     round(z,    4) if not np.isnan(z)   else np.nan,
                "p_z":        round(p_z,  6) if not np.isnan(p_z) else np.nan,
                "sig_z_0.05": p_z < 0.05  if not np.isnan(p_z)   else False,
                "ci_lo_diff": round(ci_lo, 6),
                "ci_hi_diff": round(ci_hi, 6),
                "sig_ci":     (ci_lo > 0) or (ci_hi < 0),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 1 formatting
# ---------------------------------------------------------------------------

def format_table1(summary_df):
    param_order  = [f"lambda_{m}" for m in MODELS] + ["alpha", "beta"]
    param_labels = {f"lambda_{m}": f"lambda  {m}  (r=${PRICE_MAP[m]:.2f})"
                    for m in MODELS}
    param_labels["alpha"] = "alpha  (shared)"
    param_labels["beta"]  = "beta   (shared)"

    header = ["Parameter"] + [NOISE_LABELS[n] for n in NOISE_TYPES]
    rows   = []
    for param in param_order:
        row = [param_labels[param]]
        for noise in NOISE_TYPES:
            r = summary_df[(summary_df["noise"] == noise) &
                           (summary_df["param"] == param)]
            if r.empty:
                row.append("—")
            else:
                e = r.iloc[0]
                row.append(f"{e['estimate']:.4f} +/- {e['se']:.4f}\n"
                            f"[{e['ci_lo']:.4f}, {e['ci_hi']:.4f}]")
        rows.append(row)
    return header, rows


def print_table(header, rows, title=""):
    if title:
        print(f"\n{'='*70}\n{title}\n{'='*70}")
    col_w = [max(len(h), max((len(str(r[i]).split('\n')[0])) for r in rows))
             for i, h in enumerate(header)]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"

    def fmt_row(cells, widths):
        lines = [str(c).split("\n") for c in cells]
        n = max(len(l) for l in lines)
        out = []
        for li in range(n):
            parts = [f" {(lines[ci][li] if li < len(lines[ci]) else ''):<{widths[ci]}} "
                     for ci in range(len(widths))]
            out.append("|" + "|".join(parts) + "|")
        return "\n".join(out)

    print(sep)
    print(fmt_row(header, col_w))
    print(sep)
    for row in rows:
        print(fmt_row(row, col_w))
        print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw_df  = pd.read_csv(RAW_CSV)
    orig_df = pd.read_csv(ORIG_CSV)
    print(f"Loaded {len(raw_df)} bootstrap replicates")

    # ── 1. Recompute lambda = r/x ──────────────────────────────────────────
    boot_df   = recompute_lambda_boot(raw_df)
    point_est = build_point_estimates(orig_df)

    # ── 2. Update bootstrap_summary.csv ────────────────────────────────────
    summary_df = build_summary(boot_df, point_est)
    summary_df.to_csv(SUMM_CSV, index=False)
    print(f"Updated {SUMM_CSV}")

    # Update original_estimates.csv
    updated_rows = []
    for noise in NOISE_TYPES:
        for param, val in point_est.get(noise, {}).items():
            updated_rows.append({"noise": noise, "param": param,
                                  "estimate": round(val, 6)})
    pd.DataFrame(updated_rows).to_csv(ORIG_CSV, index=False)
    print(f"Updated {ORIG_CSV}")

    # ── 3. Pairwise tests ──────────────────────────────────────────────────
    # Split boot_df by noise
    boot_dfs = {n: boot_df[boot_df["noise"] == n].reset_index(drop=True)
                for n in NOISE_TYPES}

    within_df = pairwise_within_noise(boot_dfs, point_est)
    within_df.to_csv(PAIR_MODEL, index=False)
    print(f"Updated {PAIR_MODEL}  ({len(within_df)} rows)")

    across_df = pairwise_across_noise(boot_dfs, point_est)
    across_df.to_csv(PAIR_NOISE, index=False)
    print(f"Updated {PAIR_NOISE}  ({len(across_df)} rows)")

    # ── 4. Table 1 ─────────────────────────────────────────────────────────
    header, rows = format_table1(summary_df)
    print_table(header, rows,
                "TABLE 1 -- Estimated RI parameters (lambda = r/x, B=1000 bootstrap)")

    # save table1.csv
    t1_rows = []
    for row in rows:
        for ni, noise in enumerate(NOISE_TYPES):
            lines = row[ni + 1].split("\n")
            t1_rows.append({
                "parameter": row[0],
                "noise":     NOISE_LABELS[noise],
                "est_se":    lines[0].strip(),
                "ci_95":     lines[1].strip() if len(lines) > 1 else "",
            })
    pd.DataFrame(t1_rows).to_csv(OUT_TABLE1, index=False)
    print(f"\nTable 1 -> {OUT_TABLE1}")

    # ── 5. Quick significance summary ──────────────────────────────────────
    print("\n--- Appendix 1 significant pairs (within noise) ---")
    sig1 = within_df[within_df["sig_ci"] | within_df["sig_z_0.05"]]
    if sig1.empty:
        print("  None")
    else:
        for _, r in sig1.iterrows():
            print(f"  [{r['noise']}]  {r['model_A']} vs {r['model_B']}  "
                  f"diff={r['obs_diff']:.4f}  p_z={r['p_z']:.4f}  "
                  f"CI=[{r['ci_lo_diff']:.4f},{r['ci_hi_diff']:.4f}]")

    print("\n--- Appendix 2 significant pairs (across noise) ---")
    sig2 = across_df[across_df["sig_ci"] | across_df["sig_z_0.05"]]
    if sig2.empty:
        print("  None")
    else:
        for _, r in sig2.iterrows():
            print(f"  {r['param']}  {r['noise_A']} vs {r['noise_B']}  "
                  f"diff={r['obs_diff']:.4f}  p_z={r['p_z']:.4f}  "
                  f"CI=[{r['ci_lo_diff']:.4f},{r['ci_hi_diff']:.4f}]")


if __name__ == "__main__":
    main()
