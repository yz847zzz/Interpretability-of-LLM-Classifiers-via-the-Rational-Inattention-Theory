"""
Generate Table 1: estimated RI parameters per noise condition.

λ = r / x  where r = input token price ($/1M tokens) and x is the directly
fitted RI decision parameter.  This replaces the r=1 assumption used in the
original bootstrap run.  No refitting is needed — x values are reused from
bootstrap_raw.csv; only the λ = r/x mapping changes.

Steps
-----
1. Read bootstrap_raw.csv  (contains x_<model> columns per replicate)
2. Recompute λ_<model> = price / x_<model> for each replicate
3. Point estimates come from original_estimates.csv (x columns), same mapping
4. SE and 95% CI from the recomputed bootstrap distribution
5. Overwrite bootstrap_summary.csv with updated λ values
6. Print Table 1 and save table1.csv

Usage
-----
    python report/generate_table1.py
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BOOT_DIR   = "./Dataset/results/bootstrap"
RAW_CSV    = os.path.join(BOOT_DIR, "bootstrap_raw.csv")
ORIG_CSV   = os.path.join(BOOT_DIR, "original_estimates.csv")
SUMM_CSV   = os.path.join(BOOT_DIR, "bootstrap_summary.csv")
OUT_TABLE1 = os.path.join(BOOT_DIR, "table1.csv")

MODELS = [
    "GPT-3.5-turbo",
    "GPT-5.4-nano",
    "Gemini-2.5-Flash",
    "Gemini-2.5-Flash-Lite",
]

# r = input token price ($/1M tokens, standard tier, July 2026)
# In the RI model: x = r / λ  →  λ = r / x
# Using token price as r because: identical prompts mean per-call cost is
# proportional to input price; output is a single JSON digit (≤16 tokens),
# so output cost is negligible.  r therefore reflects the monetary reward
# the provider effectively charges per decision.
PRICE_MAP = {
    "GPT-3.5-turbo":         0.50,
    "GPT-5.4-nano":          0.20,
    "Gemini-2.5-Flash":      0.30,
    "Gemini-2.5-Flash-Lite": 0.10,
}

NOISE_TYPES  = ["white", "babble", "cafe"]
NOISE_LABELS = {"white": "White noise", "babble": "Babble noise", "cafe": "Café noise"}


# ---------------------------------------------------------------------------
# Step 1: Recompute bootstrap distribution of λ = r/x
# ---------------------------------------------------------------------------

def recompute_lambda_boot(raw_df):
    """Return a copy of raw_df with lambda columns replaced by r/x."""
    df = raw_df.copy()
    for m in MODELS:
        xcol = f"x_{m}"
        lcol = f"lambda_{m}"
        if xcol not in df.columns:
            print(f"  WARNING: {xcol} not found in bootstrap_raw.csv")
            continue
        price = PRICE_MAP[m]
        df[lcol] = np.where(df[xcol] > 0, price / df[xcol], np.nan)
    return df


# ---------------------------------------------------------------------------
# Step 2: Point estimates from original full-data fit
# ---------------------------------------------------------------------------

def build_point_estimates(orig_df):
    """
    Returns dict:  {noise: {param: value}}
    lambda params are recomputed as price/x; x and alpha/beta are kept as-is.
    """
    estimates = {}
    for _, row in orig_df.iterrows():
        noise, param, val = row["noise"], row["param"], row["estimate"]
        estimates.setdefault(noise, {})[param] = val

    # recompute lambda = price/x for each noise condition
    for noise in NOISE_TYPES:
        if noise not in estimates:
            continue
        for m in MODELS:
            xkey = f"x_{m}"
            lkey = f"lambda_{m}"
            if xkey in estimates[noise]:
                x = estimates[noise][xkey]
                estimates[noise][lkey] = PRICE_MAP[m] / x if x > 0 else np.nan

    return estimates


# ---------------------------------------------------------------------------
# Step 3: Build summary (point est + bootstrap SE + 95% CI)
# ---------------------------------------------------------------------------

def build_summary(boot_df, point_estimates):
    rows = []
    for noise in NOISE_TYPES:
        sub = boot_df[boot_df["noise"] == noise] if "noise" in boot_df.columns else boot_df
        pt  = point_estimates.get(noise, {})

        param_cols = [c for c in boot_df.columns if c != "noise"]
        for param in param_cols:
            vals = sub[param].dropna().values
            orig = pt.get(param, np.nan)
            rows.append({
                "noise":    noise,
                "param":    param,
                "estimate": round(float(orig),              6),
                "se":       round(float(vals.std(ddof=1)),  6),
                "ci_lo":    round(float(np.percentile(vals, 2.5)),  6),
                "ci_hi":    round(float(np.percentile(vals, 97.5)), 6),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4: Format and print Table 1
# ---------------------------------------------------------------------------

def format_table1(summary_df):
    """
    Table 1: rows = λ per model + α + β,  cols = noise conditions.
    Cell format: estimate ± SE  [ci_lo, ci_hi]
    """
    param_order = [f"lambda_{m}" for m in MODELS] + ["alpha", "beta"]
    param_labels = {f"lambda_{m}": f"λ  {m}  (r=${PRICE_MAP[m]:.2f})" for m in MODELS}
    param_labels["alpha"] = "α  (shared)"
    param_labels["beta"]  = "β  (shared)"

    header = ["Parameter"] + [NOISE_LABELS[n] for n in NOISE_TYPES]
    rows   = []

    for param in param_order:
        if param not in param_labels:
            continue
        row = [param_labels[param]]
        for noise in NOISE_TYPES:
            r = summary_df[(summary_df["noise"] == noise) &
                           (summary_df["param"] == param)]
            if r.empty:
                row.append("—")
            else:
                e  = r.iloc[0]
                row.append(f"{e['estimate']:.4f} ± {e['se']:.4f}\n"
                            f"[{e['ci_lo']:.4f}, {e['ci_hi']:.4f}]")
        rows.append(row)

    return header, rows


def print_table1(header, rows):
    col_w = [max(len(h), max(len(str(r[i]).split('\n')[0]) for r in rows))
             for i, h in enumerate(header)]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"

    def fmt_row(cells, widths):
        # multi-line cells: split on \n
        lines_per_cell = [str(c).split("\n") for c in cells]
        n_lines = max(len(l) for l in lines_per_cell)
        out = []
        for li in range(n_lines):
            parts = []
            for ci, lines in enumerate(lines_per_cell):
                txt = lines[li] if li < len(lines) else ""
                parts.append(f" {txt:<{widths[ci]}} ")
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

    print(f"Loaded {len(raw_df)} bootstrap replicates from {RAW_CSV}")

    # recompute λ = r/x in bootstrap distribution
    boot_df = recompute_lambda_boot(raw_df)

    # point estimates with new λ
    point_est = build_point_estimates(orig_df)

    # summary
    summary_df = build_summary(boot_df, point_est)

    # overwrite bootstrap_summary.csv
    summary_df.to_csv(SUMM_CSV, index=False)
    print(f"Updated {SUMM_CSV}  (λ = r/x with r = token price)")

    # save original_estimates.csv with updated lambda values
    updated_orig_rows = []
    for noise in NOISE_TYPES:
        pe = point_est.get(noise, {})
        for param, val in pe.items():
            updated_orig_rows.append({"noise": noise, "param": param, "estimate": round(val, 6)})
    pd.DataFrame(updated_orig_rows).to_csv(ORIG_CSV, index=False)
    print(f"Updated {ORIG_CSV}  (λ columns now = r/x)")

    # Table 1
    header, rows = format_table1(summary_df)

    print("\n" + "=" * 80)
    print("TABLE 1  —  Estimated RI parameters  (λ = r/x,  r = input token price)")
    print("           Bootstrap SE and 95% CI,  B = 1 000 replicates")
    print("=" * 80)
    print_table1(header, rows)

    # save as CSV
    table1_rows = []
    for row in rows:
        for ni, noise in enumerate(NOISE_TYPES):
            param_label = row[0]
            cell = row[ni + 1]
            lines = cell.split("\n")
            est_se = lines[0].strip()
            ci     = lines[1].strip() if len(lines) > 1 else ""
            table1_rows.append({
                "parameter": param_label,
                "noise":     NOISE_LABELS[noise],
                "est_se":    est_se,
                "ci_95":     ci,
            })
    pd.DataFrame(table1_rows).to_csv(OUT_TABLE1, index=False)
    print(f"\nTable 1 saved to {OUT_TABLE1}")

    # quick sanity: print new vs old lambda for white noise
    print("\nSanity check — White noise λ (new = r/x  vs  old = 1/x):")
    for m in MODELS:
        r   = summary_df[(summary_df["noise"] == "white") &
                         (summary_df["param"] == f"lambda_{m}")]
        new = r.iloc[0]["estimate"] if not r.empty else float("nan")
        old_x_row = orig_df[orig_df["param"] == f"x_{m}"]
        # orig_df was overwritten, look at point_est
        x_val = point_est.get("white", {}).get(f"x_{m}", float("nan"))
        old   = 1.0 / x_val if x_val > 0 else float("nan")
        print(f"  {m:<28}  new λ={new:.4f}  (was 1/x={old:.4f},  price={PRICE_MAP[m]})")


if __name__ == "__main__":
    main()
