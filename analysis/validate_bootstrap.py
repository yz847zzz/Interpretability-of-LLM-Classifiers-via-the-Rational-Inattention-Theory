"""
Bootstrap validation using multiple linear regression (two predictors).

Model:  y = β₀ + β₁x₁ + β₂x₂ + ε,   ε ~ N(0, σ²)

OLS estimator β̂ is exactly N(β, σ²(X'X)⁻¹) under Gaussian errors.
Analytical SE:  SE(β̂ⱼ) = σ · √[(X'X)⁻¹ⱼⱼ]   — known reference.

Three datasets:
  A: β₁=2.0,  β₂=1.5   (reference)
  B: β₁=3.5,  β₂=1.5   (β₁ clearly different from A)
  C: β₁=2.05, β₂=1.5   (β₁ very close to A)

Three checks:
  1. SE accuracy  — bootstrap SE ≈ analytical SE for both β₁ and β₂
  2. CI coverage  — 95% bootstrap CI covers true β ≈ 95% of the time
  3. Two-sample tests
       (a) H0: β₁_A = β₁_B  →  FPR ≈ 0.05
       (b) Summary table: β₁ and β₂ for all three datasets
           Tests: A vs B (expect reject) and A vs C (expect fail to reject)

Usage:
    python validate_bootstrap.py                          # n=100, B=1000, S=500
    python validate_bootstrap.py --n 50 --B 200 --S 100  # quick
"""

import argparse
import numpy as np
from scipy.stats import norm

TRUE_BETA0   = 1.0
TRUE_BETA2   = 1.5   # same for all datasets (only β₁ varies across A/B/C)
TRUE_BETA1_A = 2.0
TRUE_BETA1_B = 3.5
TRUE_BETA1_C = 2.05
TRUE_SIGMA   = 1.0


# ── Core functions ─────────────────────────────────────────────────────────────

def generate_data(beta1, beta2, n, rng):
    x1  = rng.normal(0, 1, n)
    x2  = rng.normal(0, 1, n)
    eps = rng.normal(0, TRUE_SIGMA, n)
    y   = TRUE_BETA0 + beta1 * x1 + beta2 * x2 + eps
    return x1, x2, y


def fit_ols(x1, x2, y):
    """OLS estimates (β̂₁, β̂₂)."""
    X    = np.column_stack([np.ones(len(x1)), x1, x2])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(beta[1]), float(beta[2])


def analytical_se(x1, x2):
    """Theoretical SE = σ · √diag[(X'X)⁻¹].  Requires known σ (simulation only)."""
    X       = np.column_stack([np.ones(len(x1)), x1, x2])
    XtX_inv = np.linalg.inv(X.T @ X)
    return (TRUE_SIGMA * np.sqrt(XtX_inv[1, 1]),
            TRUE_SIGMA * np.sqrt(XtX_inv[2, 2]))


def ols_se(x1, x2, y):
    """OLS SE = σ̂ · √diag[(X'X)⁻¹],  σ̂ = √(RSS / (n−p)).  Usable in practice."""
    X       = np.column_stack([np.ones(len(x1)), x1, x2])
    beta    = np.linalg.lstsq(X, y, rcond=None)[0]
    rss     = np.sum((y - X @ beta) ** 2)
    n, p    = X.shape
    sigma_hat_sq = rss / (n - p)
    XtX_inv = np.linalg.inv(X.T @ X)
    return (np.sqrt(sigma_hat_sq * XtX_inv[1, 1]),
            np.sqrt(sigma_hat_sq * XtX_inv[2, 2]))


def bootstrap_betas(x1, x2, y, B, rng):
    """Bootstrap distributions of β̂₁ and β̂₂."""
    n  = len(x1)
    b1 = np.empty(B)
    b2 = np.empty(B)
    for b in range(B):
        idx     = rng.choice(n, size=n, replace=True)
        b1[b], b2[b] = fit_ols(x1[idx], x2[idx], y[idx])
    return b1, b2


# ── Validation 1: SE accuracy ──────────────────────────────────────────────────

def val1_se(n, B, rng):
    print("\n" + "=" * 60)
    print("  Validation 1: Three SE estimates — Theoretical, OLS, Bootstrap")
    print("=" * 60)

    x1, x2, y = generate_data(TRUE_BETA1_A, TRUE_BETA2, n, rng)
    se_an1, se_an2   = analytical_se(x1, x2)
    se_ols1, se_ols2 = ols_se(x1, x2, y)
    boot1, boot2     = bootstrap_betas(x1, x2, y, B, rng)
    se_bo1 = boot1.std(ddof=1)
    se_bo2 = boot2.std(ddof=1)

    w = 56
    print()
    print("  " + "─" * w)
    print(f"  {'':6} {'Theoretical SE':>16} {'OLS SE':>10} {'Bootstrap SE':>14}")
    print(f"  {'':6} {'σ·√[(X′X)⁻¹]':>16} {'σ̂·√[(X′X)⁻¹]':>10} {'std(β̂*)':>14}")
    print("  " + "─" * w)
    print(f"  {'β₁':<6} {se_an1:>16.4f} {se_ols1:>10.4f} {se_bo1:>14.4f}")
    print(f"  {'β₂':<6} {se_an2:>16.4f} {se_ols2:>10.4f} {se_bo2:>14.4f}")
    print("  " + "─" * w)
    print()
    print("  Theoretical SE: uses true σ (simulation only).")
    print("  OLS SE:         uses σ̂ from residuals (practical).")
    print("  Bootstrap SE:   non-parametric, no distributional assumption.")


# ── Validation 2: CI coverage ──────────────────────────────────────────────────

def val2_coverage(n, B, S, rng):
    print("\n" + "=" * 60)
    print("  Validation 2: 95% Bootstrap CI Coverage")
    print("=" * 60)
    print(f"  {S} independent datasets × {B} bootstrap resamples each.")

    cov1 = cov2 = 0
    for _ in range(S):
        x1, x2, y     = generate_data(TRUE_BETA1_A, TRUE_BETA2, n, rng)
        boot1, boot2   = bootstrap_betas(x1, x2, y, B, rng)
        lo1, hi1 = np.percentile(boot1, [2.5, 97.5])
        lo2, hi2 = np.percentile(boot2, [2.5, 97.5])
        if lo1 <= TRUE_BETA1_A <= hi1: cov1 += 1
        if lo2 <= TRUE_BETA2   <= hi2: cov2 += 1

    print(f"\n  {'':6} {'Coverage':>10}  {'Target':>8}  {'Pass?':>6}")
    print("  " + "─" * 36)
    for label, cov in [("β₁", cov1/S), ("β₂", cov2/S)]:
        ok = "PASS" if 0.90 <= cov <= 1.00 else "FAIL"
        print(f"  {label:<6} {cov:>10.3f}  {'0.95':>8}  {ok:>6}")


# ── Validation 3: Two-sample test ──────────────────────────────────────────────

def val3_two_sample(n, B, S, rng):
    print("\n" + "=" * 60)
    print("  Validation 3: Two-Sample Z-Test and CI")
    print("=" * 60)

    # ── 3a: H0 true — FPR check ───────────────────────────────────────────
    print(f"\n  3a. H0: β₁_A = β₁_B = {TRUE_BETA1_A}  →  FPR should be ≈ 0.05")
    print(f"      {S} independent pairs, {B} bootstrap resamples each.")

    fp_z = fp_ci = 0
    for _ in range(S):
        x1A, x2A, yA = generate_data(TRUE_BETA1_A, TRUE_BETA2, n, rng)
        x1B, x2B, yB = generate_data(TRUE_BETA1_A, TRUE_BETA2, n, rng)
        bA1, _ = bootstrap_betas(x1A, x2A, yA, B, rng)
        bB1, _ = bootstrap_betas(x1B, x2B, yB, B, rng)
        eA1, _ = fit_ols(x1A, x2A, yA)
        eB1, _ = fit_ols(x1B, x2B, yB)
        se_d   = np.sqrt(bA1.std(ddof=1)**2 + bB1.std(ddof=1)**2)
        z      = (eA1 - eB1) / se_d
        p_z    = 2 * norm.sf(abs(z))
        D      = bA1 - bB1
        lo, hi = np.percentile(D, [2.5, 97.5])
        if p_z < 0.05:           fp_z  += 1
        if (lo > 0) or (hi < 0): fp_ci += 1

    print(f"\n      FPR z-test : {fp_z/S:.3f}   (expected ≈ 0.05)")
    print(f"      FPR CI     : {fp_ci/S:.3f}   (expected ≈ 0.05)")

    # ── 3b: All three datasets — summary + pairwise tests ─────────────────
    print(f"\n  3b. Three datasets: A (β₁={TRUE_BETA1_A}), "
          f"B (β₁={TRUE_BETA1_B}), C (β₁={TRUE_BETA1_C})  [β₂={TRUE_BETA2} for all]")

    x1A, x2A, yA = generate_data(TRUE_BETA1_A, TRUE_BETA2, n, rng)
    x1B, x2B, yB = generate_data(TRUE_BETA1_B, TRUE_BETA2, n, rng)
    x1C, x2C, yC = generate_data(TRUE_BETA1_C, TRUE_BETA2, n, rng)

    bA1, bA2 = bootstrap_betas(x1A, x2A, yA, B, rng)
    bB1, bB2 = bootstrap_betas(x1B, x2B, yB, B, rng)
    bC1, bC2 = bootstrap_betas(x1C, x2C, yC, B, rng)

    eA1, eA2 = fit_ols(x1A, x2A, yA)
    eB1, eB2 = fit_ols(x1B, x2B, yB)
    eC1, eC2 = fit_ols(x1C, x2C, yC)

    an_A1, an_A2   = analytical_se(x1A, x2A)
    an_B1, an_B2   = analytical_se(x1B, x2B)
    an_C1, an_C2   = analytical_se(x1C, x2C)
    ol_A1, ol_A2   = ols_se(x1A, x2A, yA)
    ol_B1, ol_B2   = ols_se(x1B, x2B, yB)
    ol_C1, ol_C2   = ols_se(x1C, x2C, yC)

    # ── Table 1: β₁ ───────────────────────────────────────────────────────
    w = 80
    print()
    print("  β₁ summary")
    print("  " + "─" * w)
    print(f"  {'':12} {'True β₁':>9} {'Est. β̂₁':>9} {'Theoretical SE':>15} {'OLS SE':>8} {'Bootstrap SE':>14}")
    print("  " + "─" * w)
    for nm, tb, est, se_an, se_ol, se_bo in [
        ("Dataset A", TRUE_BETA1_A, eA1, an_A1, ol_A1, bA1.std(ddof=1)),
        ("Dataset B", TRUE_BETA1_B, eB1, an_B1, ol_B1, bB1.std(ddof=1)),
        ("Dataset C", TRUE_BETA1_C, eC1, an_C1, ol_C1, bC1.std(ddof=1)),
    ]:
        print(f"  {nm:<12} {tb:>9.4f} {est:>9.4f} {se_an:>15.4f} {se_ol:>8.4f} {se_bo:>14.4f}")
    print("  " + "─" * w)

    # ── Table 2: β₂ ───────────────────────────────────────────────────────
    print()
    print("  β₂ summary")
    print("  " + "─" * w)
    print(f"  {'':12} {'True β₂':>9} {'Est. β̂₂':>9} {'Theoretical SE':>15} {'OLS SE':>8} {'Bootstrap SE':>14}")
    print("  " + "─" * w)
    for nm, est, se_an, se_ol, se_bo in [
        ("Dataset A", eA2, an_A2, ol_A2, bA2.std(ddof=1)),
        ("Dataset B", eB2, an_B2, ol_B2, bB2.std(ddof=1)),
        ("Dataset C", eC2, an_C2, ol_C2, bC2.std(ddof=1)),
    ]:
        print(f"  {nm:<12} {TRUE_BETA2:>9.4f} {est:>9.4f} {se_an:>15.4f} {se_ol:>8.4f} {se_bo:>14.4f}")
    print("  " + "─" * w)

    # ── Table 3: pairwise tests on β₁ ─────────────────────────────────────
    def _test(bX, bY, eX, eY):
        diff     = eX - eY
        se_d     = np.sqrt(bX.std(ddof=1)**2 + bY.std(ddof=1)**2)
        z        = diff / se_d
        p_z      = 2 * norm.sf(abs(z))
        lo, hi   = np.percentile(bX - bY, [2.5, 97.5])
        sig_ci   = (lo > 0) or (hi < 0)
        rej_z    = "Reject H0"     if p_z < 0.05 else "Fail to reject"
        rej_ci   = "Reject H0"     if sig_ci     else "Fail to reject"
        return diff, z, p_z, lo, hi, rej_z, rej_ci

    print()
    print("  Pairwise tests on β₁")
    print("  " + "─" * w)
    print(f"  {'Comparison':<10} {'Test':<8} {'Statistic':>20} {'p-value':>10} {'Decision (α=0.05)':>16}")
    print("  " + "─" * w)
    for label, bX, bY, eX, eY, true_x, true_y in [
        ("A vs B", bA1, bB1, eA1, eB1, TRUE_BETA1_A, TRUE_BETA1_B),
        ("A vs C", bA1, bC1, eA1, eC1, TRUE_BETA1_A, TRUE_BETA1_C),
    ]:
        diff, z, p_z, lo, hi, rej_z, rej_ci = _test(bX, bY, eX, eY)
        ci_str = f"[{lo:.4f}, {hi:.4f}]"
        print(f"  {label:<10} {'Z-test':<8} {'z = ' + f'{z:.4f}':>20} {p_z:>10.4f} {rej_z:>16}")
        print(f"  {'':10} {'95% CI':<8} {ci_str:>20} {'—':>10} {rej_ci:>16}")
        print(f"  {'':10} (true Δβ₁={true_x-true_y:+.2f},  observed diff={diff:+.4f})")
        print("  " + "·" * w)
    print("  " + "─" * w)


# ── Validation 4: Non-Gaussian noise — z-CI fails, percentile CI holds ────────

def val4_nongaussian(n_small, B, S, rng):
    """
    With non-Gaussian noise β̂ is NOT exactly normal.

    Two failure modes:
      a) Skewed noise (chi²(1)): β̂ is right-skewed.
         z-CI = β̂ ± 1.96·SE is symmetric → wrong coverage.
         Percentile CI [2.5th, 97.5th] adapts to the shape → correct.

      b) Heavy-tailed noise (t(2), infinite kurtosis): bootstrap SE is
         unstable → z = diff/SE has heavier-than-normal tails → FPR inflated.
         Percentile CI of the difference D^b samples the actual distribution
         directly → FPR stays near 0.05.
    """
    print("\n" + "=" * 65)
    print("  Validation 4: Non-Gaussian Noise")
    print("  z-CI assumes β̂ ~ Normal.  Percentile CI makes no assumption.")
    print("=" * 65)

    def gen(beta1, beta2, n, noise_fn):
        x1  = rng.normal(0, 1, n)
        x2  = rng.normal(0, 1, n)
        y   = TRUE_BETA0 + beta1 * x1 + beta2 * x2 + noise_fn(n)
        return x1, x2, y

    # ── Part a: CI coverage — skewed noise ────────────────────────────────
    print(f"\n  Part a — CI coverage (n={n_small}, S={S} datasets, B={B} resamples)")
    print( "  Compare: z-CI (symmetric, assumes normal β̂) vs percentile CI")
    print( "  Noise: Gaussian (reference) vs chi²(1)−1 (skewness = 2√2 ≈ 2.83)")

    w = 54
    print()
    print(f"  {'Noise':<14} {'z-CI cov.':>11} {'Pct CI cov.':>13}  {'z-CI OK?':>10}")
    print("  " + "─" * w)

    for noise_name, noise_fn in [
        ("Gaussian",   lambda n: rng.normal(0, 1, n)),
        ("chi²(1)−1",  lambda n: rng.chisquare(1, n) - 1),
    ]:
        cov_z = cov_pct = 0
        for _ in range(S):
            x1, x2, y = gen(TRUE_BETA1_A, TRUE_BETA2, n_small, noise_fn)
            boot1, _  = bootstrap_betas(x1, x2, y, B, rng)
            est, _    = fit_ols(x1, x2, y)
            se_b      = boot1.std(ddof=1)
            # z-CI: symmetric, assumes normal estimator
            if est - 1.96*se_b <= TRUE_BETA1_A <= est + 1.96*se_b:
                cov_z += 1
            # percentile CI: non-parametric
            lo, hi = np.percentile(boot1, [2.5, 97.5])
            if lo <= TRUE_BETA1_A <= hi:
                cov_pct += 1
        ok = "OK" if 0.90 <= cov_z/S <= 1.00 else "WRONG ← fails"
        print(f"  {noise_name:<14} {cov_z/S:>11.3f} {cov_pct/S:>13.3f}  {ok:>10}")

    print()
    print("  With chi²(1) noise β̂ is right-skewed: the symmetric z-CI")
    print("  cuts off the wrong amount in each tail → undercoverage.")
    print("  Percentile CI adapts to the actual shape → stays near 0.95.")

    # ── Part b: two-sample FPR — heavy-tailed noise ───────────────────────
    Bh = max(B // 2, 50)
    print(f"\n  Part b — Two-sample FPR under H0 (n={n_small}, S={S} pairs, B={Bh} each)")
    print( "  Noise: Gaussian vs t(2) (symmetric but infinite kurtosis)")
    print(f"  H0: β₁_A = β₁_B = {TRUE_BETA1_A}")
    print()
    print(f"  {'Noise':<14} {'FPR z-test':>12} {'FPR CI test':>13}  {'z-test OK?':>12}")
    print("  " + "─" * (w + 4))

    for noise_name, noise_fn in [
        ("Gaussian", lambda n: rng.normal(0, 1, n)),
        ("t(2)",     lambda n: rng.standard_t(2, n)),
    ]:
        fp_z = fp_ci = 0
        for _ in range(S):
            x1A, x2A, yA = gen(TRUE_BETA1_A, TRUE_BETA2, n_small, noise_fn)
            x1B, x2B, yB = gen(TRUE_BETA1_A, TRUE_BETA2, n_small, noise_fn)
            bA1, _ = bootstrap_betas(x1A, x2A, yA, Bh, rng)
            bB1, _ = bootstrap_betas(x1B, x2B, yB, Bh, rng)
            eA, _  = fit_ols(x1A, x2A, yA)
            eB, _  = fit_ols(x1B, x2B, yB)
            se_d   = np.sqrt(bA1.std(ddof=1)**2 + bB1.std(ddof=1)**2)
            z      = (eA - eB) / se_d
            p_z    = 2 * norm.sf(abs(z))
            D      = bA1 - bB1
            lo, hi = np.percentile(D, [2.5, 97.5])
            if p_z < 0.05:           fp_z  += 1
            if (lo > 0) or (hi < 0): fp_ci += 1
        ok = "OK" if abs(fp_z/S - 0.05) < 0.03 else "WRONG ← inflated"
        print(f"  {noise_name:<14} {fp_z/S:>12.3f} {fp_ci/S:>13.3f}  {ok:>12}")

    print()
    print("  With t(2) noise the bootstrap SE estimate itself is unstable")
    print("  (occasional extreme values → inflated SE variance).")
    print("  z = diff/SE inherits heavier-than-normal tails → FPR > 0.05.")
    print("  CI test uses the percentile of D^b directly, not SE → stays near 0.05.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=100, help="Sample size for val 1-3 (default 100)")
    parser.add_argument("--n-small", type=int, default=20,  help="Small n for val 4 non-Gaussian (default 20)")
    parser.add_argument("--B",       type=int, default=1000, help="Bootstrap iterations (default 1000)")
    parser.add_argument("--S",       type=int, default=500,  help="Repetitions for coverage/FPR (default 500)")
    parser.add_argument("--seed",    type=int, default=0)
    args = parser.parse_args()
    rng  = np.random.default_rng(args.seed)

    print(f"\nModel:  y = β₀ + β₁x₁ + β₂x₂ + ε,   ε ~ N(0, σ²={TRUE_SIGMA}²)")
    print(f"True:   β₀={TRUE_BETA0},  β₂={TRUE_BETA2} (shared)")
    print(f"        β₁_A={TRUE_BETA1_A},  β₁_B={TRUE_BETA1_B},  β₁_C={TRUE_BETA1_C}")
    print(f"Setup:  n={args.n},  B={args.B},  S={args.S}")

    val1_se(args.n, args.B, rng)
    val2_coverage(args.n, args.B, args.S, rng)
    val3_two_sample(args.n, args.B, args.S, rng)

    print("\nDone.")


if __name__ == "__main__":
    main()
