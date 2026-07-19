"""
Reconstruct row-level (label, pred) datapoints from the aggregated counts in
fit_Pcorrect_joint_GPT52_GPT35_Gemini_Gemini20_shared_alpha_beta.m
("set2" reconstructed counts for GPT-5.2 / GPT-3.5-turbo / Gemini-2.5 / Gemini-2.0).

Convention (matches the MATLAB script and compute_pc in fit_text_noise.py):
  label = 0  -> true state Y1 (correct action = Aa)
  label = 1  -> true state Y2 (correct action = Ab)
  pred  = 0  -> model predicted action Aa
  pred  = 1  -> model predicted action Ab

For each p' level (11 levels, 0.0-1.0) and each model:
  - 200 rows with label=0: n_Y1_Aa of them get pred=0 (correct), rest pred=1
  - 200 rows with label=1: n_Y2_Ab of them get pred=1 (correct), rest pred=0

This reproduces  Pc = (n_Y1_Aa + n_Y2_Ab) / 400  exactly, matching computePc()
in the MATLAB script.

Output:
  data/text_noise_set2.csv  -- long format: model, p_prime, trial_id, label, pred

Usage:
    python reconstruct_text_noise_set2.py
"""

import os
import numpy as np
import pandas as pd

P_PRIME = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NY1 = 200
NY2 = 200

# ── Reconstructed counts (set2) from the MATLAB script ──────────────────────
COUNTS = {
    "GPT-5.2": dict(
        n_Y1_Aa=[190, 190, 190, 190, 187, 184, 187, 178, 183, 176, 183],
        n_Y2_Ab=[178, 178, 179, 176, 182, 180, 177, 167, 159, 156, 146],
    ),
    "GPT-3.5-turbo": dict(
        n_Y1_Aa=[158, 149, 162, 164, 155, 150, 154, 154, 157, 156, 183],
        n_Y2_Ab=[196, 196, 194, 196, 196, 178, 187, 189, 180, 171, 148],
    ),
    "Gemini-2.5": dict(
        n_Y1_Aa=[190, 183, 190, 194, 185, 194, 193, 193, 193, 187, 196],
        n_Y2_Ab=[179, 159, 167, 156, 155, 146, 162, 142, 110, 109, 82],
    ),
    "Gemini-2.0": dict(
        n_Y1_Aa=[165, 159, 160, 161, 157, 152, 157, 146, 140, 136, 144],
        n_Y2_Ab=[179, 179, 179, 179, 177, 176, 179, 170, 162, 154, 138],
    ),
}


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for model, c in COUNTS.items():
        n_Y1_Aa = c["n_Y1_Aa"]
        n_Y2_Ab = c["n_Y2_Ab"]
        assert len(n_Y1_Aa) == len(P_PRIME) == len(n_Y2_Ab)

        for i, p in enumerate(P_PRIME):
            a1 = n_Y1_Aa[i]   # correct Y1 trials (pred=0)
            a2 = n_Y2_Ab[i]   # correct Y2 trials (pred=1)
            assert 0 <= a1 <= NY1
            assert 0 <= a2 <= NY2

            label = [0] * NY1 + [1] * NY2
            pred  = [0] * a1 + [1] * (NY1 - a1) + [1] * a2 + [0] * (NY2 - a2)

            for trial_id, (lb, pr) in enumerate(zip(label, pred)):
                rows.append({
                    "model":    model,
                    "p_prime":  p,
                    "trial_id": trial_id,
                    "label":    lb,
                    "pred":     pr,
                })

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "text_noise_set2.csv")
    df.to_csv(out_path, index=False)

    # ── Sanity check: recompute Pc and compare to (n_Y1_Aa+n_Y2_Ab)/400 ─────
    print("Sanity check (Pc = (n_Y1_Aa + n_Y2_Ab)/400):")
    for model, c in COUNTS.items():
        sub = df[df["model"] == model]
        for i, p in enumerate(P_PRIME):
            g = sub[sub["p_prime"] == p]
            t, pr = g["label"].values, g["pred"].values
            p_s2, p_s1 = t.mean(), 1 - t.mean()
            p2b = pr[t == 1].mean()
            p1a = 1 - pr[t == 0].mean()
            pc_recon  = p_s2 * p2b + p_s1 * p1a
            pc_target = (c["n_Y1_Aa"][i] + c["n_Y2_Ab"][i]) / 400
            assert abs(pc_recon - pc_target) < 1e-9, (model, p, pc_recon, pc_target)
        print(f"  {model:<14} OK ({len(P_PRIME)} levels, {len(sub)} rows)")

    print(f"\nSaved {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
