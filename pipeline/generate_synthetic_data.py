"""
Generate synthetic regression datasets (CSV) shared between Python and MATLAB
bootstrap validation scripts.

Model:  y = beta0 + beta1*x1 + beta2*x2 + sigma * eps,   eps ~ N(0, 1)

For each dataset, x1 and x2 are drawn first, then sigma is solved so that the
*theoretical* SE(beta1) = sigma * sqrt[(X'X)^-1]_11 hits a target value:

  Dataset A: beta1=2.00, target SE(beta1) = 0.1
  Dataset B: beta1=3.50, target SE(beta1) = 0.2
  Dataset C: beta1=2.05, target SE(beta1) = 0.3

beta0=1.0 and beta2=1.5 are shared across all three datasets.

Outputs (in ./data/):
  dataset_A.csv, dataset_B.csv, dataset_C.csv  -- columns: x1, x2, y
  dataset_info.csv                             -- true parameters per dataset

Usage:
    python generate_synthetic_data.py
"""

import os
import numpy as np
import pandas as pd

TRUE_BETA0 = 1.0
TRUE_BETA2 = 1.5
N    = 500
SEED = 1

DATASETS = {
    "A": {"beta1": 2.00, "se_target": 0.1},
    "B": {"beta1": 3.50, "se_target": 0.2},
    "C": {"beta1": 2.05, "se_target": 0.3},
}


def main():
    rng     = np.random.default_rng(SEED)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)

    meta_rows = []
    for name, params in DATASETS.items():
        beta1     = params["beta1"]
        se_target = params["se_target"]

        x1 = rng.normal(0, 1, N)
        x2 = rng.normal(0, 1, N)

        X       = np.column_stack([np.ones(N), x1, x2])
        XtX_inv = np.linalg.inv(X.T @ X)
        c11     = XtX_inv[1, 1]
        sigma   = se_target / np.sqrt(c11)

        eps = rng.normal(0, 1, N)
        y   = TRUE_BETA0 + beta1 * x1 + TRUE_BETA2 * x2 + sigma * eps

        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        df.to_csv(os.path.join(out_dir, f"dataset_{name}.csv"), index=False)

        meta_rows.append({
            "dataset":         name,
            "n":               N,
            "true_beta0":      TRUE_BETA0,
            "true_beta1":      beta1,
            "true_beta2":      TRUE_BETA2,
            "sigma":           sigma,
            "se_target_beta1": se_target,
        })
        print(f"dataset_{name}.csv  ->  beta1={beta1:.4f}, sigma={sigma:.4f}, "
              f"target SE(beta1)={se_target:.2f}")

    pd.DataFrame(meta_rows).to_csv(os.path.join(out_dir, "dataset_info.csv"), index=False)
    print(f"\nSaved CSVs to {out_dir}")


if __name__ == "__main__":
    main()
