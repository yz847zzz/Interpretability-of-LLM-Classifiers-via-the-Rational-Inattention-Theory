"""
Summarise RI fit parameters across audio noise conditions.

Reads:
    Dataset/results/audio_{noise}/ri_fit/fitted_params.csv

Output:
    Dataset/results/audio_params_summary.csv   (printed + saved)

Usage:
    python summarize_audio_params.py
"""

import os
import pandas as pd

RESULTS_DIR = "./Dataset/results"
NOISE_TYPES = ["white", "babble", "cafe"]

MODEL_ROWS = {
    "GPT-3.5-turbo":         "lambda_GPT-3.5-turbo",
    "GPT-5.4-nano":          "lambda_GPT-5.4-nano",
    "Gemini-2.5-Flash":      "lambda_Gemini-2.5-Flash",
    "Gemini-2.5-Flash-Lite": "lambda_Gemini-2.5-Flash-Lite",
}

summary = {}

for noise in NOISE_TYPES:
    path = os.path.join(RESULTS_DIR, f"audio_{noise}", "ri_fit", "fitted_params.csv")
    if not os.path.exists(path):
        print(f"  Missing: {path} — skipping {noise}")
        continue

    df = pd.read_csv(path)
    col = {}

    for _, row in df.iterrows():
        label = row["model"]
        if label in MODEL_ROWS:
            col[MODEL_ROWS[label]] = round(float(row["lambda"]), 6)

    # shared params (same for all rows)
    col["alpha"] = round(float(df["alpha"].iloc[0]), 6)
    col["beta"]  = round(float(df["beta"].iloc[0]),  6)

    summary[noise] = col

out = pd.DataFrame(summary)
out.index.name = "parameter"

# reorder rows
row_order = list(MODEL_ROWS.values()) + ["alpha", "beta"]
out = out.reindex([r for r in row_order if r in out.index])

print("\n" + "="*60)
print("  RI Fit Parameters — Audio Noise Conditions")
print("="*60)
print(out.to_string())

save_path = os.path.join(RESULTS_DIR, "audio_params_summary.csv")
out.to_csv(save_path)
print(f"\nSaved -> {save_path}")
