"""
Download and sample the hate-speech dataset from Hugging Face.

Source  : ucberkeley-dlab/measuring-hate-speech
Output  : Dataset/hate_speech_binary.csv          (full dataset, ~136k rows)
          Dataset/hate_speech_binary_400.csv       (balanced 400-sample)

The 400-sample is formed by taking 200 texts with hate_speech_score < -3
(clearly no hate speech, label=0) and 200 texts with score > 3
(clearly hate speech, label=1), keeping only unique texts.
"""

import os
import pandas as pd
from datasets import load_dataset

# ==============================================================================
# CONFIG
# ==============================================================================
OUTPUT_DIR  = "./Dataset"
SAMPLE_SIZE = 200     # samples per class  (total = 2 * SAMPLE_SIZE = 400)
SCORE_LOW   = -3      # upper bound for "clearly benign" texts  (score < -3)
SCORE_HIGH  =  3      # lower bound for "clearly hateful" texts (score >  3)
SEED        = 42
# ==============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading ucberkeley-dlab/measuring-hate-speech ...")
    dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
    df = dataset["train"].to_pandas()[["text", "hate_speech_score"]].copy()
    df["label"] = (df["hate_speech_score"] > 0.5).astype(int)

    full_path = os.path.join(OUTPUT_DIR, "hate_speech_binary.csv")
    df.to_csv(full_path, index=False)
    print(f"Full dataset saved  ({len(df):,} rows)  ->  {full_path}")

    # Deduplicate: keep the row with the most extreme score per unique text
    df_unique = (
        df.dropna(subset=["text", "hate_speech_score"])
          .assign(abs_score=lambda x: x["hate_speech_score"].abs())
          .sort_values("abs_score", ascending=False)
          .drop_duplicates(subset=["text"], keep="first")
          .drop(columns=["abs_score"])
    )

    pool_low  = df_unique[df_unique["hate_speech_score"] < SCORE_LOW]
    pool_high = df_unique[df_unique["hate_speech_score"] > SCORE_HIGH]

    if len(pool_low) < SAMPLE_SIZE:
        raise ValueError(f"Not enough unique texts with score < {SCORE_LOW}: {len(pool_low)}")
    if len(pool_high) < SAMPLE_SIZE:
        raise ValueError(f"Not enough unique texts with score > {SCORE_HIGH}: {len(pool_high)}")

    sample = pd.concat(
        [pool_low.sample(n=SAMPLE_SIZE, random_state=SEED),
         pool_high.sample(n=SAMPLE_SIZE, random_state=SEED)],
        ignore_index=True,
    ).sort_values(by=["label", "hate_speech_score"], ascending=[False, False]).reset_index(drop=True)

    assert sample["text"].is_unique, "Sampled texts are not unique — unexpected."

    sample_path = os.path.join(OUTPUT_DIR, "hate_speech_binary_400.csv")
    sample.to_csv(sample_path, index=False)
    print(f"Balanced sample saved ({len(sample)} rows)  ->  {sample_path}")
    print(sample["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
