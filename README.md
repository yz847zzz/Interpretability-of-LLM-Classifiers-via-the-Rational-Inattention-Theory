# LLM Classification Performance Under Text Noise

This repository contains code to reproduce the experiments in our paper on modelling LLM response behaviour under degraded (noisy) input conditions using Response Inhibition (RI) theory.

The pipeline classifies hate-speech texts at 11 noise levels (p = 0.0 to 1.0) using GPT and/or Gemini, then computes sensitivity (P1a), specificity (P1b), overall accuracy (Pc), and mutual information I(Y;A).

---

## Requirements

```bash
pip install datasets pandas numpy openai google-genai python-dotenv tqdm
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
```

`.env` is listed in `.gitignore` and will never be committed.

---

## Scripts

| Script | Description |
|---|---|
| `download_data.py` | Downloads the dataset from Hugging Face and creates a balanced 400-sample |
| `add_noise.py` | Generates 11 noisy versions of the sample at noise levels p = 0.0 to 1.0 |
| `get_llm_responses.py` | Sends each noisy dataset to GPT and/or Gemini for binary hate-speech classification |
| `compute_metrics.py` | Aggregates predictions into P1a, P1b, Pc, and I(Y;A) across noise levels |

---

## How to Run

Run the scripts in order. Each script has a **CONFIG** block at the top — adjust paths and model names there before running.

### 1. Download the dataset

```bash
python download_data.py
```

Downloads `ucberkeley-dlab/measuring-hate-speech` from Hugging Face and samples 200 clearly-benign texts (score < −3, label=0) and 200 clearly-hateful texts (score > 3, label=1), keeping only unique texts.

**Output:**
```
Dataset/hate_speech_binary.csv       # full dataset (~136k rows)
Dataset/hate_speech_binary_400.csv   # balanced 400-sample  (label 0 = no hate, label 1 = hate)
```

---

### 2. Generate noisy datasets

```bash
python add_noise.py
```

For each noise level p ∈ {0.0, 0.1, …, 1.0}, every word token is independently perturbed with probability p. Perturbations are a weighted mixture of four character-level operations: swap, insert, replace, and delete. All randomness is seeded, so output is fully reproducible.

**Output:**
```
Dataset/noisy/hate_speech_400_p_00.csv   # p = 0.0  (clean)
Dataset/noisy/hate_speech_400_p_01.csv   # p = 0.1
...
Dataset/noisy/hate_speech_400_p_10.csv   # p = 1.0  (maximum noise)
```

---

### 3. Get LLM responses

```bash
python get_llm_responses.py --model gpt
python get_llm_responses.py --model gemini
python get_llm_responses.py --model both   # run both at once
```

Each noisy file is sent to the chosen LLM in batches of 50. Failed batches are recursively split in half down to individual items. Already-processed files are skipped, so it is safe to resume after interruption.

Configure which models to use in the **CONFIG** block of `get_llm_responses.py`:

```python
GPT_MODEL    = "gpt-3.5-turbo"    # or "gpt-4o", "gpt-4.1", etc.
GEMINI_MODEL = "gemini-2.5-flash"  # or "gemini-2.0-flash", etc.
```

**Output:**
```
Dataset/results/gpt_gpt_3_5_turbo/hate_speech_400_p_00.csv    # + column pred_gpt_*
...
Dataset/results/gemini_gemini_2_5_flash/hate_speech_400_p_00.csv  # + column pred_gemini_*
```

---

### 4. Compute summary statistics

```bash
python compute_metrics.py --model gpt
python compute_metrics.py --model gemini
python compute_metrics.py --model both
```

Reads the classified CSVs and computes the following metrics for each noise level:

| Metric | Definition |
|---|---|
| **P1a** | Sensitivity — P(pred = 1 \| label = 1) |
| **P1b** | Specificity — P(pred = 0 \| label = 0) |
| **Pc** | Accuracy — P(Y=1)·P1a + P(Y=0)·P1b |
| **I(Y;A)** | Mutual information between ground truth and prediction (nats) |

Results are printed to the terminal and saved as a CSV.

**Output:**
```
Dataset/results/gpt_gpt_3_5_turbo/summary_gpt_gpt_3_5_turbo.csv
Dataset/results/gemini_gemini_2_5_flash/summary_gemini_gemini_2_5_flash.csv
```

Example terminal output:
```
noise_p      P1a      P1b       Pc    I(Y;A)
    0.0   0.9600   0.9650   0.9625  0.533305
    0.1   0.9050   0.8750   0.8900  0.347334
    ...
    1.0   0.4800   0.5200   0.5000  0.000000
```

---

## Output folder structure

```
Dataset/
├── hate_speech_binary.csv             # full dataset
├── hate_speech_binary_400.csv         # balanced 400-sample
├── noisy/
│   ├── hate_speech_400_p_00.csv       # clean
│   ├── hate_speech_400_p_01.csv
│   └── ...
└── results/
    ├── gpt_gpt_3_5_turbo/
    │   ├── hate_speech_400_p_00.csv
    │   ├── ...
    │   └── summary_gpt_gpt_3_5_turbo.csv
    └── gemini_gemini_2_5_flash/
        ├── hate_speech_400_p_00.csv
        ├── ...
        └── summary_gemini_gemini_2_5_flash.csv
```

---

## Dataset

The base dataset is [ucberkeley-dlab/measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) (Kennedy et al., 2020), available from Hugging Face. No redistribution of the raw data is included in this repository.
