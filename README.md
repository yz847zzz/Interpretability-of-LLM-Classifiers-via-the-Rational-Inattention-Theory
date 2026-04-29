# LLM Classification Performance Under Text Noise

This repository provides the data generation and evaluation pipeline for:

> Y. Zhao, A. Abdi, "Interpretability of LLM Classifiers via the Rational Inattention Theory with Application to Hate Speech Detection," *ACL Student Research Workshop*, 2026.

The pipeline classifies hate-speech texts at 11 noise levels (p′ = 0.0 to 1.0) using GPT and/or Gemini, computes empirical statistics (P1a, P1b, Pc, I(Y;A)), and fits the extended Rational Inattention (RI) model to estimate the interpretability parameters x = r/λ (reward-to-cost ratio) per LLM and the shared noise-mapping parameters α, β.

---

## Requirements

```bash
pip install datasets pandas numpy scipy matplotlib openai google-genai python-dotenv tqdm
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
| `add_noise.py` | Generates 11 noisy versions of the sample at noise levels p′ = 0.0 to 1.0 |
| `get_llm_responses.py` | Sends each noisy dataset to GPT and/or Gemini for binary hate-speech classification |
| `compute_metrics.py` | Computes P1a, P1b, Pc, and I(Y;A) across noise levels |
| `fit_ri_model.py` | Runs the NIAS test and fits the extended RI model to estimate x = r/λ and λ per LLM |

---

## How to Run

Run the scripts in order. Each script has a **CONFIG** block at the top — adjust paths and model names there before running.

### 1. Download the dataset

```bash
python download_data.py
```

Downloads `ucberkeley-dlab/measuring-hate-speech` from Hugging Face and samples 200 clearly-benign texts (hate\_speech\_score < −3, label = 0) and 200 clearly-hateful texts (score > 3, label = 1), keeping only unique texts.

**Output:**
```
Dataset/hate_speech_binary.csv       # full dataset (~136k rows)
Dataset/hate_speech_binary_400.csv   # balanced 400-sample
```

---

### 2. Generate noisy datasets

```bash
python add_noise.py
```

For each noise level p′ ∈ {0.0, 0.1, …, 1.0}, every word token is independently perturbed with probability p′. Perturbations are a weighted mixture of four character-level operations: swap, insert, replace, and delete (Table 2 in the paper). All randomness is seeded, so output is fully reproducible.

**Output:**
```
Dataset/noisy/hate_speech_400_p_00.csv   # p′ = 0.0  (clean)
Dataset/noisy/hate_speech_400_p_01.csv   # p′ = 0.1
...
Dataset/noisy/hate_speech_400_p_10.csv   # p′ = 1.0  (maximum noise)
```

---

### 3. Get LLM responses

```bash
python get_llm_responses.py --model gpt
python get_llm_responses.py --model gemini
python get_llm_responses.py --model both   # run both at once
```

Each noisy file is sent to the chosen LLM in batches of 50. Failed batches are recursively split in half down to individual items. Already-processed files are skipped, so it is safe to resume after interruption.

Configure the models in the **CONFIG** block of `get_llm_responses.py`:

```python
GPT_MODEL    = "gpt-3.5-turbo"    # or "gpt-4o", "gpt-5.2", etc.
GEMINI_MODEL = "gemini-2.5-flash"  # or "gemini-2.0-flash", etc.
```

**Output:**
```
Dataset/results/gpt_gpt_3_5_turbo/hate_speech_400_p_00.csv       # + column pred_gpt_*
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

Computes the following metrics for each noise level (see Section 4.1 of the paper):

| Metric | Definition |
|---|---|
| **P1a** | Sensitivity — P(pred = 1 \| label = 1) |
| **P1b** | Specificity — P(pred = 0 \| label = 0) |
| **Pc** | Accuracy — P(Y=1)·P1a + P(Y=0)·P1b |
| **I(Y;A)** | Mutual information between ground truth and prediction (nats) |

**Output:**
```
Dataset/results/gpt_gpt_3_5_turbo/summary_gpt_gpt_3_5_turbo.csv
Dataset/results/gemini_gemini_2_5_flash/summary_gemini_gemini_2_5_flash.csv
```

---

### 5. Fit the RI model and run NIAS test

```bash
python fit_ri_model.py
```

Implements Sections 4.2 and 4.3 of the paper.

**NIAS test (Eq. 9):** verifies that each LLM's decision strategy is consistent with rational inattention by checking the No Improving Action Switches condition across all noise environments:

$$P(A=a \mid Y=1) \geq \frac{P(A=a \mid Y=2) + 2}{3}$$

**RI model fitting (Eq. 12):** estimates the following parameters by minimising the joint SSE between the model's predicted Pc and each LLM's observed Pc curve:

| Parameter | Meaning | Scope |
|---|---|---|
| **x = r/λ** | Reward-to-cost ratio | Per LLM |
| **α** | Scale of noise mapping q(p′) = α·p′^β | Shared |
| **β** | Shape of noise mapping | Shared |

After fitting, the unit information cost is recovered as **λ = 1/x** (under the assumption that the reward r = 1 is identical across all LLMs given the same prompt).

Configure which models to include in the **CONFIG** block of `fit_ri_model.py`:

```python
MODELS = [
    ("GPT-3.5",  "gpt_gpt_3_5_turbo"),
    ("Gemini",   "gemini_gemini_2_5_flash"),
]
```

**Output:**
```
Dataset/results/ri_fit/fitted_params.csv   # x, lambda, alpha, beta, R² per model
Dataset/results/ri_fit/nias_test.csv       # NIAS condition and p-value per noise level
Dataset/results/ri_fit/<model>_fit.png     # observed vs fitted Pc curves
Dataset/results/ri_fit/all_models_fit.png  # all models overlaid
```

---

## Output folder structure

```
Dataset/
├── hate_speech_binary.csv              # full dataset
├── hate_speech_binary_400.csv          # balanced 400-sample
├── noisy/
│   ├── hate_speech_400_p_00.csv        # clean
│   ├── hate_speech_400_p_01.csv
│   └── ...
└── results/
    ├── gpt_gpt_3_5_turbo/
    │   ├── hate_speech_400_p_00.csv
    │   ├── ...
    │   └── summary_gpt_gpt_3_5_turbo.csv
    ├── gemini_gemini_2_5_flash/
    │   ├── hate_speech_400_p_00.csv
    │   ├── ...
    │   └── summary_gemini_gemini_2_5_flash.csv
    └── ri_fit/
        ├── fitted_params.csv
        ├── nias_test.csv
        ├── GPT35_fit.png
        ├── Gemini_fit.png
        └── all_models_fit.png
```

---

## Dataset

The base dataset is [ucberkeley-dlab/measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) (Kennedy et al., 2020), available from Hugging Face. No redistribution of the raw data is included in this repository.
