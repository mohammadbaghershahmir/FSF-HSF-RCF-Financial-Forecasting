## Adaptive LLM-Based Forecasting Frameworks for Bitcoin & Gold

This repository contains three related research frameworks for forecasting the daily **direction** of Bitcoin (BTC) and Gold (XAU) prices using large language models (LLMs) and multi‑source market data:

- **FSF**: Fusion Single-Agent Framework  
- **HSF**: Hierarchical Specialized-Agent Framework  
- **RCF**: Reflective Consensus Framework

All three frameworks share the same high‑level training procedure:

1. **Supervised Fine‑Tuning (SFT)** on high‑confidence model outputs that match the ground truth.
2. **Preference-based fine‑tuning with DPO** (Direct Preference Optimization), using pairs of preferred vs. rejected answers generated through reflection.

They differ mainly in **how they structure agents and produce preference data**, not in the core optimization logic.

---

## 1. Data & Information Sources (IS Module)

Each framework consumes the same preprocessed data, produced by an upstream **Information Sources (IS)** module. This module is not implemented here, but its outputs are provided as files under each project’s `data/` directory.

For each trading day t, the IS module generates:

- **Numerical price features**
  - `x_candle_BTC,t` from BTC OHLCV data
  - `x_candle_XAU,t` from XAU (gold) OHLCV data
- **Textual summaries**
  - `s_sum_BTC,t`: news / social media summaries for BTC
  - `s_sum_XAU,t`: news / social media summaries for XAU
- **Sentiment distributions**
  - `π_sent_BTC,t`: sentiment probabilities for BTC
  - `π_sent_XAU,t`: sentiment probabilities for XAU

These are stored in:

- `FSF/data/…`
- `HSF/data/…`
- `RCF/data/…`

The raw tweet and sentiment samples (used to create the above features) are under `data/sample_tweet/raw/` and `data/sample_price/preprocessed/` in each project.

---

## 2. Shared Training Procedure (SFT + DPO)

All three frameworks use the same two‑stage learning strategy:

- **Stage 1 – Supervised Fine‑Tuning (SFT)**  
  A reference model pi_ref is trained on a supervised dataset  built from high‑confidence predictions that match the true label g_t.  
  The loss is the usual language‑model cross‑entropy over the target explanation sequence.

- **Stage 2 – Preference-based DPO Fine‑Tuning**  
  When the model’s initial answer is wrong or has low confidence, a **reflection cycle** is triggered. The model analyzes its own mistake and produces a corrected answer; we store a **preferred** answer y_w and a **less preferred** answer y_l, building a preference dataset.  
  The model is then trained with the **DPO loss**, which directly optimizes the log‑odds between pi_theta and pi_ref over (y_w, y_l).

All implementations use the Tiny‑Vicuna model (`Jiayi-Pan/Tiny-Vicuna-1B`) by default, with LoRA adapters and optional DeepSpeed for scalable training.

---

## 3. FSF – Fusion Single-Agent Framework

**Folder**: `FSF/`  
**Main entry point**: `python main.py`

### 3.1 Concept

FSF uses a **single adaptive LLM agent** that directly consumes all available information sources for both markets in a unified prompt. For each trading day t, the input vector x_t concatenates:

- BTC and XAU candlestick features
- BTC and XAU textual summaries
- BTC and XAU sentiment distributions

The agent produces a **two‑part output**:

1. A **directional prediction** (e.g., Positive / Negative price movement)  
2. A **natural‑language explanation** justifying the prediction

### 3.2 Reflection & Preference Generation

- If the initial answer matches the true label and its probability exceeds a threshold tau, the pair (x_t, y_tilde) is added to the supervised dataset.
- Otherwise, the model enters a **self‑reflection cycle**:
  - It analyzes why its previous prediction may be wrong.
  - It generates a corrected answer and an improved explanation.
  - The original answer and the corrected answer are stored as a **preference pair** \((y^w, y^l)\) in \(\mathcal{D}_\text{Pref}\).

This framework is the most compact: a **single, generalist agent** learns to jointly integrate numeric, textual, and sentiment information for both assets.

### 3.3 How to Run

From the repository root:

```bash
cd FSF
python main.py \
  --price_dir data/sample_price/preprocessed/ \
  --tweet_dir data/sample_tweet/raw/ \
  --data_path ./data/merge_sample.json
```

Key arguments in `main.py`:

- `--model_path`: base LLM (default `Jiayi-Pan/Tiny-Vicuna-1B`)
- `--data_path`: JSON file with merged IS features and labels
- `--datasets_dir`: where reflection‑generated datasets are stored
- `--rl_base_model`: base checkpoint for DPO training

By default, the training line (`exp_model.train()`) is commented; you can uncomment it to enable full training before testing.

---

## 4. HSF – Hierarchical Specialized-Agent Framework

**Folder**: `HSF/`  
**Main entry point**: `python main.py`

### 4.1 Concept

HSF introduces a **hierarchical architecture with two specialized LLM agents**, each focusing on a different modality:

- **Numerical & sentiment agent**  
  - Input: price features `x_candle` + sentiment distributions `pi_sent`  
  - Task: predict direction and explain based mainly on quantitative and sentiment signals.

- **Textual & sentiment agent**  
  - Input: textual summaries `s_sum` + sentiment distributions `pi_sent`  
  - Task: predict direction and explain based on news / textual signals.

The outputs of these two agents are then **combined at a higher level**:

- If both agents agree on the direction and pass quality checks, their explanations are **merged** into a final, more comprehensive explanation.
- If only one agent is incorrect, **only that agent** enters a reflection cycle to generate preference data; the other agent’s output is kept as the trusted reference.

This design allows you to **separately analyze the contribution of numeric vs. textual information** to the final trading decision.

### 4.2 Reflection & Training

The reflection logic is similar to FSF but applied **per agent**:

- Correct, high‑confidence agreements feed the supervised dataset.
- Errors trigger agent‑specific reflection and populate the preference dataset.
- Training again follows SFT → DPO, as described in Section 2.

### 4.3 How to Run

From the repository root:

```bash
cd HSF
python main.py \
  --price_dir data/sample_price/preprocessed/ \
  --tweet_dir data/sample_tweet/raw/ \
  --data_path ./data/merge_sample.json
```

Some additional utilities in `HSF/utils/` (e.g., `enhanced_prompts.py`, `evaluation_metrics.py`) provide:

- More structured prompts for each specialized agent
- Improved evaluation metrics to analyze disagreements and explanation quality

As in FSF, training (`exp_model.train()`) is currently commented out and can be enabled as needed.

---

## 5. RCF – Reflective Consensus Framework

**Folder**: `RCF/`  
**Main entry points**:

- Single‑agent DPO pipeline: `python main.py`  
- Multi‑model ensemble with consensus: `python main_multi_model.py`

### 5.1 Concept

RCF is the most advanced framework, designed to:

- Increase **prediction stability**
- Reduce **prompt‑induced bias**
- Improve **explanation quality** via multi‑step reflection

It uses an **ensemble of identical LLMs** that share weights but receive **different prompts**, encouraging diversity in reasoning. For each trading day:

1. Multiple models run **in parallel** on the same IS input x_t.
2. Each model outputs:
   - A **direction label** (Positive / Negative)
   - A **textual explanation**
3. The final label is determined by **majority vote** across the models.

### 5.2 Consensus, Reflection, and Datasets

- If the **majority vote** label matches the ground truth g_t, the explanations of the agreeing models are **combined** and stored in the supervised dataset.
- If the vote is **incorrect**, all mis‑predicting models enter a **two‑stage reflection process**:
  1. **Solution reflection** – each model analyzes its own previous answer, identifies the most likely sources of error, and produces a refined prediction and explanation.
  2. **Combined reflection** – the various reflected explanations (keywords, evidence, reasoning paths) are merged into a more coherent, higher‑quality explanation.

From these reflections, RCF constructs preference pairs (y_w, y_l) and trains the ensemble using the same DPO loss as in FSF/HSF.

### 5.3 How to Run

From the repository root:

```bash
cd RCF

# Single-agent DPO-style pipeline
python main.py \
  --price_dir data/sample_price/preprocessed/ \
  --tweet_dir data/sample_tweet/raw/ \
  --data_path ./data/merge_sample.json

# Multi-model reflective consensus (default: 3 models)
python main_multi_model.py \
  --price_dir data/sample_price/preprocessed/ \
  --tweet_dir data/sample_tweet/raw/ \
  --data_path ./data/merge_sample.json \
  --num_models 3
```

In `main_multi_model.py`, if you pass an even `--num_models`, it is automatically rounded up to the next odd number to keep majority voting well‑defined.

---

## 6. Environment & Dependencies

The repository includes a `shahmirenv.yml` file that can be used to reproduce the original conda environment:

```bash
conda env create -f shahmirenv.yml
conda activate shahmirenv
```

All three frameworks assume:

- Python 3.8+  
- PyTorch (GPU strongly recommended)  
- Hugging Face Transformers / Accelerate  
- Optional: DeepSpeed for large‑scale DPO training  
- Optional: Weights & Biases (`wandb`) for experiment tracking

You may need to configure your Hugging Face cache, API access (if using hosted models), and GPU settings according to your system.

---

## 7. Repository Structure

At a high level, the repository is organized as:

- `FSF/` – Flexible Single-Agent Framework
- `HSF/` – Hierarchical Specialized-Agent Framework
- `RCF/` – Reflective Consensus Framework
- `shahmirenv.yml` – Conda environment definition

Each framework has a similar internal layout:

- `data/` – Preprocessed sample data (prices, tweets, sentiments)
- `data_load/` – Data loaders
- `datasets/` – Generated SFT / preference datasets
- `exp/` – Experiment orchestration (`Exp_Model`, `Exp_MultiModel`, etc.)
- `explain_module/` – Agent definitions and reflection logic
- `predict_module/` – Fine‑tuning and DPO training utilities
- `summarize_module/` – Text summarization utilities
- `utils/` – LLM helpers, prompts, and evaluation metrics

---

## 8. Using This Repository

- **Researchers** can use FSF, HSF, and RCF as reference implementations for:
  - LLM‑based financial forecasting
  - Self‑reflection and preference‑based training (DPO)
  - Multi‑agent and ensemble architectures for robust prediction
- **Practitioners** can adapt the code to other assets or domains by:
  - Replacing the IS module outputs in `data/`
  - Adjusting prompts and reflection logic in `utils/` and `explain_module/`

If you use this code or the associated architectures in academic work, please consider citing the underlying paper and the DPO and self‑reflection references mentioned in the thesis.


