# Indonesian TikTok Comments Sentiment Analysis: NLP with Classical Machine Learning

**FINAL PROJECT – NATURAL LANGUAGE PROCESSING**  
**Group:** 19  
**BINUS University**  
**Academic Year:** 2025/2026  
**Faculty / Department:** School of Computer Science / Computer Science Department  
**Course Code:** COMP6885001 / NATURAL LANGUAGE PROCESSING  

---

## 1. Problem Statement & Motivation

Social media platforms like TikTok generate massive volumes of user-generated content daily. Understanding public sentiment in these comments is valuable for content creators, brands, and researchers. This project builds an end-to-end NLP pipeline for multi-class sentiment classification (positive / negative / neutral) on real-world Indonesian TikTok comment data.

The pipeline applies classical machine learning models with TF-IDF feature extraction, benchmarked against IndoBERT — a transformer-based model pre-trained on Indonesian text — to demonstrate the performance gap between traditional and deep learning approaches.

---

## 2. Dataset Description

The project uses a real-world Indonesian TikTok comment dataset sourced from Kaggle.

- **Format:** Two CSV files — a training set and a test set
- **Schema:** Each row contains three columns: `content` (raw comment text), `score` (star rating), and `sentiment` (ground-truth label: `positive`, `negative`, or `neutral`)
- **Important:** The `score` column is **excluded from all model training** because it perfectly correlates with the `sentiment` label, which would constitute data leakage. It is only used during EDA.
- **Language:** Indonesian, including informal slang, typos, emojis, and hashtags typical of TikTok comments

---

## 3. Machine Learning Problem Formulation

Sentiment classification is treated as a supervised multi-class classification problem. Given a raw comment string, the model predicts one of three sentiment labels. **Macro F1** is the primary evaluation metric, as it treats each class equally regardless of class imbalance.

---

## 4. NLP Pipeline Overview

The notebook is organized into the following stages:

| Step | Description |
|------|-------------|
| 1 | Install & import libraries |
| 2 | Load train/test datasets from Kaggle input |
| 3 | Schema validation & data quality checks |
| 4 | Exploratory Data Analysis (label distribution, text length) |
| 5 | Automated profiling report via `ydata-profiling` |
| 6 | Train-test phrase overlap check (n-gram leakage audit) |
| 7 | Text cleaning with a custom sklearn-compatible `TextCleaner` transformer |
| 8 | Train-validation split (80/20 stratified) |
| 9 | Baseline model training (4 classical ML pipelines) |
| 10 | Hyperparameter tuning via `GridSearchCV` (5-fold stratified CV) |
| 11 | Overfitting check (train vs. validation gap analysis) |
| 12 | Final test set evaluation (run once, after model selection) |
| 13 | Error analysis on misclassified samples |
| 14 | Top discriminative features per class (TF-IDF coefficients) |
| 15 | Per-model comparison (Logistic Regression deep-dive) |
| 16 | Save best model to disk with `joblib` |
| 17 | Inference on new text + Hate Speech Ensemble Override |
| 18 | Data leakage audit |
| 19 | IndoBERT fine-tuning (benchmark, GPU T4 required) |

---

## 5. Text Cleaning

A custom `TextCleaner` class (sklearn `BaseEstimator` + `TransformerMixin`) handles all preprocessing inline within the sklearn `Pipeline`. Cleaning steps:

1. **HTML unescape** — decode entities like `&amp;`
2. **Lowercase** — normalize casing
3. **Remove URLs, mentions, and hashtags**
4. **Emoji → sentiment token** — preserves sentiment signal (e.g., 😢 → `EMOJI_SAD`)
5. **Repeated punctuation → token** — `!!` or `?!` → `EXCL`
6. **Repeated character normalization** — reduces informal typos (e.g., `bangeet` → `banget`)
7. **Remove remaining non-alphanumeric characters** and normalize whitespace

---

## 6. Model Selection & Rationale

> Only Classical Machine Learning models are used for primary evaluation. IndoBERT is included as a deep learning benchmark.

**1. Logistic Regression** (`sklearn.linear_model.LogisticRegression`)  
Probabilistic outputs, interpretable coefficients, regularization via `C` parameter. Performs well in high-dimensional TF-IDF spaces.

**2. SVM — Linear Kernel** (`sklearn.svm.SVC`)  
Strong generalization in high-dimensional spaces. Effective when feature count exceeds sample count. `class_weight='balanced'` handles label imbalance.

**3. Naive Bayes** (`sklearn.naive_bayes.MultinomialNB`)  
Computationally efficient baseline for discrete TF-IDF features. Well-suited for short, noisy social media text.

**4. Random Forest** (`sklearn.ensemble.RandomForestClassifier`)  
Ensemble approach that handles non-linear patterns. Provides feature importance scores for interpretability.

**5. IndoBERT (Benchmark)** — `indobenchmark/indobert-base-p1` via Hugging Face Transformers  
Fine-tuned on a stratified subset for practical runtime on Kaggle. Uses minimal BERT-specific preprocessing (URLs → `URL`, mentions → `USER`). Requires a **GPU T4 runtime**.

---

## 7. TF-IDF Configuration (Shared Across Classical Models)

```python
TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),   # unigrams and bigrams
    min_df=3,
    max_df=0.90,
    max_features=25_000,
    sublinear_tf=True,
)
```

---

## 8. Environment & Setup

### A. Prerequisites

- Python 3.8+
- A **Kaggle** notebook environment (recommended) — dataset is loaded from `/kaggle/input/`
- For IndoBERT: **GPU T4 runtime** (Kaggle Accelerator setting)

### B. Dataset Structure (Kaggle Input)

The notebook auto-detects CSV files under `/kaggle/input/`. Ensure your Kaggle dataset contains:

```
/kaggle/input/
└── <your-dataset>/
    ├── tiktok_train.csv
    └── tiktok_test.csv
```
Each CSV must include the columns: `content`, `score`, `sentiment`.

### C. Install Dependencies

The notebook installs all required packages at the top:

```bash
# Core NLP / ML
pip install ydata-profiling scikit-learn pandas numpy matplotlib joblib

# PyTorch (cu124 wheel for T4 GPU)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Transformers (for IndoBERT)
pip install transformers accelerate safetensors packaging
```

### D. Run the Notebook

Open `nlpfinalproject.ipynb` in Kaggle and click **Run All**. Ensure:
- The correct dataset is attached via **Add Input**
- The accelerator is set to **GPU T4** before running (required for Section 19 — IndoBERT)

---

## 9. Outputs

| Output | Description |
|--------|-------------|
| `tiktok_training_profile.html` | ydata-profiling EDA report for the training set |
| `tiktok_testing_profile.html` | ydata-profiling EDA report for the test set |
| `best_model.joblib` | Serialized best classical ML pipeline |
| `indobert_tiktok_sentiment/` | Fine-tuned IndoBERT model + tokenizer |
| `indobert_tiktok_checkpoints/` | Training checkpoints |

---

## 10. Methodology Notes

1. The `score` column is excluded from all model inputs to prevent data leakage.
2. The test set is evaluated **exactly once**, after all model selection and tuning is complete.
3. Hyperparameter tuning uses 5-fold stratified cross-validation on the training split only.
4. Emojis are mapped to sentiment tokens rather than stripped, preserving their signal.
5. Macro F1 is the primary evaluation metric to equally weight all three sentiment classes.
6. IndoBERT is fine-tuned in fast mode by default (`INDOBERT_FAST_MODE = True`) using stratified subsets to avoid timeout on Kaggle. Set `INDOBERT_FAST_MODE = False` for full fine-tuning.
7. A train-test phrase overlap check (3–5 n-gram leakage audit) is included to measure data contamination.

---

*Disclaimer: This project is developed as part of the COMP6885001 Natural Language Processing course at BINUS University. All scripts and formulations adhere to project policies established by the Faculty of Computer Science.*
