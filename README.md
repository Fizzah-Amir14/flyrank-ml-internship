# FlyRank ML Internship — Search Ranking Decay Classifier

**Status:** v2 complete (82.1% accuracy · 97% recall on 409,205 pages from a 79M-row Google Search Console warehouse) · **v3 in progress** — reframing the task as true forward-looking decay prediction (see [What v3 changes](#what-v3-changes-and-why))

---

## What It Does
Flags web pages for SEO/content review using Google Search Console performance and query-footprint signals, built on FlyRank's ~79M-row pseudonymized search warehouse.

> **Honest scope of v2:** the v2 label is *"the page's average position over the last 30 days is deeper than the dataset median (81.7)"* — i.e. it identifies **low-visibility pages**, not pages whose ranking is *changing*. v3 replaces this with a forward-looking decay label.

## Who It's For
- SEO teams tracking declining search performance
- Content strategists prioritizing pages for review
- SEO researchers/data scientists studying ranking changes at scale

## Results 
v2: 82.1% accuracy, 97% recall vs. a 50.1% majority-class baseline. Methodology, error analysis and known issues are in `work/capstone_report.md`.

---

## Setup

**Prerequisites:** Python 3.10+, pip, Git, Hugging Face account (for full dataset access)

**Libraries:** pandas, NumPy, scikit-learn, DuckDB, Hugging Face Hub, Matplotlib, Streamlit

**Install:**
```bash
git clone https://github.com/Fizzah-Amir14/flyrank-ml-internship.git
cd flyrank-ml-internship
pip install -r requirements.txt
```

---

## Data

**1. Bundled sample** — `data/raw/content_refresh_anonymized.csv` (~30,000 anonymized pages) — runs the reference pipeline without full warehouse access.

**2. Full-scale dataset** — ~79M GSC rows, accessed via DuckDB + Hugging Face (not downloaded locally):
1. Create/sign in to Hugging Face
2. Request access to gated `FlyRank/internship-warehouse`
3. Accept data-use terms
4. Create a read token
5. Set as environment variable:
```bash
# Windows PowerShell
$env:HF_TOKEN="your_huggingface_token"

# macOS/Linux
export HF_TOKEN="your_huggingface_token"
```

> **Note:** The full warehouse is not redistributed in this repo — only the anonymized sample is included.

---

## Running It

**Reference pipeline (bundled sample):**
```bash
python scripts/run_all.py
```
Runs the five-stage pipeline:
```
01_prepare_features.py → 02_baseline_score.py → 03_train_model.py → 04_evaluate_and_export.py → 05_build_pdf_report.py
```
Results written to `outputs/`.

**Interactive dashboard:**
```bash
streamlit run app.py
```
Opens a local dashboard (`http://localhost:8501`) — load a Quick Scenario (Healthy / Borderline / High-Risk) or adjust sliders manually to see the model's decay flag update live.

**Live demo (no install needed):**
https://flyrank-ml-internship-5tx26sdqullcgj7petr9yk.streamlit.app

**Full capstone workflow:**
`work/notebooks/capstone.ipynb` — open in Jupyter/Colab, run top to bottom after configuring HF warehouse access. Uses DuckDB to query/aggregate the 79M-row dataset, then scikit-learn for training/evaluation.

---

## Architecture
```
FlyRank internship-warehouse (Hugging Face, ~79M GSC rows)
        │  DuckDB query + feature extraction
        ▼
Feature dataset (409,205 content pages, 5 query/engagement features)
        │  train/test split (80/20, random_state=42)
        ▼
Logistic Regression classifier → flyrank_logistic_model.pkl
        │
        ▼
app.py (Streamlit) → loads saved model → live decay predictions
```
The Streamlit app loads the pre-trained model artifact and runs predictions locally — no Hugging Face call at runtime, so the live demo works without a data-access token.

---

## Evaluation Results (v2)

Trained on 409,205 records, evaluated on an 81,841-row held-out split (random_state=42):

| Metric | Value |
|---|---|
| Base rate (majority class) | 50.1% |
| Model accuracy | 82.1% |
| Recall (decaying class) | 97% |
| Precision (decaying class) | 75% |

Intentionally recall-oriented: catches 97% of decaying pages at the cost of lower precision (~75%) — built as a high-recall screening filter for human review, not an autonomous decision-maker.

---

## Limitations (v2)
- **Label measures low visibility, not decay** — `pos_last30 > 81.7` flags deep-ranking pages; it does not measure change over time
- **Feature/label window overlap** — `ctr_last30` comes from the same 30 days as the label, and CTR is strongly position-dependent, so part of the 97% recall reflects that mechanical link rather than early warning
- **Query features overlap the label period** — `fact_content_query_90d` covers the most recent ~90 days, which includes the label window
- **Random (not client-grouped) split** — pages from the same client appear in both train and test, which can inflate scores
- **No historical impression trend** — `imp_prev30` came out as 0 for every row; the likely cause is that the daily *sample* table covers ≤ 30 days, so the "previous 30 days" window was empty (being verified in v3)
- **~32% of healthy pages get flagged; ~25% of the review queue are false alarms** — tuned for high recall as a screening filter, not an autonomous decision-maker
- **Directional, not causal** — and not a Google ranking-algorithm predictor

---

## What v3 changes and why
| v2 | v3 |
|---|---|
| Label: position deeper than median (static) | Label: impressions/position **worsen over the next 30 days** after an anchor date T |
| Features and label from the same window | Features from the 90 days **before** T only; label from the 30 days **after** T |
| Daily *sample* table | Full `fact_content_daily_performance` (all months) |
| Random 80/20 split | **Client-grouped** split + time-based check |
| Compared to majority class | Compared to majority class **and** a "past trend continues" naive baseline |
| Logistic Regression | Logistic Regression vs. LightGBM, with SHAP explanations and calibration |

Step 1 (data checks + label comparison): `work/scripts/v3_01_explore_windows.py`.

---



## Repository Structure
```
flyrank-ml-internship/
│
├── app.py                      # Streamlit dashboard (live demo)
├── flyrank_logistic_model.pkl  # trained model artifact used by app.py
├── requirements.txt
├── README.md
│
├── data/raw/content_refresh_anonymized.csv
├── notebooks/                  # Week 1-2 guided starter notebooks
├── scripts/                    # reference pipeline (baseline)
├── outputs/                    # reference pipeline results
├── docs/                       # data dictionary, dataset/lane guide
├── skills/                     # AI-assistant skill files
├── submission/paper_url.txt    # deployed capstone paper URL
│
└── work/                       # all actual assignment + capstone work
    ├── notebooks/
    │   ├── w01_research_question.ipynb ... w07_action_playbook.ipynb
    │   └── capstone.ipynb
    └── capstone_report.md
```
