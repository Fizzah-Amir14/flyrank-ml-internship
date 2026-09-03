# FlyRank ML Internship — Search Ranking Decay Classifier

**Performance:** 82.1% accuracy · 97% recall on a 409,205-record feature set derived from a 79M-row Google Search Console dataset

---

## What It Does
Identifies web pages at risk of search-ranking decay by analyzing Google Search Console performance and engagement signals over time. Uses a classifier trained on FlyRank's ~79M-row search-ranking warehouse to flag pages needing SEO/content intervention.

## Who It's For
- SEO teams tracking declining search performance
- Content strategists prioritizing pages for review
- SEO researchers/data scientists studying ranking changes at scale

## Results
82.1% accuracy, 97% recall. Full methodology, baseline comparison, and error analysis are in `work/capstone_report.md`.

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

## Limitations
- **Directional, not causal** — flags correlated signals, doesn't claim they cause decay
- **No historical impression trend** — `imp_prev30` was missing for most of the sample; relies on current signals, not trend over time
- **~33% false-flag rate on healthy pages** — tuned for high recall, meant to narrow a review queue, not auto-act
- **Trained on a single 60-day snapshot** — not validated against newer data beyond the audit in `work/notebooks/w06_validation_audit.ipynb`
- **Not a ranking-algorithm predictor** — flags existing decay signals, doesn't reverse-engineer Google's algorithm

---

## Built With AI
Built with Claude (Anthropic) as a coding and debugging assistant — used for drafting pipeline code, debugging deployment, and structuring documentation. All modeling decisions (feature selection, leakage checks, train/test split design, evaluation metrics) were made and verified independently; the notebook was re-run to confirm the reported accuracy before publishing.

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
