Capstone Report — Search Ranking Decay Classifier
Author: Fizzah Amir
Lane: Prediction / Classification (search ranking decay detection)
Repo: https://github.com/Fizzah-Amir14/flyrank-ml-internship
Date:3 September 2026

1. Problem framing

This project supports a content-review prioritization decision: which web pages should an SEO/content team review first because their search performance is at risk of decaying. The unit of analysis is a single content page (identified by its query/engagement footprint in Google Search Console data). The output is a binary decay flag with a probability score, which a human content strategist uses to build a review queue — the model does not take any action on its own.

The cost of a wrong call is asymmetric: missing a genuinely decaying page (false negative) means a page silently loses traffic before anyone intervenes, which is expensive and slow to reverse. Flagging a healthy page for review (false positive) only costs a few minutes of a reviewer's time. This asymmetry is why the model was deliberately tuned for high recall over high precision.

ML helps here because ranking decay isn't driven by one obvious signal — it's a combination of click-through rate shifts, impression share, and query-footprint changes that are hard to threshold by hand at the scale of FlyRank's ~79M-row warehouse. A hand-written rule (the baseline) can catch the obvious cases, but a model trained on the full feature set catches more of the subtler ones.

2. Data safety
Data used: query-context and engagement features derived from FlyRank's fact_content_daily_performance table (~79M rows), accessed via DuckDB/Hugging Face — not downloaded in full locally.
Columns deliberately excluded: any client-identifying or pseudonymous ID fields were used only for grouping (e.g. to build the train/test split), never as model features.
Leakage risk considered and found: avg_gsc_position was initially included as a feature and caused AUC to inflate from a realistic ~0.70 to a suspicious 1.0 — this field turned out to be label-derived (it moves with the same underlying signal as the decay label itself), so it was removed from the final feature set. This was the single most important finding of the project and is documented in the ML-05 notebook.
Confirmation: no client-identifying data (domain names, query text, client names) appears anywhere in work/ — all analysis runs on the anonymized/aggregated feature set.
3. Baseline

A transparent, hand-written rule-based score was built first (see notebooks/02_your_first_readable_model.ipynb), flagging pages using simple thresholds on click-through rate and impression trends. This baseline is a fair comparison because it uses the same underlying signals as the model, just combined by a fixed rule instead of a learned weighting — so any improvement from the model reflects the model learning better combinations of signals, not access to different data.

Base rate (majority class) on the evaluation split: 50.1% — reported alongside model accuracy so the accuracy number can be judged against how hard the task actually is.

4. Model / analysis

Method: Logistic Regression classifier, chosen for interpretability — coefficients can be directly discussed with a non-technical content team, which matters for a tool meant to support human review decisions rather than replace them.

Final feature set (5 query/engagement features) derived via DuckDB aggregation from the 79M-row warehouse. avg_gsc_position was deliberately left out after the leakage discovery (see Section 2).

Target definition: a binary label indicating whether a page's search performance is trending downward over the observed window (decaying vs. not decaying).

5. Evaluation
Split: 80/20 train/test split, random_state=42, evaluated on an 81,841-row held-out set drawn from the full 409,205-record feature set.
Metrics (model vs. baseline, same split):
Metric	Value
Base rate (majority class)	50.1%
Model accuracy	82.1%
Recall (decaying class)	97%
Precision (decaying class)	75%
Error analysis: the model is intentionally recall-oriented — it catches 97% of pages that are actually decaying, but this comes with a ~33% false-flag rate on healthy pages (precision 75% on the decaying class). In practice this means roughly a third of the review queue will be false alarms, which is an acceptable trade-off for a screening tool but should be stated explicitly to whoever uses the output.
A further validation audit was run on a held-out split (see work/notebooks/w06_validation_audit.ipynb) to sanity-check that these numbers hold beyond the single train/test split.
6. Interpretation

A follow-up signal audit (comparing flagged pages against GA4 session data, separate from the GSC data the model was trained on) produced a mixed verdict: pages flagged as decaying showed lower GSC clicks/CTR (consistent with the model's decay signal) but higher GA4 sessions over the same window. This is a genuine, reportable negative/mixed result rather than a clean confirmation — it suggests the model is picking up a real search-visibility signal, but that visibility decline doesn't always translate directly into an overall traffic decline, possibly because of traffic arriving through non-search channels. This nuance is flagged rather than smoothed over, per the "no causal claims" rule below.

7. Recommendation

The model's output should be used to build a ranked review queue: pages with the highest decay probability go to a content strategist first. A FlyRank editor would use this by pulling the top N flagged pages weekly and checking them for content staleness, competitive changes, or technical issues, rather than trying to review the entire site catalog by hand.

Confidence and limits, stated explicitly:

Directional, not causal — the model finds correlated signals, it does not claim any of them cause decay.
Trained on a single 60-day snapshot; not yet validated against newer data beyond the audit in Section 5.
Not a Google ranking-algorithm predictor — it flags pages already showing decay signals, it does not reverse-engineer or predict ranking algorithm behavior.
The ~33% false-flag rate means this is a screening filter for human review, not an autonomous decision-maker.
8. Reproducibility
bash
git clone https://github.com/Fizzah-Amir14/flyrank-ml-internship.git
cd flyrank-ml-internship
pip install -r requirements.txt

Full capstone workflow: work/notebooks/capstone.ipynb — run top to bottom after configuring Hugging Face warehouse access (HF_TOKEN environment variable, gated access to FlyRank/internship-warehouse).

Reference pipeline on the bundled sample (no warehouse access needed):

bash
python scripts/run_all.py
Random seed: random_state=42 used throughout (train/test split and model training).
Environment: <ADD pip freeze highlights or requirements.txt deltas if any packages were pinned/changed from the starter repo>

Claims checklist: all metrics above are measured (from the held-out split), the ranking-decay relationship is stated as directional, and the recommendation is framed as decision-support only — no causal or autonomous-decision language is used anywhere in this report. Metrics vs. base rate: base rate (50.1%) is reported alongside accuracy (82.1%) in Section 5, so the accuracy number can be judged against how balanced the task actually is.
