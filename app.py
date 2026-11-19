import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(page_title="FlyRank Content Decay Predictor", layout="centered")
st.title("📊 Content Ranking Decay Predictor")
st.write("Flags pages at risk of low search visibility using query and engagement features (v2 model).")
st.caption("v2 identifies pages ranking deeper than the dataset median position. A forward-looking decay model (v3) is in progress.")

# ============================================================
# 2. LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load('flyrank_logistic_model.pkl')

model = load_model()

FEATURE_NAMES = [
    "Click-Through Rate",
    "Visible Query Count",
    "Top Query Share",
    "Rare Impressions Share",
    "Anonymized Impressions Share",
]
FEATURE_KEYS = ["ctr", "visible_queries", "top_query_share", "rare_share", "anon_share"]

# ============================================================
# 3. SESSION STATE — for preset scenarios + prediction history
# ============================================================
defaults = {
    "ctr": 0.05,
    "visible_queries": 10.0,
    "top_query_share": 0.40,
    "rare_share": 0.15,
    "anon_share": 0.10,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "history" not in st.session_state:
    st.session_state.history = []

PRESETS = {
    # Presets chosen so each lands in the zone its label claims (checked against the saved model):
    # healthy ~0%, borderline ~38%, high-risk ~61% risk probability.
    "🟢 Healthy Page": {"ctr": 0.03, "visible_queries": 25.0, "top_query_share": 0.25, "rare_share": 0.15, "anon_share": 0.10},
    "🟡 Borderline Page": {"ctr": 0.002, "visible_queries": 3.0, "top_query_share": 0.60, "rare_share": 0.10, "anon_share": 0.10},
    "🔴 High-Risk Page": {"ctr": 0.0, "visible_queries": 1.0, "top_query_share": 0.90, "rare_share": 0.05, "anon_share": 0.05},
}

# ============================================================
# 4. PRESET SCENARIO BUTTONS
# ============================================================
st.subheader("⚡ Quick Scenarios")
st.caption("Load an example instead of typing values. In this model, query count and CTR carry almost all the signal.")
preset_cols = st.columns(len(PRESETS))
for col, (label, values) in zip(preset_cols, PRESETS.items()):
    if col.button(label, width="stretch"):
        for k, v in values.items():
            st.session_state[k] = v
        st.rerun()

st.write("---")

# ============================================================
# 5. INPUT FORM (bound to session_state so presets + sliders sync)
# ============================================================
st.subheader("📥 Enter Page Context Metrics")

col1, col2 = st.columns(2)

with col1:
    ctr = st.slider("Click-Through Rate (ctr_last30)", 0.0, 1.0, key="ctr", step=0.01,
                     help="Fraction of impressions that resulted in a click over the last 30 days.")
    visible_queries = st.slider("Visible Query Count", 0.0, 100.0, key="visible_queries", step=1.0,
                                 help="Number of distinct (non-anonymized) queries surfacing this page.")
    top_query_share = st.slider("Top Query Share", 0.0, 1.0, key="top_query_share", step=0.01,
                                 help="Share of impressions concentrated in the single top query.")

with col2:
    rare_share = st.slider("Rare Impressions Share", 0.0, 1.0, key="rare_share", step=0.01,
                            help="Share of impressions coming from long-tail/rare queries.")
    anon_share = st.slider("Anonymized Impressions Share", 0.0, 1.0, key="anon_share", step=0.01,
                            help="Share of impressions from queries GSC hides for privacy.")

input_values = [ctr, visible_queries, top_query_share, rare_share, anon_share]

# ============================================================
# 6. RUN PREDICTION
# ============================================================
st.write("---")
run = st.button("🚀 Run Machine Learning Prediction", width="stretch", type="primary")

if run:
    input_data = pd.DataFrame([input_values], columns=model.feature_names_in_)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.session_state.history.append({
        "CTR": ctr, "Queries": visible_queries, "Top Share": top_query_share,
        "Rare": rare_share, "Anon": anon_share,
        "Risk Prob (%)": round(probability * 100, 1),
        "Verdict": "High Risk" if prediction == 1 else "Lower Risk",
    })

    st.subheader("🔮 Model Output Analysis")

    # ---- 6a. Verdict banner ----
    if prediction == 1:
        st.error(f"🔴 **High Risk:** likely ranking deeper than the dataset median position. (Risk probability: {probability:.2%})")
    else:
        st.success(f"🟢 **Lower Risk:** likely ranking above the dataset median position. (Risk probability: {probability:.2%})")

    # ---- 6b. Gauge chart ----
    gauge_color = "#e74c3c" if probability >= 0.5 else "#2ecc71"
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': "%"},
        title={'text': "Risk Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, 50], 'color': '#1e3d2f'},
                {'range': [50, 100], 'color': '#3d1e1e'},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.9,
                'value': 50,
            }
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=30, r=30),
                             paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    st.plotly_chart(fig_gauge, width="stretch")

    # ---- 6c. Feature contribution breakdown ----
    st.subheader("🧩 What Drove This Prediction")
    st.caption("Each feature's contribution = its logistic regression coefficient × your input value. Positive bars push toward 'high risk', negative bars toward 'lower risk'.")

    coefs = model.coef_[0]
    contributions = coefs * np.array(input_values)
    contrib_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Contribution": contributions,
    }).sort_values("Contribution")

    fig_bar = go.Figure(go.Bar(
        x=contrib_df["Contribution"],
        y=contrib_df["Feature"],
        orientation='h',
        marker_color=["#e74c3c" if v > 0 else "#2ecc71" for v in contrib_df["Contribution"]],
    ))
    fig_bar.update_layout(
        height=300, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}, xaxis_title="Push toward high risk →",
    )
    st.plotly_chart(fig_bar, width="stretch")

    # ---- 6d. Engineering note + recommendation ----
    if prediction == 1:
        st.info(
            "⚠️ **Engineering Note:** This model runs on a high-sensitivity configuration "
            "(97% Recall on validation) to guarantee no dropping page is missed. Because it prioritizes "
            "catch-rate over precision (75%), roughly 1 in 4 flags is a false alarm. "
            "Account managers should manually verify context before major restructuring."
        )
        st.write("**Recommendation:** Prioritize this page for content optimization and query diversity adjustments.")
    else:
        st.write("**Recommendation:** No action needed. Continue routine monitoring.")

# ============================================================
# 7. PREDICTION HISTORY (session-only, resets on refresh)
# ============================================================
if st.session_state.history:
    st.write("---")
    st.subheader("🕓 This Session's Predictions")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, width="stretch", hide_index=True)
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

# ============================================================
# 8. SIDEBAR — model context
# ============================================================
with st.sidebar:
    st.header("ℹ️ About This Model")
    st.markdown(
        """
        **Type:** Logistic Regression
        **Training data:** 409,205 content-level rows
        (FlyRank GSC warehouse, last-30-day window)

        **Validation performance:**
        - Accuracy: 82.1%
        - Recall (decay class): 97%
        - Precision (decay class): 75%

        **Label:** Page is flagged "high risk" if its
        average position over the last 30 days is
        deeper than the dataset median (81.7).
        This measures low visibility at a point in
        time, not change over time.

        **Note:** `pos_last30` was excluded from the
        features to prevent direct target leakage.
        CTR comes from the same window and is
        position-dependent, so scores partly reflect
        that link.

        **Next:** v3 predicts future decay from past-only
        features. See the repo README.
        """
    )