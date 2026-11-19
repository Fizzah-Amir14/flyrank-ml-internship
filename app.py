"""
FlyRank Search Visibility Risk — interactive demo for the v2 capstone model.

v2 label: a page is "high risk" when its average Google Search Console position over the
last 30 days is deeper than the dataset median (81.7). That measures low visibility at a
point in time, not change over time. The forward-looking decay model is v3 (in progress).
"""
import io

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------ setup
st.set_page_config(
    page_title="Search Visibility Risk · FlyRank capstone",
    page_icon="🔎",
    layout="wide",
)

INK = "#13203A"
BRONZE = "#9C6B30"
PAPER = "#F5F2EC"
RULE = "#D9D1C2"
MUTED = "#5E6676"
RISK = "#A23B2A"
CLEAR = "#2E6A4F"
MEDIAN_POS = 81.7
REPO = "https://github.com/Fizzah-Amir14/flyrank-ml-internship"
PAPER_URL = "https://fizzah-amir14.github.io/flyrank-ml-internship/"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, label, button, input, textarea {{
  font-family: 'Inter', system-ui, sans-serif;
}}
h1, h2, h3 {{ font-family: 'Fraunces', Georgia, serif !important; color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 600 !important; font-size: 2.6rem !important; line-height: 1.1 !important; }}
h3 {{ font-weight: 600 !important; font-size: 1.25rem !important; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
.lede {{ font-size: 1.05rem; color: {MUTED}; max-width: 62ch; line-height: 1.55; margin-top: -0.4rem; }}
.facts {{ color: {MUTED}; font-size: 0.86rem; margin: 0.6rem 0 0.4rem; }}
.facts a {{ color: {BRONZE}; text-decoration: none; border-bottom: 1px solid {RULE}; }}
.score {{ font-family: 'Fraunces', Georgia, serif; font-size: 5.2rem; line-height: 0.95; font-weight: 600; margin: 0.2rem 0 0.4rem; }}
.score small {{ font-size: 2rem; font-weight: 400; color: {MUTED}; margin-left: 0.15rem; }}
.verdict {{ font-size: 1.05rem; font-weight: 600; margin-bottom: 0.3rem; }}
.explain {{ color: {MUTED}; font-size: 0.92rem; line-height: 1.5; max-width: 46ch; }}
.note {{ border-left: 3px solid {BRONZE}; padding: 0.55rem 0.9rem; background: #EFE9DE; color: {INK};
         font-size: 0.9rem; line-height: 1.5; border-radius: 0 6px 6px 0; }}
.mono {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.82em; }}
[data-testid="stMetricValue"] {{ font-family: 'Fraunces', Georgia, serif; }}
div[data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {RULE}; }}
button[data-baseweb="tab"] {{ font-size: 1rem; padding-left: 0; padding-right: 0; }}
@media (max-width: 640px) {{ h1 {{ font-size: 2rem !important; }} .score {{ font-size: 4rem; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("flyrank_logistic_model.pkl")


model = load_model()
COLS = list(model.feature_names_in_)
COEF = model.coef_[0]
B0 = float(model.intercept_[0])

FEATURES = {
    "ctr_last30": ("Click-through rate, last 30 days", 0.0, 0.20, 0.001,
                   "Clicks ÷ impressions over the last 30 days, as a fraction (0.03 = 3%)."),
    "visible_queries": ("Visible search queries", 0.0, 100.0, 1.0,
                        "Distinct queries GSC reports for this page (rare and hidden queries excluded)."),
    "top_query_share": ("Share from the top query", 0.0, 1.0, 0.01,
                        "Share of impressions that come from the page's single biggest query."),
    "rare_share": ("Share from rare queries", 0.0, 1.0, 0.01,
                   "Share of impressions from queries with under 10 impressions."),
    "anon_share": ("Share from hidden queries", 0.0, 1.0, 0.01,
                   "Share of impressions from queries GSC anonymises for privacy."),
}

PRESETS = {
    "Well-visible page": {"ctr_last30": 0.03, "visible_queries": 25.0, "top_query_share": 0.25, "rare_share": 0.15, "anon_share": 0.10},
    "Borderline page": {"ctr_last30": 0.002, "visible_queries": 3.0, "top_query_share": 0.60, "rare_share": 0.10, "anon_share": 0.10},
    "Barely-visible page": {"ctr_last30": 0.0, "visible_queries": 1.0, "top_query_share": 0.90, "rare_share": 0.05, "anon_share": 0.05},
}


def risk(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[COLS])[:, 1]


def one(values: dict) -> float:
    return float(risk(pd.DataFrame([values], columns=COLS))[0])


def base_layout(fig, height):
    fig.update_layout(
        height=height, margin=dict(t=10, b=40, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=INK, size=13),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=RULE, zeroline=False, linecolor=RULE)
    fig.update_yaxes(gridcolor=RULE, zeroline=False, linecolor=RULE)
    return fig


def depth_ruler(p: float) -> str:
    """Search-results depth ruler: where the median line sits, and how likely the page is below it."""
    h, top, bottom = 300, 18, 282
    y_of = lambda pos: top + (bottom - top) * (pos - 1) / 99
    y_med = y_of(MEDIAN_POS)
    ticks = "".join(
        f'<line x1="58" x2="66" y1="{y_of(pos):.1f}" y2="{y_of(pos):.1f}" stroke="{RULE}"/>'
        f'<text x="52" y="{y_of(pos)+4:.1f}" text-anchor="end" font-size="11" fill="{MUTED}">p.{(pos-1)//10+1}</text>'
        for pos in (1, 21, 41, 61, 81, 100)
    )
    shade = 0.12 + 0.6 * p
    return f"""
<svg viewBox="0 0 230 {h}" width="100%" style="max-width:260px" role="img"
     aria-label="Search results depth ruler. Chance this page ranks deeper than position {MEDIAN_POS}: {p:.0%}">
  <rect x="66" y="{top}" width="38" height="{y_med-top:.1f}" fill="{CLEAR}" fill-opacity="{0.10 + 0.55*(1-p):.2f}" rx="3"/>
  <rect x="66" y="{y_med:.1f}" width="38" height="{bottom-y_med:.1f}" fill="{RISK}" fill-opacity="{shade:.2f}" rx="3"/>
  {ticks}
  <line x1="60" x2="112" y1="{y_med:.1f}" y2="{y_med:.1f}" stroke="{BRONZE}" stroke-width="2"/>
  <text x="118" y="{y_med-6:.1f}" font-size="12" fill="{INK}" font-weight="600">median rank {MEDIAN_POS}</text>
  <text x="118" y="{y_med+12:.1f}" font-size="11" fill="{MUTED}">deeper = high risk</text>
  <text x="118" y="{top+10}" font-size="11" fill="{MUTED}">top of Google</text>
  <text x="118" y="{y_med-46:.1f}" font-size="22" fill="{RISK}" font-family="Fraunces, Georgia, serif" font-weight="600">{p:.0%}</text>
  <text x="118" y="{y_med-30:.1f}" font-size="11" fill="{MUTED}">chance it sits below</text>
</svg>"""


# ------------------------------------------------------------------ state
for k, v in PRESETS["Borderline page"].items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("threshold", 0.50)


def apply_preset():
    choice = st.session_state.get("preset")
    if choice in PRESETS:
        for k, v in PRESETS[choice].items():
            st.session_state[k] = v


# ------------------------------------------------------------------ header
st.title("Which pages is Google barely showing?")
st.markdown(
    '<p class="lede">Score a page’s search-query footprint and get the chance it ranks deeper than the '
    "median result in FlyRank’s data. Built to help an SEO team decide which pages to review first.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="facts">Logistic regression on 409,205 pages from a 79M-row Google Search Console warehouse. '
    f'Held-out accuracy 82.1%, recall 97%, precision 75%.<br>'
    f'<a href="{PAPER_URL}">Read the paper</a> &nbsp;&nbsp; <a href="{REPO}">View the code</a></p>',
    unsafe_allow_html=True,
)

tab_page, tab_list, tab_model = st.tabs(["Score a page", "Score a list of pages", "How the model works"])

# ------------------------------------------------------------------ tab 1
with tab_page:
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown("### Page signals")
        st.pills(
            "Start from an example", list(PRESETS), key="preset", on_change=apply_preset,
            selection_mode="single",
        )
        for key, (label, lo, hi, step, helptext) in FEATURES.items():
            st.slider(label, lo, hi, key=key, step=step, help=helptext,
                      format="%.3f" if key == "ctr_last30" else ("%d" if key == "visible_queries" else "%.2f"))

    values = {k: float(st.session_state[k]) for k in COLS}
    p = one(values)
    thr = st.session_state["threshold"]
    flagged = p >= thr

    with right:
        r1, r2 = st.columns([1.2, 1])
        with r1:
            color = RISK if flagged else CLEAR
            st.markdown(
                f'<div class="score" style="color:{color}">{p*100:.0f}<small>%</small></div>'
                f'<div class="verdict" style="color:{color}">'
                f'{"Flag for review" if flagged else "No review needed"}</div>'
                f'<div class="explain">Chance this page ranks deeper than position {MEDIAN_POS}. '
                f'Pages at or above your {thr:.0%} cut-off go into the review queue.</div>',
                unsafe_allow_html=True,
            )
            st.slider("Review cut-off", 0.05, 0.95, key="threshold", step=0.05, format="%.2f",
                      help="Lower it to catch more low-visibility pages, at the cost of more false alarms.")
        with r2:
            st.markdown(depth_ruler(p), unsafe_allow_html=True)

        st.markdown("### Why this score")
        logits = COEF * np.array([values[c] for c in COLS])
        names = [FEATURES[c][0] for c in COLS]
        fig = go.Figure(go.Waterfall(
            orientation="h",
            measure=["absolute"] + ["relative"] * len(COLS) + ["total"],
            y=["Starting point"] + names + ["Final score"],
            x=[B0] + list(logits) + [0],
            connector=dict(line=dict(color=RULE)),
            increasing=dict(marker=dict(color=RISK)),
            decreasing=dict(marker=dict(color=CLEAR)),
            totals=dict(marker=dict(color=INK)),
            textposition="none",
            hovertemplate="%{y}: %{x:+.2f}<extra></extra>",
        ))
        base_layout(fig, 300).update_yaxes(autorange="reversed")
        fig.update_xaxes(title_text="← lower risk     log-odds     higher risk →", title_font=dict(size=12, color=MUTED))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("### What would change the score")
    st.caption("Each curve moves one signal and keeps the others where you set them. The dot is this page.")
    c1, c2 = st.columns(2, gap="large")
    for col, key, xs in (
        (c1, "visible_queries", np.linspace(0, 40, 81)),
        (c2, "ctr_last30", np.linspace(0, 0.05, 101)),
    ):
        grid = pd.DataFrame([{**values, key: x} for x in xs], columns=COLS)
        ys = risk(grid)
        f = go.Figure()
        f.add_hrect(y0=thr, y1=1, fillcolor=RISK, opacity=0.06, line_width=0)
        f.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=INK, width=2.5),
                               hovertemplate=f"{FEATURES[key][0]}: %{{x:.3f}}<br>risk %{{y:.0%}}<extra></extra>"))
        f.add_trace(go.Scatter(x=[values[key]], y=[p], mode="markers",
                               marker=dict(size=12, color=BRONZE, line=dict(color=PAPER, width=2)),
                               hoverinfo="skip"))
        f.add_hline(y=thr, line=dict(color=BRONZE, width=1, dash="dot"))
        base_layout(f, 240)
        f.update_yaxes(range=[0, 1], tickformat=".0%", title_text="Risk")
        f.update_xaxes(title_text=FEATURES[key][0], tickformat=".1%" if key == "ctr_last30" else None)
        with col:
            st.plotly_chart(f, width="stretch", config={"displayModeBar": False})

# ------------------------------------------------------------------ tab 2
def example_pages(n=40, seed=7) -> pd.DataFrame:
    """Synthetic pages for the demo — not real FlyRank data."""
    rng = np.random.default_rng(seed)
    q = np.where(rng.random(n) < 0.35, rng.integers(0, 3, n), rng.gamma(1.6, 7, n).round())
    df = pd.DataFrame({
        "page": [f"example-page-{i+1:02d}" for i in range(n)],
        "ctr_last30": np.clip(rng.gamma(1.0, 0.012, n) * (q > 2), 0, 0.2).round(4),
        "visible_queries": q,
        "top_query_share": np.clip(rng.beta(2, 3, n) + 0.3 * (q < 3), 0, 1).round(2),
        "rare_share": np.where(q < 3, rng.beta(1, 14, n), rng.beta(2, 8, n)).round(2),
        "anon_share": np.where(q < 3, rng.beta(1, 16, n), rng.beta(2, 9, n)).round(2),
    })
    return df


with tab_list:
    st.markdown("### Build a review queue")
    st.markdown(
        '<p class="explain">Upload a CSV with one row per page and the five signal columns '
        f'(<span class="mono">{", ".join(COLS)}</span>), plus an optional <span class="mono">page</span> column. '
        "Scores are calculated in this session only; nothing is stored.</p>",
        unsafe_allow_html=True,
    )
    up_col, ex_col = st.columns([2, 1])
    with up_col:
        upload = st.file_uploader("Upload pages (CSV)", type="csv", label_visibility="collapsed")
    with ex_col:
        use_example = st.toggle("Use 40 example pages", value=upload is None,
                                help="Randomly generated pages for trying the tool. Not real client data.")

    pages, problem = None, None
    if upload is not None and not use_example:
        try:
            pages = pd.read_csv(upload)
            missing = [c for c in COLS if c not in pages.columns]
            if missing:
                problem = f"Your file is missing these columns: {', '.join(missing)}. Add them and upload again."
                pages = None
            else:
                pages[COLS] = pages[COLS].apply(pd.to_numeric, errors="coerce")
                bad = pages[COLS].isna().any(axis=1).sum()
                if bad:
                    problem = f"{bad} rows had empty or non-numeric values and were skipped."
                    pages = pages.dropna(subset=COLS)
                if "page" not in pages.columns:
                    pages.insert(0, "page", [f"row-{i+1}" for i in range(len(pages))])
        except Exception:
            problem = "That file could not be read as a CSV. Export it again as comma-separated values."
    elif use_example:
        pages = example_pages()

    if problem:
        st.warning(problem)

    if pages is None or pages.empty:
        st.info("Upload a CSV or switch on the example pages to see a ranked review queue.")
    else:
        pages = pages.copy()
        pages["risk"] = risk(pages) * 100
        qthr = st.slider("Review cut-off", 0.05, 0.95, 0.50, 0.05, format="%.2f", key="queue_threshold")
        pages["action"] = np.where(pages["risk"] >= qthr * 100, "Review", "Monitor")
        pages = pages.sort_values("risk", ascending=False).reset_index(drop=True)

        n_review = int((pages["action"] == "Review").sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Pages scored", f"{len(pages):,}")
        m2.metric("In review queue", f"{n_review:,}")
        m3.metric("Expected false alarms", f"≈{round(n_review * 0.25):,}",
                  help="At v2's 75% precision, about 1 in 4 flagged pages is a false alarm.")

        hist = go.Figure(go.Histogram(x=pages["risk"] / 100, xbins=dict(start=0, end=1, size=0.05),
                                      marker=dict(color=INK, line=dict(color=PAPER, width=1))))
        hist.add_vline(x=qthr, line=dict(color=BRONZE, width=2))
        base_layout(hist, 200).update_xaxes(range=[0, 1], tickformat=".0%", title_text="Risk")
        hist.update_yaxes(title_text="Pages")
        st.plotly_chart(hist, width="stretch", config={"displayModeBar": False})

        st.dataframe(
            pages[["page", "risk", "action"] + COLS],
            width="stretch", hide_index=True, height=360,
            column_config={
                "page": "Page",
                "risk": st.column_config.ProgressColumn("Risk", format="%.0f%%", min_value=0, max_value=100),
                "action": "Action",
                "ctr_last30": st.column_config.NumberColumn("CTR", format="%.3f"),
                "visible_queries": st.column_config.NumberColumn("Queries", format="%d"),
                "top_query_share": st.column_config.NumberColumn("Top query", format="%.2f"),
                "rare_share": st.column_config.NumberColumn("Rare", format="%.2f"),
                "anon_share": st.column_config.NumberColumn("Hidden", format="%.2f"),
            },
        )
        buf = io.StringIO()
        pages.to_csv(buf, index=False)
        st.download_button("Download review queue (CSV)", buf.getvalue(), "review_queue.csv", "text/csv")

# ------------------------------------------------------------------ tab 3
with tab_model:
    a, b = st.columns([1.3, 1], gap="large")
    with a:
        st.markdown("### What it predicts")
        st.markdown(
            f"A page is **high risk** when its average Google position over the last 30 days is deeper than "
            f"the dataset median, **{MEDIAN_POS}**. That is a snapshot of low visibility. It does not say the page "
            "is getting worse over time."
        )
        st.markdown("### What it looks at")
        st.markdown(
            "Five signals from a page’s search-query footprint: click-through rate, how many queries it shows up "
            "for, and how its impressions split between its top query, rare queries and hidden queries. "
            "Position itself is left out, so the model can’t simply read the answer."
        )
        st.markdown("### What it learned")
        SPREAD = {"ctr_last30": 0.03, "visible_queries": 10, "top_query_share": 0.3,
                  "rare_share": 0.3, "anon_share": 0.3}
        effect = np.array([COEF[i] * SPREAD[c] for i, c in enumerate(COLS)])
        coef_df = pd.DataFrame({"Signal": [FEATURES[c][0] for c in COLS], "Effect": effect}).sort_values("Effect")
        cf = go.Figure(go.Bar(x=coef_df["Effect"], y=coef_df["Signal"], orientation="h",
                              marker=dict(color=[RISK if w > 0 else CLEAR for w in coef_df["Effect"]]),
                              hovertemplate="%{y}: %{x:+.2f} log-odds<extra></extra>"))
        base_layout(cf, 230).update_xaxes(title_text="Change in log-odds from a typical increase (left = lower risk)")
        st.plotly_chart(cf, width="stretch", config={"displayModeBar": False})
        st.caption("Typical increase: +3 points of CTR, +10 queries, or +0.3 in a share. Every signal lowers risk "
                   "as it rises; query count and CTR carry almost all of it.")
    with b:
        st.markdown("### How well it does")
        st.dataframe(pd.DataFrame({
            "Measure": ["Accuracy", "Recall (high-risk pages caught)", "Precision (flags that are right)",
                        "Majority-class baseline"],
            "Held-out result": ["82.1%", "97%", "75%", "50.1%"],
        }), hide_index=True, width="stretch")
        st.markdown("### Limits worth knowing")
        st.markdown(
            "- CTR comes from the same 30 days as the label, and CTR depends on position, so part of the score "
            "reflects that link rather than an early warning.\n"
            "- Train and test pages were split at random, not by client.\n"
            "- About 1 in 4 flagged pages is a false alarm. Use it to order a review queue, not to act on its own.\n"
            "- Associations only: it does not say why a page ranks where it does."
        )
        st.markdown(
            '<div class="note"><b>Next version.</b> v3 predicts whether a page will lose impressions or '
            "position over the next 30 days using only data from before that window, split by client, "
            f'and compared against a “past trend continues” baseline. <a href="{REPO}">Follow progress on GitHub</a>.</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    f'<p class="facts" style="margin-top:2.5rem">Built by Fizzah Amir during the FlyRank ML internship, 2026. '
    "Model and demo use pseudonymised, aggregated data only.</p>",
    unsafe_allow_html=True,
)
