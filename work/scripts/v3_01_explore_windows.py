"""
FlyRank v3 - Step 1: check the data and preview a forward-looking decay label.

Run locally (needs HF_TOKEN with gated access to FlyRank/internship-warehouse):
    pip install duckdb pandas pyarrow
    export HF_TOKEN=hf_xxx
    python work/scripts/v3_01_explore_windows.py

Output: printed diagnostics + data/v3/features_preview.parquet
DATA_USE rule: data/v3/ stays out of git (already covered by the data/** rule in .gitignore).
"""
import os
from pathlib import Path

import duckdb

HF_TOKEN = os.environ["HF_TOKEN"]
REL = "hf://datasets/FlyRank/internship-warehouse"
FACT = f"read_parquet('{REL}/fact_content_daily_performance/**/*.parquet', hive_partitioning=true)"
SAMPLE = f"read_parquet('{REL}/fact_content_daily_performance_sample.parquet')"
CLIENTS = f"read_parquet('{REL}/dim_clients.parquet')"

con = duckdb.connect()
con.execute(f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{HF_TOKEN}')")

# --- 1. Why was imp_prev30 always 0 in v2? Compare date coverage -----------------
print("\n[1] Date coverage")
for name, src in [("sample (used in v2)", SAMPLE), ("full fact", FACT)]:
    lo, hi, days = con.sql(
        f"SELECT MIN(report_date), MAX(report_date), COUNT(DISTINCT report_date) FROM {src}"
    ).fetchone()
    print(f"  {name:20} {lo} -> {hi}  ({days} distinct days)")
# If the sample covers <= 30 days, the "prev30" window was simply empty.

# --- 2. Pick the anchor date T --------------------------------------------------
# Features:  [T-90, T)      (past only)
# Label:     [T, T+30)      (future only)
T = con.sql(f"SELECT MAX(report_date) - INTERVAL 30 DAY FROM {FACT}").fetchone()[0]
print(f"\n[2] Anchor date T = {T}")

eligible = con.sql(f"""
    SELECT COUNT(*) FILTER (WHERE gsc_data_start <= DATE '{T:%Y-%m-%d}' - INTERVAL 90 DAY) AS ok,
           COUNT(*) AS total
    FROM {CLIENTS}
""").fetchone()
print(f"    clients with >= 90 days GSC history before T: {eligible[0]} / {eligible[1]}")

# --- 3. Build windows per content item -----------------------------------------
df = con.sql(f"""
    WITH c AS (
        SELECT client_hash_id FROM {CLIENTS}
        WHERE gsc_data_start <= DATE '{T:%Y-%m-%d}' - INTERVAL 90 DAY
    ),
    f AS (
        SELECT f.*,
               DATE_DIFF('day', f.report_date, DATE '{T:%Y-%m-%d}') AS d  -- days before T (negative = after)
        FROM {FACT} f JOIN c USING (client_hash_id)
        WHERE f.report_date >= DATE '{T:%Y-%m-%d}' - INTERVAL 90 DAY
          AND f.report_date <  DATE '{T:%Y-%m-%d}' + INTERVAL 30 DAY
    )
    SELECT client_hash_id, content_hash_id,
        -- feature window (past)
        SUM(gsc_impressions) FILTER (WHERE d BETWEEN 1 AND 30)  AS imp_f1,   -- last 30d before T
        SUM(gsc_impressions) FILTER (WHERE d BETWEEN 31 AND 60) AS imp_f2,
        SUM(gsc_impressions) FILTER (WHERE d BETWEEN 61 AND 90) AS imp_f3,
        SUM(gsc_clicks)      FILTER (WHERE d BETWEEN 1 AND 30)  AS clk_f1,
        SUM(gsc_avg_position * gsc_impressions) FILTER (WHERE d BETWEEN 1 AND 30)
            / NULLIF(SUM(gsc_impressions) FILTER (WHERE d BETWEEN 1 AND 30), 0) AS pos_f1,
        SUM(gsc_avg_position * gsc_impressions) FILTER (WHERE d BETWEEN 31 AND 60)
            / NULLIF(SUM(gsc_impressions) FILTER (WHERE d BETWEEN 31 AND 60), 0) AS pos_f2,
        COUNT(*) FILTER (WHERE d BETWEEN 1 AND 90 AND gsc_impressions > 0) AS active_days_f,
        MIN(report_date) AS first_seen,
        -- label window (future)
        SUM(gsc_impressions) FILTER (WHERE d <= 0) AS imp_label,
        SUM(gsc_avg_position * gsc_impressions) FILTER (WHERE d <= 0)
            / NULLIF(SUM(gsc_impressions) FILTER (WHERE d <= 0), 0) AS pos_label
    FROM f
    GROUP BY 1, 2
""").df()

print(f"\n[3] Content items in window: {len(df):,}")

# Keep pages that existed for the whole feature window and have enough signal
full_history = df["first_seen"] <= (T - __import__("datetime").timedelta(days=89))
enough_signal = df["imp_f1"].fillna(0) >= 50
df = df[full_history & enough_signal].copy()
print(f"    after filters (full 90d history, >= 50 impressions in last 30d): {len(df):,}")
print(f"    clients represented: {df['client_hash_id'].nunique()}")

# --- 4. Compare candidate labels ------------------------------------------------
df["imp_label"] = df["imp_label"].fillna(0)
labels = {
    "A: impressions drop > 20% next 30d": df["imp_label"] < 0.8 * df["imp_f1"],
    "B: position worsens by >= 3 next 30d": (df["pos_label"] - df["pos_f1"]) >= 3,
    "C: A and B together (strict decay)": (df["imp_label"] < 0.8 * df["imp_f1"])
                                          & ((df["pos_label"] - df["pos_f1"]) >= 3),
}
print("\n[4] Label base rates")
for name, y in labels.items():
    print(f"    {name:40} {y.mean():6.1%}")

# Sanity check: how often is the future label just "the past trend continuing"?
past_down = df["imp_f1"] < 0.8 * df["imp_f2"].fillna(0)
agree = (past_down == labels["A: impressions drop > 20% next 30d"]).mean()
print(f"\n    'past trend continues' agreement with label A: {agree:.1%}")
print("    -> this is the naive baseline v3 must beat.")

# --- 5. Save locally (never commit) ---------------------------------------------
out = Path("data/v3")
out.mkdir(parents=True, exist_ok=True)
df.to_parquet(out / "features_preview.parquet", index=False)
print(f"\n[5] Saved {len(df):,} rows -> {out/'features_preview.parquet'}")
