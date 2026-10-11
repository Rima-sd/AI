"""
Primark-style Vendor Performance Scorecard (Streamlit).

Run:  streamlit run app.py
Demo data is generated automatically on first launch (data/raw). Scoring is fully deterministic:
no LLM is used to calculate scores, risk bands or alternate-vendor ranks.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import generate_data as gd  # noqa: E402
import pipeline as pl  # noqa: E402

st.set_page_config(page_title="Vendor Performance Scorecard", page_icon="📦", layout="wide")

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
LOG_PATH = ROOT / "data" / "actions_log.csv"
BAND_ICON = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}
STATUS_ICON = {"Sustained underperformance": "🔴 Sustained underperformance", "Early warning": "🟠 Early warning",
               "Below threshold": "🟡 Below threshold", "OK": "🟢 OK"}
LABELS = pl.METRIC_LABELS


# ------------------------------------------------------------------------------ data loading
@st.cache_data(show_spinner="Loading data and scoring vendors...")
def load_everything(cfg_text: str, seed: int):
    cfg = json.loads(cfg_text)
    if not (RAW_DIR / "po_lines.csv").exists():
        gd.generate(RAW_DIR, {"seed": seed})
    raw = pl.load_raw(RAW_DIR)
    return raw, pl.run_pipeline(raw, cfg, OUT_DIR)


def vendor_level(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Roll vendor x category rows up to one row per vendor (PO-count weighted)."""
    cols = ["score", "on_time_pct", "fill_rate_pct", "defect_pct", "cost_variance_pct"]
    t = df.copy()
    for c in cols:
        t[c + "_w"] = t[c] * t["po_count"]
    agg = t.groupby(["vendor_id", "vendor_name", "tier", "country"]).agg(
        po_count=("po_count", "sum"), spend_gbp=("spend_gbp", "sum"), categories=("category", "nunique"),
        flagged_categories=("status", lambda s: int((s != "OK").sum())),
        **{c + "_w": (c + "_w", "sum") for c in cols}).reset_index()
    for c in cols:
        agg[c] = agg[c + "_w"] / agg["po_count"]
    agg["risk_band"] = pl.band_for(agg["score"], cfg)
    return agg.drop(columns=[c + "_w" for c in cols])


# ------------------------------------------------------------------------------ sidebar
cfg_path = ROOT / "config.json"
cfg_text = cfg_path.read_text()
cfg = json.loads(cfg_text)
if "seed" not in st.session_state:
    st.session_state["seed"] = gd.DEFAULT_SETTINGS["seed"]

raw, res = load_everything(cfg_text, st.session_state["seed"])
scored, po_all, monthly = res["scored"], res["po_all"], res["monthly"]
periods = sorted(scored["period"].unique())

st.sidebar.title("📦 Vendor Scorecard")
st.sidebar.caption(f"Scoring config v{cfg['config_version']} · deterministic, no LLM in scoring")
period = st.sidebar.selectbox("Period (promised month)", periods, index=len(periods) - 1)
f_cat = st.sidebar.multiselect("Category", sorted(scored["category"].unique()))
f_tier = st.sidebar.multiselect("Vendor tier", sorted(scored["tier"].unique()))
f_ctry = st.sidebar.multiselect("Market (vendor country)", sorted(scored["country"].unique()))
f_band = st.sidebar.multiselect("Risk band", ["High", "Medium", "Low"])
with st.sidebar.expander("Demo data"):
    seed_in = st.number_input("Random seed", value=int(st.session_state["seed"]), step=1)
    if st.button("Regenerate synthetic data"):
        gd.generate(RAW_DIR, {"seed": int(seed_in)})
        st.session_state["seed"] = int(seed_in)
        st.cache_data.clear()
        st.rerun()
    st.caption(f"{len(res['po_all']):,} active PO lines · {scored['vendor_id'].nunique()} vendors · "
               f"{scored['category'].nunique()} categories · {len(periods)} months")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if f_cat:
        df = df[df["category"].isin(f_cat)]
    if f_tier:
        df = df[df["tier"].isin(f_tier)]
    if f_ctry:
        df = df[df["country"].isin(f_ctry)]
    if f_band:
        df = df[df["risk_band"].isin(f_band)]
    return df


cur_all = scored[scored["period"] == period]
cur = apply_filters(cur_all)
actions = pl.build_action_list(scored, cfg, period)

st.title("Vendor Performance Scorecard")
st.caption("Transparent, explainable vendor scoring from PO, delivery, inspection and invoice data. "
           "Demo uses synthetic data.")

tabs = st.tabs(["🏆 Leaderboard", "🔎 Vendor drill-down", "⚠️ Early warnings", "🔁 Alternate vendors",
                "🎚️ Scenario weights", "🧪 Data quality", "📝 Action log"])
tab_lb, tab_dd, tab_ew, tab_alt, tab_sc, tab_dq, tab_log = tabs

# ------------------------------------------------------------------------------ leaderboard
with tab_lb:
    c = st.columns(5)
    c[0].metric("Vendor-categories scored", f"{len(cur):,}")
    c[1].metric("High risk", int((cur["risk_band"] == "High").sum()))
    c[2].metric("Early warnings", int((cur["status"] == "Early warning").sum()))
    c[3].metric("Avg on-time", f"{cur['on_time_pct'].mean():.1f}%" if len(cur) else "-")
    c[4].metric("Spend (GBP)", f"£{cur['spend_gbp'].sum() / 1e6:.1f}m")

    level = st.radio("View", ["Vendor × category", "Vendor overall"], horizontal=True)
    if cur.empty:
        st.info("No rows match the current filters.")
    elif level == "Vendor × category":
        t = cur.sort_values("score").copy()
        t["Risk"] = t["risk_band"].map(BAND_ICON)
        t["Status"] = t["status"].map(STATUS_ICON)
        t["Cat. percentile"] = cur_all.groupby("category")["score"].rank(pct=True).reindex(t.index) * 100
        show = t[["vendor_name", "category", "tier", "country", "score", "Risk", "Status", "confidence",
                  "po_count", "on_time_pct", "fill_rate_pct", "defect_pct", "cost_variance_pct",
                  "trend_label", "Cat. percentile"]].rename(columns={
            "vendor_name": "Vendor", "category": "Category", "tier": "Tier", "country": "Market", "score": "Score",
            "confidence": "Data confidence", "po_count": "POs", "on_time_pct": "On-time %",
            "fill_rate_pct": "Fill rate %", "defect_pct": "Defect %", "cost_variance_pct": "Cost var %",
            "trend_label": "Trend"})
        st.dataframe(show.round(1), hide_index=True, column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f")})
        st.caption("Sorted worst-first. Category percentile compares a vendor with others supplying the same category. "
                   "Low-volume scores are shrunk toward the category median and carry a low data-confidence label.")
        st.bar_chart(pd.crosstab(cur["category"], cur["risk_band"]).reindex(
            columns=[b for b in ["High", "Medium", "Low"] if b in set(cur["risk_band"])]))
    else:
        v = vendor_level(cur, cfg).sort_values("score")
        v["Risk"] = v["risk_band"].map(BAND_ICON)
        show = v[["vendor_name", "tier", "country", "score", "Risk", "categories", "flagged_categories", "po_count",
                  "on_time_pct", "fill_rate_pct", "defect_pct", "spend_gbp"]].rename(columns={
            "vendor_name": "Vendor", "tier": "Tier", "country": "Market", "score": "Score",
            "categories": "Categories", "flagged_categories": "Flagged categories", "po_count": "POs",
            "on_time_pct": "On-time %", "fill_rate_pct": "Fill rate %", "defect_pct": "Defect %",
            "spend_gbp": "Spend GBP"})
        st.dataframe(show.round(1), hide_index=True, column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f")})

# ------------------------------------------------------------------------------ drill-down
with tab_dd:
    names = (scored[["vendor_id", "vendor_name"]].drop_duplicates().sort_values("vendor_name"))
    label = {r.vendor_id: f"{r.vendor_name} ({r.vendor_id})" for r in names.itertuples()}
    vids = list(label)
    default_v = actions.iloc[0]["vendor_id"] if len(actions) else vids[0]
    vid = st.selectbox("Vendor", vids, index=vids.index(default_v), format_func=label.get, key="dd_vendor")
    cats = sorted(scored.loc[scored["vendor_id"] == vid, "category"].unique())
    cat = st.selectbox("Category", cats, key="dd_cat")
    hist = scored[(scored["vendor_id"] == vid) & (scored["category"] == cat) & (scored["period"] <= period)]
    hist = hist.sort_values("period")
    if hist.empty:
        st.info("No history for this vendor-category up to the selected period.")
    else:
        row = hist.iloc[-1]
        if row["period"] != period:
            st.warning(f"No orders in {period}; showing the latest period with data ({row['period']}).")
        prev = hist.iloc[-2]["score"] if len(hist) > 1 else np.nan
        c = st.columns(5)
        c[0].metric("Score", f"{row['score']:.0f}", None if np.isnan(prev) else f"{row['score'] - prev:+.1f} vs prior month")
        c[1].metric("Risk band", BAND_ICON[row["risk_band"]])
        c[2].metric("Status", row["status"])
        c[3].metric("Data confidence", row["confidence"])
        c[4].metric("PO lines", int(row["po_count"]))

        st.subheader("Why this rating?")
        st.markdown(
            f"**{row['vendor_name']} · {cat}** scores **{row['score']:.0f}** ({row['risk_band']} risk). "
            f"Biggest drivers of lost points: **{row['driver_1']}** and **{row['driver_2']}**.")
        if abs(row["score"] - row["score_raw"]) > 0.5:
            st.caption(f"Raw score {row['score_raw']:.1f} adjusted to {row['score']:.1f} for low PO volume "
                       f"(shrinkage toward the category median).")
        if row["capped_low_confidence"]:
            st.caption("High-risk band capped to Medium because data confidence is low.")
        if row["disruption_flag"]:
            st.caption(f"Disruption in this period: {row['disruption']} (affects vendors in this market).")
        if row["status"] != "OK":
            st.info(f"Suggested corrective action: {cfg['corrective_actions'][row['top_driver_metric']]}")

        contrib = pd.DataFrame([{
            "Metric": LABELS[m], "Value": row[m], "Normalised (0-100)": row[f"norm_{m}"],
            "Weight": row[f"weight_{m}"], "Points lost": row[f"loss_{m}"]} for m in pl.METRICS]
        ).sort_values("Points lost", ascending=False)
        left, right = st.columns([3, 2])
        left.dataframe(contrib.round(2), hide_index=True)
        right.bar_chart(contrib.set_index("Metric")["Points lost"])

        st.subheader("Trend")
        trend = hist.set_index("period")[["score", "score_roll3"]].rename(
            columns={"score": "Score", "score_roll3": "3-month weighted avg"})
        trend["Underperformance threshold"] = cfg["early_warning"]["underperf_threshold"]
        st.line_chart(trend)
        mcols = st.columns(3)
        for i, m in enumerate(pl.METRICS):
            with mcols[i % 3]:
                st.caption(LABELS[m])
                st.line_chart(hist.set_index("period")[m])

        st.subheader("Recent late deliveries")
        pl_rows = po_all[(po_all["vendor_id"] == vid) & (po_all["category"] == cat) & po_all["valid"]
                         & (po_all["days_late"] > 0) & (po_all["period"] <= period)]
        pl_rows = pl_rows.sort_values("promised_date", ascending=False).head(10)
        if pl_rows.empty:
            st.caption("No late deliveries in the data up to this period.")
        else:
            out = pl_rows[["po_line_id", "sku_id", "promised_date", "receipt_date", "days_late", "ordered_qty",
                           "received_qty", "defect_rate", "cost_variance_pct"]].copy()
            out["promised_date"] = out["promised_date"].dt.date
            out["receipt_date"] = out["receipt_date"].dt.date
            out["defect_rate"] = (out["defect_rate"] * 100).round(1)
            st.dataframe(out.round(2), hide_index=True)
        st.caption(f"Data completeness for this month: {row['data_completeness'] * 100:.0f}% of PO lines fully "
                   f"matched to delivery, inspection and invoice.")

# ------------------------------------------------------------------------------ early warnings
with tab_ew:
    ew = cfg["early_warning"]
    st.markdown(
        f"**Early warning** = score is trending down (≥ {ew['drop_points']:.0f} points below its prior baseline and "
        f"falling ≥ {ew['slope_points_per_month']:.0f} pts/month), has slipped below {ew['warn_below_score']:.0f}, "
        f"but is not yet underperforming (< {ew['underperf_threshold']:.0f}). **Sustained underperformance** = below "
        f"{ew['underperf_threshold']:.0f} for {ew['sustained_months']}+ consecutive months. Only data up to the selected "
        f"period is used.")
    a = apply_filters(actions) if len(actions) else actions
    st.subheader(f"Prioritised corrective-action list · {period}")
    if a.empty:
        st.success("No flagged vendor-categories for the current filters.")
    else:
        show = a.copy()
        show["status"] = show["status"].map(STATUS_ICON)
        show = show.rename(columns={"priority": "Priority", "status": "Status", "vendor_name": "Vendor",
                                    "category": "Category", "tier": "Tier", "country": "Market", "score": "Score",
                                    "confidence": "Data confidence", "driver_1": "Main driver",
                                    "driver_2": "Second driver", "spend_3m_gbp": "Spend 3m GBP",
                                    "recommended_action": "Recommended action"}).drop(columns=["vendor_id", "risk_band"])
        st.dataframe(show.round(1), hide_index=True)
        st.download_button("Download action list (CSV)", a.to_csv(index=False).encode(),
                           file_name=f"corrective_actions_{period}.csv", mime="text/csv")

    st.subheader("Back-test of the early-warning rule")
    bt = res["backtest"]
    st.caption(
        f"Evaluated on {bt['rows_evaluated']:,} vendor-category-months that were not yet underperforming. "
        f"Outcome = enters sustained underperformance within {ew['horizon_months']} months "
        f"(base rate {bt['base_rate'] * 100:.1f}%). Baseline = static rule flagging any score between "
        f"{ew['underperf_threshold']:.0f} and {ew['static_watch_threshold']:.0f}.")
    cmp_df = pd.DataFrame({
        "Metric": ["Rows flagged", "Precision", "Recall", "Events caught early", "Median lead time (months)"],
        "Early-warning rule": [bt["early_warning"]["flagged"], f"{bt['early_warning']['precision'] * 100:.1f}%",
                               f"{bt['early_warning']['recall'] * 100:.1f}%",
                               f"{bt['events_caught_early']} / {bt['events']}", f"{bt['median_lead_months']:.1f}"],
        "Static threshold rule": [bt["static_rule"]["flagged"], f"{bt['static_rule']['precision'] * 100:.1f}%",
                                  f"{bt['static_rule']['recall'] * 100:.1f}%",
                                  f"{bt['events_caught_static']} / {bt['events']}",
                                  f"{bt['median_lead_months_static']:.1f}"]})
    st.dataframe(cmp_df, hide_index=True)
    st.caption("Reading this honestly: warnings are review prompts, not verdicts. Gradual deterioration is detected "
               "well ahead of the threshold, while volatile vendors with random bad months are largely "
               "unpredictable from trend alone. Calibrate thresholds on real history before relying on them.")

# ------------------------------------------------------------------------------ alternates
with tab_alt:
    st.markdown("Alternates are limited to vendors supplying **the same category**, with acceptable recent "
                "performance, sufficient data confidence and spare capacity. Every factor is shown.")
    flagged_vendor = actions.iloc[0] if len(actions) else None
    vids2 = list(label)
    av = st.selectbox("Vendor to replace / de-risk", vids2,
                      index=vids2.index(flagged_vendor["vendor_id"]) if flagged_vendor is not None else 0,
                      format_func=label.get, key="alt_vendor")
    acats = sorted(scored.loc[scored["vendor_id"] == av, "category"].unique())
    default_c = acats.index(flagged_vendor["category"]) if (
        flagged_vendor is not None and flagged_vendor["vendor_id"] == av and flagged_vendor["category"] in acats) else 0
    ac = st.selectbox("Category", acats, index=default_c, key="alt_cat")
    alt = pl.rank_alternates(scored, raw, cfg, av, ac, period)
    if alt.empty:
        st.info("No other vendors supply this category in the data.")
    else:
        w = cfg["alternates"]["weights"]
        st.caption(f"Ranking weights: performance {w['performance']:.0%}, cost {w['cost']:.0%}, "
                   f"capacity {w['capacity']:.0%}, lead-time consistency {w['lead_time']:.0%}.")
        ok = alt[alt["eligible"]]
        if ok.empty:
            st.warning("No eligible alternates under the current rules. See exclusions below.")
        else:
            show = ok[["rank", "vendor_name", "tier", "country", "avg_score", "on_time_pct", "defect_pct",
                       "cost_index", "lead_time_index", "capacity_headroom_x", "rank_score", "why"]].rename(columns={
                "rank": "Rank", "vendor_name": "Vendor", "tier": "Tier", "country": "Market",
                "avg_score": "3-mo score", "on_time_pct": "On-time %", "defect_pct": "Defect %",
                "cost_index": "Cost index", "lead_time_index": "Lead-time index",
                "capacity_headroom_x": "Spare capacity (x required)", "rank_score": "Rank score", "why": "Why"})
            st.dataframe(show.round(1), hide_index=True)
        with st.expander(f"Excluded vendors ({int((~alt['eligible']).sum())})"):
            ex = alt[~alt["eligible"]][["vendor_name", "tier", "country", "avg_score", "excluded_because"]]
            st.dataframe(ex.rename(columns={"vendor_name": "Vendor", "tier": "Tier", "country": "Market",
                                            "avg_score": "3-mo score", "excluded_because": "Reason"}).round(1),
                         hide_index=True)

# ------------------------------------------------------------------------------ scenario
with tab_sc:
    st.markdown("Adjust metric weights to see how scores, risk bands and rankings change. Nothing is saved; "
                "approved weights are changed in `config.json` under version control.")
    cols = st.columns(len(pl.METRICS))
    weights = {}
    for col, m in zip(cols, pl.METRICS):
        weights[m] = col.slider(LABELS[m], 0, 100, int(round(cfg["weights"][m] * 100)), key=f"w_{m}")
    total = sum(weights.values())
    if total == 0:
        st.error("At least one weight must be above zero.")
    else:
        norm_w = {m: v / total for m, v in weights.items()}
        st.caption("Effective weights: " + ", ".join(f"{LABELS[m]} {v:.0%}" for m, v in norm_w.items()))
        scn = pl.add_trends(pl.score_monthly(monthly, cfg, norm_w), cfg)
        b = cur_all[["vendor_id", "vendor_name", "category", "score", "risk_band"]].copy()
        s2 = scn[scn["period"] == period][["vendor_id", "category", "score", "risk_band", "status"]]
        m2 = b.merge(s2, on=["vendor_id", "category"], suffixes=("_base", "_scenario"))
        m2["rank_base"] = m2["score_base"].rank(ascending=False, method="first").astype(int)
        m2["rank_scenario"] = m2["score_scenario"].rank(ascending=False, method="first").astype(int)
        m2["rank_change"] = m2["rank_base"] - m2["rank_scenario"]
        m2["score_change"] = m2["score_scenario"] - m2["score_base"]
        changed = m2[m2["risk_band_base"] != m2["risk_band_scenario"]]
        c = st.columns(3)
        c[0].metric("Risk-band changes", len(changed))
        c[1].metric("Largest rank move", int(m2["rank_change"].abs().max()))
        top_b, top_s = set(m2.nsmallest(10, "rank_base")["vendor_id"]), set(m2.nsmallest(10, "rank_scenario")["vendor_id"])
        c[2].metric("Top-10 vendors that changed", len(top_b - top_s))
        view = m2.reindex(m2["rank_change"].abs().sort_values(ascending=False).index).head(25)
        st.dataframe(view[["vendor_name", "category", "score_base", "score_scenario", "score_change", "rank_base",
                           "rank_scenario", "rank_change", "risk_band_base", "risk_band_scenario"]].rename(columns={
            "vendor_name": "Vendor", "category": "Category", "score_base": "Score (current)",
            "score_scenario": "Score (scenario)", "score_change": "Δ score", "rank_base": "Rank (current)",
            "rank_scenario": "Rank (scenario)", "rank_change": "Rank move (+ = up)",
            "risk_band_base": "Band (current)", "risk_band_scenario": "Band (scenario)"}).round(1), hide_index=True)
        st.download_button("Download these weights as JSON", json.dumps(norm_w, indent=2).encode(),
                           file_name="scenario_weights.json", mime="application/json")

# ------------------------------------------------------------------------------ data quality
with tab_dq:
    dq = res["dq"]
    c = st.columns(4)
    c[0].metric("Raw PO lines", f"{dq['po_lines_raw']:,}")
    c[1].metric("Duplicates removed", f"{dq['duplicate_po_lines_removed']:,}")
    c[2].metric("Cancelled (separate)", f"{dq['cancelled_po_lines']:,}")
    c[3].metric("Quarantined", f"{dq['quarantined_lines']:,}")
    c = st.columns(4)
    c[0].metric("PO→delivery match", f"{dq['match_rate_delivery_pct']:.1f}%")
    c[1].metric("PO→inspection match", f"{dq['match_rate_inspection_pct']:.1f}%")
    c[2].metric("PO→invoice match", f"{dq['match_rate_invoice_pct']:.1f}%")
    c[3].metric("Fully matched", f"{dq['fully_matched_pct']:.1f}%")
    st.caption(f"Orphan records ignored: {dq['orphan_deliveries']} deliveries, {dq['orphan_invoices']} invoices, "
               f"{dq['duplicate_deliveries_removed']} duplicate deliveries removed.")
    st.subheader("Quarantine reasons")
    st.dataframe(pd.DataFrame(list(dq["quarantine_breakdown"].items()), columns=["Reason", "PO lines"]),
                 hide_index=True)
    st.subheader("Quantity reconciliation (valid lines)")
    v = po_all[po_all["valid"]]
    rec = pd.DataFrame({"Stage": ["Ordered", "Received", "Invoiced"],
                        "Units": [v["ordered_qty"].sum(), v["received_qty"].sum(), v["invoiced_qty"].sum()]})
    st.bar_chart(rec.set_index("Stage"))
    gap = v["qty_recon_gap"].dropna()
    st.caption(f"{int((gap != 0).sum()):,} invoiced lines ({(gap != 0).mean() * 100:.1f}%) bill a different quantity "
               f"than was received. These are surfaced for AP follow-up, not silently corrected.")
    st.subheader("Data completeness over time")
    gsum = monthly.groupby("period")[["matched_lines", "total_lines"]].sum()
    st.line_chart((gsum["matched_lines"] / gsum["total_lines"]).rename("Fully matched share"))
    with st.expander("Quarantined records (sample)"):
        st.dataframe(res["quarantine"].head(200), hide_index=True)

# ------------------------------------------------------------------------------ action log
with tab_log:
    st.markdown("Record the buyer's decision on a risk flag. In this demo decisions are appended to "
                "`data/actions_log.csv`; production would write to an audited database with role-based access.")
    if actions.empty:
        st.info("Nothing flagged in this period.")
    else:
        opts = {f"{r.vendor_name} · {r.category} · {r.status}": i for i, r in actions.iterrows()}
        pick = st.selectbox("Flagged vendor-category", list(opts))
        r = actions.loc[opts[pick]]
        decision = st.radio("Decision", ["Confirm", "Dismiss", "Escalate"], horizontal=True)
        note = st.text_area("Corrective action / note", value=r["recommended_action"] if decision != "Dismiss" else "")
        reviewer = st.text_input("Reviewer")
        if st.button("Save decision"):
            entry = pd.DataFrame([{
                "timestamp": datetime.now().isoformat(timespec="seconds"), "period": period,
                "vendor_id": r["vendor_id"], "vendor_name": r["vendor_name"], "category": r["category"],
                "status": r["status"], "score": round(float(r["score"]), 1), "decision": decision, "note": note,
                "reviewer": reviewer, "config_version": cfg["config_version"]}])
            entry.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)
            st.success("Decision recorded.")
    if LOG_PATH.exists():
        st.subheader("Decision log")
        st.dataframe(pd.read_csv(LOG_PATH).sort_values("timestamp", ascending=False), hide_index=True)
