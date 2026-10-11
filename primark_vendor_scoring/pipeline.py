"""
Vendor Performance Scoring - deterministic pipeline (no LLM, no black box).

    raw tables -> standardise & match -> PO-line KPIs -> vendor x category x month features
               -> configurable weighted score + risk band -> trend / early-warning flags
               -> alternate-vendor ranking -> prioritised corrective-action list

Every step is a plain function so it can be unit-tested and re-run reproducibly:
identical raw data + identical config => identical scores (NFR8).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["on_time_pct", "fill_rate_pct", "defect_pct", "cost_variance_abs_pct", "lead_time_cv"]
METRIC_LABELS = {
    "on_time_pct": "On-time delivery",
    "fill_rate_pct": "Fill rate",
    "defect_pct": "Defect rate",
    "cost_variance_abs_pct": "Cost variance",
    "lead_time_cv": "Lead-time consistency",
}
KEYS = ["vendor_id", "category", "period"]
VC = ["vendor_id", "category"]


# ----------------------------------------------------------------------------- config / io
def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else Path(__file__).parent / "config.json"
    return json.loads(path.read_text())


def _std_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper().astype(object)


def load_raw(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(raw_dir)
    return {
        "po_lines": pd.read_csv(d / "po_lines.csv", parse_dates=["order_date", "promised_date"]),
        "deliveries": pd.read_csv(d / "deliveries.csv", parse_dates=["ship_date", "receipt_date"]),
        "inspections": pd.read_csv(d / "inspections.csv", parse_dates=["inspection_date"]),
        "invoices": pd.read_csv(d / "invoices.csv", parse_dates=["invoice_date"]),
        "vendors": pd.read_csv(d / "vendors.csv"),
        "vendor_categories": pd.read_csv(d / "vendor_categories.csv"),
        "fx_rates": pd.read_csv(d / "fx_rates.csv"),
        "disruption_events": pd.read_csv(d / "disruption_events.csv"),
    }


# ----------------------------------------------------------------------------- match & clean
def build_po_level(raw: dict, cfg: dict):
    """Standardise IDs, de-duplicate, match PO line -> delivery/inspection/invoice, compute KPIs.

    Returns (po_all, quarantine, cancelled, dq_report). `po_all` holds every active PO line with
    flags; KPI columns are only populated for lines that pass validation.
    """
    dq: dict = {}
    po = raw["po_lines"].copy()
    dq["po_lines_raw"] = len(po)
    po["vendor_id"] = _std_id(po["vendor_id"])
    po["po_line_id"] = _std_id(po["po_line_id"])

    dup = po.duplicated("po_line_id", keep="first")
    dq["duplicate_po_lines_removed"] = int(dup.sum())
    po = po[~dup]

    is_cancel = po["status"].str.lower().eq("cancelled")
    cancelled = po[is_cancel].copy()
    po = po[~is_cancel].copy()
    dq["cancelled_po_lines"] = len(cancelled)
    dq["active_po_lines"] = len(po)
    ids = set(po["po_line_id"])

    dl = raw["deliveries"].copy()
    dl["po_line_id"] = _std_id(dl["po_line_id"])
    dq["orphan_deliveries"] = int((~dl["po_line_id"].isin(ids | set(cancelled["po_line_id"]))).sum())
    dq["duplicate_deliveries_removed"] = int(dl[dl["po_line_id"].isin(ids)].duplicated("po_line_id").sum())
    dl = dl[dl["po_line_id"].isin(ids)].drop_duplicates("po_line_id")

    ins = raw["inspections"].copy()
    ins["po_line_id"] = _std_id(ins["po_line_id"])
    dq["orphan_inspections"] = int((~ins["po_line_id"].isin(ids)).sum())
    ins = ins[ins["po_line_id"].isin(ids)].drop_duplicates("po_line_id")

    inv = raw["invoices"].copy()
    inv["po_line_id"] = _std_id(inv["po_line_id"])
    dq["orphan_invoices"] = int((~inv["po_line_id"].isin(ids | set(cancelled["po_line_id"]))).sum())
    inv = inv[inv["po_line_id"].isin(ids)].drop_duplicates("po_line_id")
    inv = inv.rename(columns={"currency": "invoice_currency", "unit_cost": "invoice_unit_cost"})

    df = (po.merge(dl[["po_line_id", "ship_date", "receipt_date", "received_qty"]], on="po_line_id", how="left")
            .merge(ins[["po_line_id", "inspection_date", "sample_size", "defects", "result"]], on="po_line_id", how="left")
            .merge(inv[["po_line_id", "invoice_date", "invoiced_qty", "invoice_unit_cost", "invoice_currency"]],
                   on="po_line_id", how="left"))
    df["has_delivery"] = df["receipt_date"].notna()
    df["has_inspection"] = df["sample_size"].notna()
    df["has_invoice"] = df["invoiced_qty"].notna()

    # ---- validation rules -> quarantine
    seq_bad = df["has_delivery"] & (
        (df["ship_date"] < df["order_date"]) | (df["receipt_date"] < df["ship_date"])
        | (df["has_invoice"] & (df["invoice_date"] < df["receipt_date"])))
    val_bad = (df["ordered_qty"] <= 0) | (df["received_qty"] < 0) | (df["defects"] > df["sample_size"])
    reason = pd.Series("", index=df.index)
    reason = reason.mask(~df["has_delivery"], "no_delivery_record")
    reason = reason.mask(seq_bad, "date_sequence_invalid")
    reason = reason.mask(val_bad.fillna(False) & (reason == ""), "invalid_value")
    df["quarantine_reason"] = reason
    df["valid"] = reason == ""

    dq["match_rate_delivery_pct"] = round(float(100 * df["has_delivery"].mean()), 2)
    dq["match_rate_inspection_pct"] = round(float(100 * df["has_inspection"].mean()), 2)
    dq["match_rate_invoice_pct"] = round(float(100 * df["has_invoice"].mean()), 2)
    df["fully_matched"] = df["valid"] & df["has_inspection"] & df["has_invoice"]
    dq["fully_matched_pct"] = round(float(100 * df["fully_matched"].mean()), 2)
    dq["quarantined_lines"] = int((~df["valid"]).sum())
    dq["quarantine_breakdown"] = df.loc[~df["valid"], "quarantine_reason"].value_counts().to_dict()

    # ---- PO-line KPIs (only valid rows)
    v = df["valid"]
    grace = pd.Timedelta(days=cfg.get("on_time_grace_days", 0))
    df["period"] = df["promised_date"].dt.strftime("%Y-%m")
    df["on_time"] = np.where(v, (df["receipt_date"] <= df["promised_date"] + grace).astype(float), np.nan)
    df["days_late"] = np.where(v, (df["receipt_date"] - df["promised_date"]).dt.days.clip(lower=0), np.nan)
    df["received_capped"] = np.where(v, np.minimum(df["received_qty"], df["ordered_qty"]), np.nan)
    df["fill_rate"] = df["received_capped"] / df["ordered_qty"]
    df["defect_rate"] = np.where(v & df["has_inspection"], df["defects"] / df["sample_size"], np.nan)
    df["cost_variance_pct"] = np.where(
        v & df["has_invoice"], (df["invoice_unit_cost"] - df["agreed_unit_cost"]) / df["agreed_unit_cost"] * 100, np.nan)
    df["lead_time_days"] = np.where(v, (df["receipt_date"] - df["order_date"]).dt.days, np.nan)
    df["qty_recon_gap"] = np.where(v & df["has_invoice"], df["invoiced_qty"] - df["received_qty"], np.nan)

    # spend in reporting currency (GBP) using the promised-month FX rate
    fx = raw["fx_rates"].rename(columns={"month": "period", "rate_to_gbp": "fx_rate"})
    df = df.merge(fx, on=["period", "currency"], how="left")
    df["fx_rate"] = df["fx_rate"].fillna(1.0)
    qty = df["invoiced_qty"].fillna(df["received_qty"])
    unit = df["invoice_unit_cost"].fillna(df["agreed_unit_cost"])
    df["spend_gbp"] = np.where(v, qty * unit * df["fx_rate"], np.nan)

    quarantine = df.loc[~df["valid"], ["po_line_id", "vendor_id", "category", "period", "quarantine_reason"]].copy()
    return df, quarantine, cancelled, dq


# ----------------------------------------------------------------------------- aggregate
def aggregate_monthly(po_all: pd.DataFrame, cancelled: pd.DataFrame, raw: dict, cfg: dict) -> pd.DataFrame:
    v = po_all[po_all["valid"]].copy()
    v["cost_var_abs"] = v["cost_variance_pct"].abs()
    g = v.groupby(KEYS)
    m = g.agg(po_count=("po_line_id", "size"), ordered_units=("ordered_qty", "sum"),
              received_units=("received_qty", "sum"), fill_num=("received_capped", "sum"),
              on_time=("on_time", "mean"), avg_days_late=("days_late", "mean"),
              lead_time_mean=("lead_time_days", "mean"), lead_time_std=("lead_time_days", "std"),
              defects=("defects", "sum"), sample=("sample_size", "sum"),
              cost_variance_pct=("cost_variance_pct", "mean"), cost_variance_abs_pct=("cost_var_abs", "mean"),
              spend_gbp=("spend_gbp", "sum")).reset_index()
    m["on_time_pct"] = m["on_time"] * 100
    m["fill_rate_pct"] = m["fill_num"] / m["ordered_units"] * 100
    m["defect_pct"] = np.where(m["sample"] > 0, m["defects"] / m["sample"].replace(0, np.nan) * 100, np.nan)
    m["lead_time_cv"] = m["lead_time_std"] / m["lead_time_mean"]
    min_n = cfg["confidence"]["min_po_for_lead_time_std"]
    m.loc[m["po_count"] < min_n, "lead_time_cv"] = np.nan
    cat_cv = m.groupby("category")["lead_time_cv"].transform("median")
    m["lead_time_cv_imputed"] = m["lead_time_cv"].isna()
    m["lead_time_cv"] = m["lead_time_cv"].fillna(cat_cv)

    tot = po_all.groupby(KEYS).agg(total_lines=("po_line_id", "size"), matched_lines=("fully_matched", "sum")).reset_index()
    m = m.merge(tot, on=KEYS, how="left")
    m["data_completeness"] = m["matched_lines"] / m["total_lines"]
    canc = cancelled.assign(period=cancelled["promised_date"].dt.strftime("%Y-%m")).groupby(KEYS).size()
    m = m.merge(canc.rename("cancelled_lines").reset_index(), on=KEYS, how="left")
    m["cancelled_lines"] = m["cancelled_lines"].fillna(0).astype(int)

    vendors = raw["vendors"][["vendor_id", "vendor_name", "tier", "country"]]
    m = m.merge(vendors, on="vendor_id", how="left")
    ev = raw["disruption_events"]
    flags = []
    for r in m[["country", "period"]].itertuples(index=False):
        hit = ev[(ev["start_month"] <= r.period) & (ev["end_month"] >= r.period)
                 & ev["countries"].str.split(";").apply(lambda cs: r.country in cs)]
        flags.append(hit["event"].iloc[0] if len(hit) else "")
    m["disruption"] = flags
    m["disruption_flag"] = m["disruption"] != ""
    return m.drop(columns=["on_time", "fill_num", "defects", "sample", "lead_time_std"]).sort_values(KEYS).reset_index(drop=True)


# ----------------------------------------------------------------------------- scoring
def band_for(score: pd.Series, cfg: dict) -> pd.Series:
    rb = cfg["risk_bands"]
    return pd.Series(np.select([score >= rb["low_min"], score >= rb["medium_min"]], ["Low", "Medium"], "High"),
                     index=score.index)


def score_monthly(monthly: pd.DataFrame, cfg: dict, weights: dict | None = None) -> pd.DataFrame:
    """Normalise metrics to 0-100, apply weights, shrink low-volume scores, assign risk bands."""
    df = monthly.copy()
    w = {k: float((weights or cfg["weights"])[k]) for k in METRICS}
    for mt in METRICS:
        b = cfg["metric_bounds"][mt]
        df[f"norm_{mt}"] = ((df[mt] - b["floor"]) / (b["target"] - b["floor"])).clip(0, 1) * 100
    norm = df[[f"norm_{mt}" for mt in METRICS]].to_numpy(dtype=float)
    avail = ~np.isnan(norm)
    w_eff = avail * np.array([w[mt] for mt in METRICS])
    w_eff = w_eff / np.where(w_eff.sum(axis=1, keepdims=True) == 0, 1, w_eff.sum(axis=1, keepdims=True))
    df["score_raw"] = np.nansum(np.where(avail, norm, 0) * w_eff, axis=1)
    for i, mt in enumerate(METRICS):
        df[f"loss_{mt}"] = np.where(avail[:, i], (100 - np.nan_to_num(norm[:, i])) * w_eff[:, i], 0.0)
        df[f"weight_{mt}"] = w_eff[:, i]

    # low-volume protection: shrink toward the category-month median
    c = cfg["confidence"]
    prior = df.groupby(["category", "period"])["score_raw"].transform("median")
    k, n = c["shrinkage_k"], df["po_count"]
    df["score"] = (n * df["score_raw"] + k * prior) / (n + k)

    vol = np.minimum(1.0, n / c["min_po_full_confidence"])
    df["confidence_score"] = 0.6 * vol + 0.4 * df["data_completeness"].fillna(0)
    df["confidence"] = np.select([df["confidence_score"] >= c["high_min"], df["confidence_score"] >= c["medium_min"]],
                                 ["High", "Medium"], "Low")
    band = band_for(df["score"], cfg)
    capped = (band == "High") & (df["confidence"] == "Low")
    df["risk_band"] = band.mask(capped, "Medium")
    df["capped_low_confidence"] = capped

    loss_cols = [f"loss_{mt}" for mt in METRICS]
    losses = df[loss_cols].to_numpy()
    order = np.argsort(-losses, axis=1)

    def driver(rank):
        idx = order[:, rank]
        return [f"{METRIC_LABELS[METRICS[j]]} (-{losses[i, j]:.1f})" for i, j in enumerate(idx)]

    df["driver_1"], df["driver_2"] = driver(0), driver(1)
    df["top_driver_metric"] = [METRICS[j] for j in order[:, 0]]
    return df


def add_trends(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Early-warning logic. Every feature at month t uses data up to and including month t only.

    Trend features are PO-volume-weighted so a month with one PO line cannot dominate the signal.
    Early warning = persistent deterioration AND the score has slipped out of the Low-risk zone
    (below `warn_below_score`) but is not yet underperforming.
    """
    ew = cfg["early_warning"]
    df = df.sort_values(VC + ["period"]).reset_index(drop=True)
    df["_sw"] = df["score"] * df["po_count"]
    grp = df.groupby(VC, sort=False)

    def wmean(window, shift=0, min_periods=1):
        num = grp["_sw"].transform(lambda s: s.shift(shift).rolling(window, min_periods=min_periods).sum())
        den = grp["po_count"].transform(lambda s: s.shift(shift).rolling(window, min_periods=min_periods).sum())
        return num / den

    df["score_roll3"] = wmean(3)
    df["score_roll6"] = wmean(6)
    smooth = wmean(2)
    baseline = wmean(4, shift=2, min_periods=3)
    df["delta_vs_baseline"] = smooth - baseline
    df["trend_slope3"] = (smooth - wmean(2, shift=2)) / 2
    df = df.drop(columns="_sw")

    below = (df["score"] < ew["underperf_threshold"]).astype(int)
    df["months_below"] = below.groupby([df["vendor_id"], df["category"]], sort=False).transform(
        lambda s: s.groupby((s == 0).cumsum()).cumsum())

    raw_det = ((df["delta_vs_baseline"] <= -ew["drop_points"])
               & (df["trend_slope3"] <= -ew["slope_points_per_month"])).astype(int)
    # a single-month spike is noise; require the deterioration to persist for N consecutive months
    persist = raw_det.groupby([df["vendor_id"], df["category"]], sort=False).transform(
        lambda s: s.rolling(ew["persistence_months"], min_periods=ew["persistence_months"]).min())
    df["deteriorating"] = persist.fillna(0).astype(bool)
    df["sustained_underperf"] = df["months_below"] >= ew["sustained_months"]
    df["early_warning"] = (df["deteriorating"] & (df["score"] >= ew["underperf_threshold"])
                           & (df["score"] < ew["warn_below_score"]) & (df["confidence"] != "Low"))
    df["static_watch"] = (df["score"] < ew["static_watch_threshold"]) & (df["score"] >= ew["underperf_threshold"])
    df["status"] = np.select(
        [df["sustained_underperf"], df["early_warning"], df["score"] < ew["underperf_threshold"]],
        ["Sustained underperformance", "Early warning", "Below threshold"], "OK")
    df["trend_label"] = np.select([df["trend_slope3"] >= 3, df["trend_slope3"] <= -3], ["Improving", "Deteriorating"], "Stable")
    return df


def backtest(scored: pd.DataFrame, cfg: dict) -> dict:
    """Back-test early warnings against realised outcomes.

    Outcome (label) = the vendor-category ENTERS sustained underperformance (score below the threshold for
    `sustained_months` consecutive months) within the next `horizon_months` months. The label uses future data;
    the warning flag never does. Evaluated only on rows that are not yet underperforming, after a burn-in.
    Baseline = a static rule flagging any vendor already drifting into the 'watch' zone just above the threshold.
    """
    ew = cfg["early_warning"]
    thr, h, burn, sm = ew["underperf_threshold"], ew["horizon_months"], ew["burn_in_months"], ew["sustained_months"]
    rows, events = [], []
    for (vid, cat), g in scored.groupby(VC):
        g = g.sort_values("period").reset_index(drop=True)
        below = (g["score"] < thr).to_numpy()
        n = len(g)
        # run starts: first month of a below-threshold run that reaches `sm` months
        starts = [i for i in range(1, n - sm + 1) if below[i:i + sm].all() and not below[i - 1]]
        for i in range(burn, n - 1):
            if below[i]:
                continue
            outcome = any(i < st <= i + h for st in starts)
            rows.append((vid, cat, g.loc[i, "period"], outcome, bool(g.loc[i, "early_warning"]),
                         bool(g.loc[i, "static_watch"])))
        for st in starts:
            if st < burn:
                continue
            look = g.loc[max(0, st - 4):st - 1]
            ew_idx = look.index[look["early_warning"]].tolist()
            sw_idx = look.index[look["static_watch"]].tolist()
            events.append((vid, cat, g.loc[st, "period"],
                           (st - ew_idx[0]) if ew_idx else None, (st - sw_idx[0]) if sw_idx else None))
    bt = pd.DataFrame(rows, columns=["vendor_id", "category", "period", "outcome", "early_warning", "static_watch"])
    ev = pd.DataFrame(events, columns=["vendor_id", "category", "period", "ew_lead_months", "static_lead_months"])

    def prf(col):
        tp = int((bt[col] & bt["outcome"]).sum())
        fp = int((bt[col] & ~bt["outcome"]).sum())
        fn = int((~bt[col] & bt["outcome"]).sum())
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        return {"flagged": tp + fp, "precision": p, "recall": r}

    def med(col):
        return float(ev[col].median()) if ev[col].notna().any() else float("nan")

    return {
        "rows_evaluated": len(bt),
        "base_rate": float(bt["outcome"].mean()) if len(bt) else float("nan"),
        "early_warning": prf("early_warning"),
        "static_rule": prf("static_watch"),
        "events": len(ev),
        "events_caught_early": int(ev["ew_lead_months"].notna().sum()),
        "events_caught_static": int(ev["static_lead_months"].notna().sum()),
        "median_lead_months": med("ew_lead_months"),
        "median_lead_months_static": med("static_lead_months"),
        "events_table": ev,
    }


# ----------------------------------------------------------------------------- alternates & actions
def rank_alternates(scored: pd.DataFrame, raw: dict, cfg: dict, vendor_id: str, category: str,
                    period: str) -> pd.DataFrame:
    """Same-category, capacity-constrained alternates with transparent ranking factors."""
    a = cfg["alternates"]
    hist = scored[(scored["period"] <= period) & (scored["category"] == category)]
    cur = hist[hist["vendor_id"] == vendor_id].sort_values("period").tail(a["lookback_months"])
    required = max(cur["ordered_units"].mean(), 1.0) if len(cur) else 1.0
    caps = raw["vendor_categories"].set_index(VC)["capacity_units_month"]
    out = []
    for vid, g in hist.groupby("vendor_id"):
        if vid == vendor_id:
            continue
        g = g.sort_values("period").tail(a["lookback_months"])
        cap = float(caps.get((vid, category), np.nan))
        used = g["ordered_units"].mean()
        headroom = (cap - used) / required if cap == cap else np.nan
        last = g.iloc[-1]
        reasons = []
        if len(g) < a["min_months_data"]:
            reasons.append("insufficient history")
        if last["period"] != period:
            reasons.append("no recent orders")
        if g["score"].mean() < cfg["risk_bands"]["medium_min"]:
            reasons.append("high-risk performance")
        if last["confidence"] == "Low":
            reasons.append("low data confidence")
        if headroom != headroom or headroom < a["min_headroom_ratio"]:
            reasons.append("insufficient capacity headroom")
        out.append({
            "vendor_id": vid, "vendor_name": last["vendor_name"], "tier": last["tier"], "country": last["country"],
            "avg_score": g["score"].mean(), "cost_index": g["norm_cost_variance_abs_pct"].mean(),
            "lead_time_index": g["norm_lead_time_cv"].mean(),
            "capacity_headroom_x": headroom,
            "capacity_index": float(np.clip(headroom, 0, 1) * 100) if headroom == headroom else 0.0,
            "on_time_pct": g["on_time_pct"].mean(), "defect_pct": g["defect_pct"].mean(),
            "eligible": not reasons, "excluded_because": "; ".join(reasons),
        })
    res = pd.DataFrame(out)
    if res.empty:
        return res
    w = a["weights"]
    res["rank_score"] = (w["performance"] * res["avg_score"] + w["cost"] * res["cost_index"]
                         + w["capacity"] * res["capacity_index"] + w["lead_time"] * res["lead_time_index"])
    res = res.sort_values(["eligible", "rank_score"], ascending=[False, False]).reset_index(drop=True)
    res["rank"] = np.where(res["eligible"], np.arange(1, len(res) + 1), np.nan)
    res["why"] = [f"Score {r.avg_score:.0f}; on-time {r.on_time_pct:.0f}%; defects {r.defect_pct:.1f}%; "
                  f"spare capacity {r.capacity_headroom_x:.1f}x required volume" if r.eligible else "" for r in res.itertuples()]
    return res


def build_action_list(scored: pd.DataFrame, cfg: dict, period: str) -> pd.DataFrame:
    """Prioritised corrective-action list for the period (P1 sustained, P2 early warning, P3 below threshold)."""
    cur = scored[(scored["period"] == period) & (scored["status"] != "OK")].copy()
    if cur.empty:
        return cur
    recent = scored[scored["period"] <= period].sort_values("period").groupby(VC).tail(3)
    spend = recent.groupby(VC)["spend_gbp"].sum().rename("spend_3m_gbp").reset_index()
    cur = cur.merge(spend, on=VC, how="left")
    prio = {"Sustained underperformance": 1, "Early warning": 2, "Below threshold": 3}
    cur["priority"] = cur["status"].map(prio)
    cur["recommended_action"] = cur["top_driver_metric"].map(cfg["corrective_actions"])
    cols = ["priority", "status", "vendor_id", "vendor_name", "category", "tier", "country", "score", "risk_band",
            "confidence", "driver_1", "driver_2", "spend_3m_gbp", "recommended_action"]
    return cur.sort_values(["priority", "spend_3m_gbp"], ascending=[True, False])[cols].reset_index(drop=True)


# ----------------------------------------------------------------------------- orchestrator
def run_pipeline(raw: dict, cfg: dict, out_dir: str | Path | None = None) -> dict:
    po_all, quarantine, cancelled, dq = build_po_level(raw, cfg)
    monthly = aggregate_monthly(po_all, cancelled, raw, cfg)
    scored = add_trends(score_monthly(monthly, cfg), cfg)
    result = {"po_all": po_all, "quarantine": quarantine, "cancelled": cancelled, "dq": dq,
              "monthly": monthly, "scored": scored, "backtest": backtest(scored, cfg),
              "config_version": cfg["config_version"]}
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out / "vendor_category_month_scores.csv", index=False)
        quarantine.to_csv(out / "quarantine.csv", index=False)
        (out / "dq_report.json").write_text(json.dumps(dq, indent=2, default=str))
    return result


if __name__ == "__main__":
    cfg = load_config()
    res = run_pipeline(load_raw(Path(__file__).parent / "data" / "raw"), cfg, Path(__file__).parent / "data" / "processed")
    sc = res["scored"]
    last = sc["period"].max()
    print(f"scored rows: {len(sc):,}  latest period: {last}")
    print(sc[sc["period"] == last]["risk_band"].value_counts().to_string())
    print(sc[sc["period"] == last]["status"].value_counts().to_string())
    bt = res["backtest"]
    print({k: v for k, v in bt.items() if k != "events_table"})
