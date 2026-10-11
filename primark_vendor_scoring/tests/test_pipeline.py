"""
Business-rule tests. Run with either:
    pytest -q
    python tests/test_pipeline.py
"""
from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import generate_data as gd  # noqa: E402
import pipeline as pl  # noqa: E402

_CACHE: dict = {}


def _world():
    """Generate a small dataset once and run the pipeline on it."""
    if "w" not in _CACHE:
        tmp = Path(tempfile.mkdtemp())
        gd.generate(tmp, {"seed": 11, "n_vendors": 30, "months": 18, "end_month": "2026-03",
                          "profile_mix": {"reliable": 11, "average": 10, "deteriorating": 5, "volatile": 4}})
        cfg = pl.load_config()
        raw = pl.load_raw(tmp)
        _CACHE["w"] = (tmp, cfg, raw, pl.run_pipeline(raw, cfg))
    return _CACHE["w"]


def test_generator_is_reproducible():
    a, b = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    s = {"seed": 5, "n_vendors": 20, "months": 8, "end_month": "2026-03",
         "profile_mix": {"reliable": 8, "average": 6, "deteriorating": 3, "volatile": 3}}
    gd.generate(a, s)
    gd.generate(b, s)
    for f in ["po_lines.csv", "deliveries.csv", "invoices.csv", "inspections.csv"]:
        assert (a / f).read_text() == (b / f).read_text(), f


def test_duplicates_removed_and_cancelled_separate():
    _, _, raw, res = _world()
    po = res["po_all"]
    assert po["po_line_id"].is_unique
    assert res["dq"]["duplicate_po_lines_removed"] > 0
    assert res["dq"]["cancelled_po_lines"] > 0
    assert not po["po_line_id"].isin(res["cancelled"]["po_line_id"]).any()


def test_vendor_ids_standardised():
    _, _, _, res = _world()
    assert res["po_all"]["vendor_id"].str.fullmatch(r"V\d{3}").all()


def test_valid_rows_follow_date_sequence_and_ranges():
    _, _, _, res = _world()
    v = res["po_all"][res["po_all"]["valid"]]
    assert (v["ship_date"] >= v["order_date"]).all()
    assert (v["receipt_date"] >= v["ship_date"]).all()
    inv = v[v["has_invoice"]]
    assert (inv["invoice_date"] >= inv["receipt_date"]).all()
    assert v["fill_rate"].between(0, 1).all()
    assert v["defect_rate"].dropna().between(0, 1).all()
    assert set(res["quarantine"]["quarantine_reason"]) <= {"no_delivery_record", "date_sequence_invalid", "invalid_value"}


def test_scores_and_metrics_in_range():
    _, cfg, _, res = _world()
    s = res["scored"]
    assert s["score"].between(0, 100).all()
    assert s["on_time_pct"].between(0, 100).all()
    assert s["fill_rate_pct"].between(0, 100).all()
    for m in pl.METRICS:
        assert s[f"norm_{m}"].dropna().between(0, 100).all()
    # contributions reconcile: score_raw = 100 - sum(points lost)
    lost = s[[f"loss_{m}" for m in pl.METRICS]].sum(axis=1)
    assert np.allclose(s["score_raw"], 100 - lost, atol=1e-6)


def test_scoring_is_deterministic():
    _, cfg, raw, res = _world()
    again = pl.run_pipeline(raw, cfg)
    cols = ["score", "risk_band", "status", "early_warning"]
    pd.testing.assert_frame_equal(res["scored"][cols], again["scored"][cols])


def test_no_future_leakage_in_early_warning():
    """Flags at month T must be identical whether or not later months exist."""
    _, cfg, _, res = _world()
    monthly = res["monthly"]
    periods = sorted(monthly["period"].unique())
    cutoff = periods[-6]
    full = res["scored"]
    trunc = pl.add_trends(pl.score_monthly(monthly[monthly["period"] <= cutoff], cfg), cfg)
    a = full[full["period"] <= cutoff].sort_values(pl.KEYS).reset_index(drop=True)
    b = trunc.sort_values(pl.KEYS).reset_index(drop=True)
    for col in ["score", "delta_vs_baseline", "trend_slope3", "early_warning", "status"]:
        if pd.api.types.is_numeric_dtype(a[col]) and not pd.api.types.is_bool_dtype(a[col]):
            assert np.allclose(a[col].fillna(-999), b[col].fillna(-999)), col
        else:
            assert (a[col].to_numpy() == b[col].to_numpy()).all(), col


def test_low_volume_vendor_not_unfairly_high_risk():
    _, cfg, _, res = _world()
    base = res["monthly"].iloc[:1].copy()
    rows = []
    for i, (n, ot, df_) in enumerate([(12, 97, 0.5), (11, 96, 0.6), (1, 0, 9.0)]):  # third = one terrible PO
        r = base.iloc[0].copy()
        r.update({"vendor_id": f"T{i}", "category": "Womenswear", "period": "2026-01", "po_count": n,
                  "on_time_pct": ot, "fill_rate_pct": 80.0 if i == 2 else 99.0, "defect_pct": df_,
                  "cost_variance_abs_pct": 0.5, "lead_time_cv": 0.06, "total_lines": n, "matched_lines": n,
                  "data_completeness": 1.0})
        rows.append(r)
    out = pl.score_monthly(pd.DataFrame(rows), cfg)
    thin = out.iloc[2]
    assert thin["score_raw"] < 30            # the raw score is terrible...
    assert thin["risk_band"] != "High"       # ...but one PO line must not make the vendor High risk
    assert thin["confidence"] in ("Low", "Medium")


def test_weights_change_rankings():
    _, cfg, _, res = _world()
    mon = res["monthly"]
    last = mon["period"].max()
    a = pl.score_monthly(mon, cfg)
    only_defects = {m: 0.0 for m in pl.METRICS}
    only_defects["defect_pct"] = 1.0
    b = pl.score_monthly(mon, cfg, only_defects)
    ra = a[a["period"] == last].set_index(pl.VC)["score"].rank()
    rb = b[b["period"] == last].set_index(pl.VC)["score"].rank()
    assert (ra != rb.reindex(ra.index)).any()


def test_alternates_same_category_and_capacity():
    _, cfg, raw, res = _world()
    sc = res["scored"]
    last = sc["period"].max()
    flagged = sc[(sc["period"] == last)].sort_values("score").iloc[0]
    alt = pl.rank_alternates(sc, raw, cfg, flagged["vendor_id"], flagged["category"], last)
    supplies = set(raw["vendor_categories"].query("category == @flagged.category")["vendor_id"])
    assert set(alt["vendor_id"]) <= supplies
    assert flagged["vendor_id"] not in set(alt["vendor_id"])
    ok = alt[alt["eligible"]]
    assert (ok["capacity_headroom_x"] >= cfg["alternates"]["min_headroom_ratio"]).all()
    assert (ok["avg_score"] >= cfg["risk_bands"]["medium_min"]).all()
    assert (alt.loc[~alt["eligible"], "excluded_because"] != "").all()


def test_every_flag_has_an_explanation():
    _, cfg, _, res = _world()
    flagged = res["scored"].query("status != 'OK'")
    assert flagged["driver_1"].str.len().gt(0).all()
    assert flagged["top_driver_metric"].isin(pl.METRICS).all()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
