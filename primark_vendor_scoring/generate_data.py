"""
Synthetic data generator for the Vendor Performance Scoring demo.

Produces fragmented, slightly dirty source tables that mimic what a retailer would
pull from ERP / WMS / QC / AP systems:

    vendors.csv, vendor_categories.csv, skus.csv, fx_rates.csv, disruption_events.csv
    po_lines.csv, deliveries.csv, inspections.csv, invoices.csv
    vendor_profiles.csv   (ground truth used only for validation, never for scoring)
    generator_settings.json

Everything is driven by a fixed random seed so a run is reproducible.

Usage:
    python generate_data.py                 # writes to data/raw
    python generate_data.py --seed 7 --out data/raw
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SETTINGS = {
    "seed": 42,
    "n_vendors": 50,
    "months": 24,
    "end_month": "2026-09",
    "profile_mix": {"reliable": 18, "average": 16, "deteriorating": 9, "volatile": 7},
    "tier_mix": {"Strategic": 0.20, "Preferred": 0.40, "Standard": 0.40},
    "mean_po_lines_per_month": {"Strategic": 11.0, "Preferred": 7.0, "Standard": 3.8},
    "thin_pair_share_standard": 0.25,  # share of Standard vendor-category pairs with very low volume
    "peak_months": [8, 9, 10, 11],      # autumn/winter peak pressure
    "dirty_rates": {
        "cancelled": 0.025,
        "missing_delivery": 0.012,
        "missing_inspection": 0.030,
        "missing_invoice": 0.025,
        "duplicate_po_line": 0.010,
        "duplicate_delivery": 0.005,
        "orphan_records": 0.003,
        "invalid_date_sequence": 0.006,
        "invoice_qty_mismatch": 0.030,
        "messy_vendor_id": 0.010,
    },
}

# name: (code, base unit cost GBP, production/QA days)
CATEGORIES = {
    "Womenswear": ("WOM", 5.2, 28),
    "Menswear": ("MEN", 5.8, 28),
    "Kidswear": ("KID", 3.6, 25),
    "Lingerie & Sleepwear": ("LNG", 2.9, 32),
    "Footwear": ("FTW", 7.5, 40),
    "Accessories": ("ACC", 1.8, 22),
    "Homeware": ("HOM", 4.1, 35),
    "Beauty": ("BTY", 1.2, 18),
    "Hosiery & Socks": ("HSY", 0.9, 20),
    "Seasonal & Gifting": ("SEA", 3.0, 26),
}

# country: (currency, typical transit days, sampling weight)
COUNTRIES = {
    "Bangladesh": ("USD", 32, 0.20),
    "India": ("USD", 28, 0.16),
    "China": ("USD", 30, 0.14),
    "Turkey": ("EUR", 9, 0.12),
    "Pakistan": ("USD", 30, 0.08),
    "Vietnam": ("USD", 28, 0.08),
    "Portugal": ("EUR", 5, 0.06),
    "Sri Lanka": ("USD", 26, 0.06),
    "Cambodia": ("USD", 30, 0.05),
    "Egypt": ("USD", 14, 0.05),
}

DISRUPTIONS = [
    {"event": "Red Sea shipping disruption", "start_month": "2025-06", "end_month": "2025-08",
     "countries": "Bangladesh;India;Pakistan;Sri Lanka", "extra_delay_days": 6},
    {"event": "Far East port congestion", "start_month": "2026-02", "end_month": "2026-03",
     "countries": "China;Vietnam;Cambodia", "extra_delay_days": 5},
]

PREFIXES = ["Apex", "Bluegate", "Crescent", "Dynamo", "Everest", "Fairway", "Golden", "Harbor", "Ivory",
            "Jade", "Kestrel", "Lotus", "Meridian", "Northstar", "Orchid", "Pinnacle", "Quantum",
            "Riverside", "Sterling", "Titan", "Unity", "Vertex", "Willow", "Zenith", "Aurora"]
SUFFIXES = ["Textiles", "Apparel", "Garments", "Manufacturing", "Industries", "Exports", "Group"]

# Performance parameter sets: R = reliable, A = average, B = bad
_R = dict(p_late=0.04, late_mean=2.0, lt_noise=1.2, defect=0.006, partial_prob=0.03, partial_low=0.93,
          cv_mean=0.000, cv_sd=0.004)
_A = dict(p_late=0.20, late_mean=4.0, lt_noise=2.5, defect=0.020, partial_prob=0.10, partial_low=0.88,
          cv_mean=0.006, cv_sd=0.012)
_B = dict(p_late=0.55, late_mean=9.0, lt_noise=6.0, defect=0.060, partial_prob=0.40, partial_low=0.65,
          cv_mean=0.050, cv_sd=0.030)
TIER_MULT = {"Strategic": 0.8, "Preferred": 1.0, "Standard": 1.2}


def _blend(a: dict, b: dict, w: float) -> dict:
    return {k: a[k] + w * (b[k] - a[k]) for k in a}


def _vendor_params(profile, tier, m, det_start, severity, bad_month, period_month, peak_months):
    """Performance parameters for a vendor-category in month index m."""
    if profile == "reliable":
        p = dict(_R)
    elif profile == "average":
        p = dict(_A)
    elif profile == "deteriorating":
        g = float(np.clip((m - det_start + 1) / 10.0, 0.0, 1.0)) * severity
        p = _blend(_R, _B, g)
    else:  # volatile
        p = _blend(_A, _B, 0.85) if bad_month else _blend(_R, _A, 0.6)
    tm = TIER_MULT[tier]
    p["p_late"] = min(0.9, p["p_late"] * tm)
    p["defect"] = p["defect"] * tm
    p["partial_prob"] = min(0.8, p["partial_prob"] * tm)
    if period_month in peak_months:  # peak-season pressure
        p["p_late"] = min(0.9, p["p_late"] * 1.35)
        p["late_mean"] *= 1.2
    return p


def generate(out_dir: str | Path = "data/raw", settings: dict | None = None) -> dict[str, pd.DataFrame]:
    s = json.loads(json.dumps(DEFAULT_SETTINGS))
    if settings:
        s.update(settings)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(s["seed"])
    periods = pd.period_range(end=s["end_month"], periods=s["months"], freq="M")
    cat_names = list(CATEGORIES.keys())

    # ------------------------------------------------------------------ vendors
    n = s["n_vendors"]
    tiers = rng.choice(list(s["tier_mix"].keys()), size=n, p=list(s["tier_mix"].values()))
    cw = np.array([v[2] for v in COUNTRIES.values()])
    countries = rng.choice(list(COUNTRIES.keys()), size=n, p=cw / cw.sum())
    mix = s["profile_mix"]
    profile_list = [p for p, c in mix.items() for _ in range(c)]
    profile_list = (profile_list + ["average"] * n)[:n]
    profiles = rng.permutation(profile_list)
    det_starts_pool = [9, 11, 12, 14, 15, 16, 17, 18, 19, 20]
    vendors = pd.DataFrame({
        "vendor_id": [f"V{i + 1:03d}" for i in range(n)],
        "vendor_name": [f"{PREFIXES[i % 25]} {SUFFIXES[(i * 3 + i // 25) % 7]}" for i in range(n)],
        "tier": tiers,
        "country": countries,
        "currency": [COUNTRIES[c][0] for c in countries],
        "onboarded_year": rng.integers(2008, 2025, n),
    })
    truth = pd.DataFrame({"vendor_id": vendors.vendor_id, "profile": profiles})
    truth["deterioration_start"] = ""
    truth["severity"] = np.nan
    det_start = {}
    severity = {}
    for i, row in truth.iterrows():
        if row.profile == "deteriorating":
            idx = int(rng.choice(det_starts_pool))
            det_start[row.vendor_id] = idx
            severity[row.vendor_id] = float(rng.uniform(0.75, 1.0))
            truth.loc[i, "deterioration_start"] = str(periods[min(idx, len(periods) - 1)])
            truth.loc[i, "severity"] = severity[row.vendor_id]

    # ----------------------------------------------------------- vendor x category
    supplies: dict[str, list[str]] = {}
    for _, v in vendors.iterrows():
        k = {"Strategic": rng.integers(3, 5), "Preferred": rng.integers(2, 4), "Standard": rng.integers(1, 3)}[v.tier]
        supplies[v.vendor_id] = [cat_names[j] for j in rng.choice(len(cat_names), int(k), replace=False)]
    for cat in cat_names:  # guarantee >= 6 vendors per category so alternates exist
        have = [vid for vid, cs in supplies.items() if cat in cs]
        while len(have) < 6:
            vid = str(rng.choice(vendors.vendor_id))
            if vid not in have:
                supplies[vid].append(cat)
                have.append(vid)

    mean_qty = 9500.0
    tier_of = dict(zip(vendors.vendor_id, vendors.tier))
    country_of = dict(zip(vendors.vendor_id, vendors.country))
    ccy_of = dict(zip(vendors.vendor_id, vendors.currency))
    vc_rows, pairs = [], {}
    for vid, cats in supplies.items():
        offset = int(rng.integers(-4, 5))
        for cat in cats:
            tier = tier_of[vid]
            lam = s["mean_po_lines_per_month"][tier]
            if tier == "Standard" and rng.random() < s["thin_pair_share_standard"]:
                lam = 1.3  # thin low-volume pair
            lam *= float(rng.lognormal(0, 0.2))
            prod_days = CATEGORIES[cat][2]
            planned = prod_days + COUNTRIES[country_of[vid]][1] + offset
            cap = int(round(lam * mean_qty * float(rng.uniform(1.5, 3.2)), -2))
            pairs[(vid, cat)] = dict(lam=lam, planned=planned, noise=float(rng.lognormal(0, 0.12)),
                                     jitter=int(rng.integers(-1, 2)))
            vc_rows.append({"vendor_id": vid, "category": cat, "capacity_units_month": cap,
                            "planned_lead_time_days": planned})
    vendor_categories = pd.DataFrame(vc_rows)

    # ------------------------------------------------------------- skus & fx
    sku_rows = []
    for cat, (code, base, _) in CATEGORIES.items():
        for j in range(30):
            sku_rows.append({"sku_id": f"{code}-{j + 1:04d}", "category": cat,
                             "base_cost_gbp": round(base * float(rng.uniform(0.7, 1.4)), 3)})
    skus = pd.DataFrame(sku_rows)
    sku_by_cat = {c: g.reset_index(drop=True) for c, g in skus.groupby("category")}

    fx_rows, usd, eur = [], 0.79, 0.85
    for p in periods:
        usd = float(np.clip(usd + rng.normal(0, 0.006), 0.72, 0.86))
        eur = float(np.clip(eur + rng.normal(0, 0.005), 0.80, 0.90))
        fx_rows += [{"month": str(p), "currency": "USD", "rate_to_gbp": round(usd, 4)},
                    {"month": str(p), "currency": "EUR", "rate_to_gbp": round(eur, 4)},
                    {"month": str(p), "currency": "GBP", "rate_to_gbp": 1.0}]
    fx = pd.DataFrame(fx_rows)
    fx_lookup = {(r.month, r.currency): r.rate_to_gbp for r in fx.itertuples()}
    events = pd.DataFrame(DISRUPTIONS)

    def disruption_delay(country, period):
        for e in DISRUPTIONS:
            if e["start_month"] <= str(period) <= e["end_month"] and country in e["countries"].split(";"):
                return e["extra_delay_days"]
        return 0

    # vendor-level "bad month" states for volatile vendors (shared across categories)
    bad_state = {(vid, m): bool(rng.random() < 0.30) for vid in vendors.vendor_id for m in range(len(periods))}
    profile_of = dict(zip(truth.vendor_id, truth.profile))

    # ----------------------------------------------------------- PO line loop
    chunks = []
    for (vid, cat), pr in pairs.items():
        tier, country, ccy = tier_of[vid], country_of[vid], ccy_of[vid]
        prof = profile_of[vid]
        for m, period in enumerate(periods):
            lam = pr["lam"] * (1.25 if period.month in s["peak_months"] else 1.0)
            k = int(rng.poisson(lam))
            if k == 0:
                continue
            p = _vendor_params(prof, tier, m, det_start.get(vid, 0) + pr["jitter"], severity.get(vid, 0.0),
                               bad_state[(vid, m)], period.month, s["peak_months"])
            p_late = min(0.9, p["p_late"] * pr["noise"])
            defect = p["defect"] * pr["noise"]

            late = rng.random(k) < p_late
            base = np.where(late, np.ceil(rng.exponential(p["late_mean"], k)) + 1, -rng.integers(2, 8, k))
            delay = np.rint(base + rng.normal(0, p["lt_noise"], k)).astype(int)
            extra = disruption_delay(country, period)
            if extra:
                hit = rng.random(k) < 0.75
                delay = delay + np.where(hit, np.rint(rng.uniform(0.5, 1.3, k) * extra).astype(int), 0)

            qty = np.clip(rng.lognormal(np.log(7000), 0.7, k), 500, 60000)
            qty = (np.round(qty / 50) * 50).astype(int)
            partial = rng.random(k) < p["partial_prob"]
            fill = np.where(partial, rng.uniform(p["partial_low"], 0.99, k), 1.0)
            cv = rng.normal(p["cv_mean"], p["cv_sd"], k)

            skus_c = sku_by_cat[cat]
            pick = rng.integers(0, len(skus_c), k)
            promised = (np.datetime64(period.start_time.date(), "D")
                        + rng.integers(0, period.days_in_month, k).astype("timedelta64[D]"))
            lead = pr["planned"] + rng.integers(-4, 5, k)
            order = promised - lead.astype("timedelta64[D]")
            order_month = pd.to_datetime(order).strftime("%Y-%m")
            fx_r = np.array([fx_lookup.get((om, ccy), fx_lookup.get((str(periods[0]), ccy), 1.0))
                             for om in order_month])
            agreed = np.round(skus_c.base_cost_gbp.to_numpy()[pick] / fx_r, 3)

            chunks.append(pd.DataFrame({
                "vendor_id": vid, "category": cat, "sku_id": skus_c.sku_id.to_numpy()[pick],
                "order_date": order, "promised_date": promised, "ordered_qty": qty,
                "agreed_unit_cost": agreed, "currency": ccy,
                "_delay": delay, "_fill": fill, "_defect": defect, "_cv": cv,
                "_transit": np.clip(COUNTRIES[country][1] + rng.integers(-3, 4, k), 2, None),
            }))

    L = pd.concat(chunks, ignore_index=True)
    L = L.sort_values(["promised_date", "vendor_id"]).reset_index(drop=True)
    n_lines = len(L)
    L["po_line_id"] = [f"POL{i + 1:07d}" for i in range(n_lines)]
    new_po = rng.random(n_lines) < 0.45
    L["_new_po"] = new_po
    seq = L.groupby("vendor_id")["_new_po"].cumsum()
    L["po_id"] = "PO-" + L["vendor_id"] + "-" + seq.astype(str).str.zfill(4)

    d = s["dirty_rates"]
    cancelled = rng.random(n_lines) < d["cancelled"]
    L["status"] = np.where(cancelled, "Cancelled", "Closed")
    live = ~cancelled

    # --------------------------------------------------------------- deliveries
    miss_del = rng.random(n_lines) < d["missing_delivery"]
    has_del = live & ~miss_del
    D = L.loc[has_del].copy()
    D["receipt_date"] = D["promised_date"] + pd.to_timedelta(D["_delay"], unit="D")
    D["ship_date"] = D["receipt_date"] - pd.to_timedelta(D["_transit"], unit="D")
    D["received_qty"] = np.floor(D["ordered_qty"] * D["_fill"]).astype(int)
    bad_seq = rng.random(len(D)) < d["invalid_date_sequence"]
    D.loc[bad_seq, "receipt_date"] = D.loc[bad_seq, "ship_date"] - pd.to_timedelta(rng.integers(1, 6, bad_seq.sum()), unit="D")
    deliveries = D[["po_line_id", "ship_date", "receipt_date", "received_qty"]].copy()
    deliveries.insert(0, "delivery_id", [f"DLV{i + 1:07d}" for i in range(len(deliveries))])

    # -------------------------------------------------------------- inspections
    keep_insp = rng.random(len(D)) >= d["missing_inspection"]
    I = D.loc[keep_insp].copy()
    sample = rng.choice([80, 125, 200], size=len(I))
    sample = np.minimum(sample, np.maximum(I["received_qty"].to_numpy(), 1))
    defects = rng.binomial(sample, I["_defect"].clip(0, 0.5).to_numpy())
    inspections = pd.DataFrame({
        "inspection_id": [f"INS{i + 1:07d}" for i in range(len(I))],
        "po_line_id": I["po_line_id"].to_numpy(),
        "inspection_date": (I["receipt_date"] + pd.to_timedelta(rng.integers(1, 4, len(I)), unit="D")).to_numpy(),
        "sample_size": sample, "defects": defects,
        "result": np.where(defects / sample > 0.05, "Fail", "Pass"),
    })

    # ----------------------------------------------------------------- invoices
    keep_inv = rng.random(len(D)) >= d["missing_invoice"]
    V = D.loc[keep_inv].copy()
    mismatch = rng.random(len(V)) < d["invoice_qty_mismatch"]
    inv_qty = np.where(mismatch, V["ordered_qty"], V["received_qty"])
    invoices = pd.DataFrame({
        "invoice_id": [f"INV{i + 1:07d}" for i in range(len(V))],
        "po_line_id": V["po_line_id"].to_numpy(),
        "invoice_date": (V["receipt_date"] + pd.to_timedelta(rng.integers(1, 11, len(V)), unit="D")).to_numpy(),
        "invoiced_qty": inv_qty,
        "unit_cost": np.round(V["agreed_unit_cost"] * (1 + V["_cv"]), 3).to_numpy(),
        "currency": V["currency"].to_numpy(),
    })

    # -------------------------------------------------- fragmentation / dirt
    po_cols = ["po_id", "po_line_id", "vendor_id", "sku_id", "category", "order_date", "promised_date",
               "ordered_qty", "agreed_unit_cost", "currency", "status"]
    po_lines = L[po_cols].copy()
    dup = po_lines.sample(frac=d["duplicate_po_line"], random_state=s["seed"])
    po_lines = pd.concat([po_lines, dup], ignore_index=True)
    messy = rng.random(len(po_lines)) < d["messy_vendor_id"]
    po_lines.loc[messy, "vendor_id"] = [
        (v.lower() if rng.random() < 0.5 else v + " ") for v in po_lines.loc[messy, "vendor_id"]]

    dup_d = deliveries.sample(frac=d["duplicate_delivery"], random_state=s["seed"] + 1).copy()
    dup_d["delivery_id"] = [f"DLV9{i:06d}" for i in range(len(dup_d))]
    n_orph = max(1, int(d["orphan_records"] * len(deliveries)))
    orph_d = deliveries.sample(n=n_orph, random_state=s["seed"] + 2).copy()
    orph_d["po_line_id"] = [f"POL9{i:06d}" for i in range(n_orph)]
    orph_d["delivery_id"] = [f"DLV8{i:06d}" for i in range(n_orph)]
    deliveries = pd.concat([deliveries, dup_d, orph_d], ignore_index=True)

    orph_i = invoices.sample(n=n_orph, random_state=s["seed"] + 3).copy()
    orph_i["po_line_id"] = [f"POL9{i:06d}" for i in range(n_orph)]
    orph_i["invoice_id"] = [f"INV8{i:06d}" for i in range(n_orph)]
    invoices = pd.concat([invoices, orph_i], ignore_index=True)

    tables = {
        "vendors": vendors, "vendor_categories": vendor_categories, "skus": skus, "fx_rates": fx,
        "disruption_events": events, "po_lines": po_lines, "deliveries": deliveries,
        "inspections": inspections, "invoices": invoices, "vendor_profiles": truth,
    }
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "generator_settings.json").write_text(json.dumps(s, indent=2))
    return tables


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--seed", type=int, default=DEFAULT_SETTINGS["seed"])
    args = ap.parse_args()
    t = generate(args.out, {"seed": args.seed})
    for k, v in t.items():
        print(f"{k:20s} {len(v):>8,d} rows")
