# Vendor Performance Scoring: Demo Application

A working demo of the **Vendor Performance Scoring** specification (Retail Rapid Build Ideas v3.0):
a transparent, explainable vendor scorecard with early-warning flags and alternate-vendor
recommendations. It runs entirely on **synthetic data**, and **no LLM is involved in scoring**:
every score, risk band and alternate rank is reproducible from structured data and `config.json`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On first launch the app generates demo data into `data/raw/` (about 21k PO lines, 50 vendors,
10 categories, 24 months, seed 42) and scores it. Use the sidebar to change period and filters,
or to regenerate the data with a different seed.

Other commands:

```bash
python generate_data.py --seed 42      # write synthetic source tables to data/raw
python pipeline.py                     # run the pipeline headless, outputs to data/processed
python tests/test_pipeline.py          # or: pytest -q
```

## What's in the app

| Tab | What it does | Spec |
|---|---|---|
| Leaderboard | Vendor × category or vendor-level ranking, risk bands, data-confidence, category percentile; filter by category, tier, market, period, band | FR3, FR8 |
| Vendor drill-down | Score explanation (points lost per metric), metric trends, recent late POs, completeness | FR2, FR5, NFR10 |
| Early warnings | Prioritised corrective-action list (CSV export) and a back-test against a static-threshold baseline | FR4, NFR3 |
| Alternate vendors | Same-category, capacity-constrained shortlist with ranking factors and exclusion reasons | FR6 |
| Scenario weights | Move the weight sliders and see score, band and rank changes instantly | FR7 |
| Data quality | Match rates, duplicates, quarantine reasons, ordered/received/invoiced reconciliation | FR1, NFR9 |
| Action log | Confirm / dismiss / escalate a flag and record the corrective action | FR9 |

## How scoring works

1. **Match & clean:** standardise vendor IDs, drop duplicate PO lines/deliveries, ignore orphan records,
   separate cancelled orders, link PO → delivery → inspection → invoice, quarantine lines with no delivery
   or an impossible date sequence (order ≤ ship ≤ receipt ≤ invoice).
2. **PO-line KPIs:** on-time (with a configurable grace period), days late, fill rate (capped at 100%), defect rate,
   cost variance % (computed in transaction currency so FX movements are not mistaken for price changes),
   lead time. Spend is converted to GBP.
3. **Aggregate** to vendor × category × month (by promised month).
4. **Score:** each metric is scaled linearly to 0–100 between a configurable *floor* and *target*; the score is
   the weighted sum. Points lost per metric are kept, so score = 100 − Σ points lost, and every flag is explainable.
5. **Low-volume protection:** scores are shrunk toward the category-month median in proportion to PO volume, and a
   data-confidence label (volume + match completeness) caps a "High" band to "Medium" when confidence is Low.
6. **Early warning (prior periods only):** volume-weighted score has fallen ≥ 6 points below its baseline and is
   falling ≥ 2 pts/month, has slipped below 80, but is not yet below 60. **Sustained underperformance** = below 60 for
   2+ consecutive months. A test proves that results at month T never change when later months are added.
7. **Alternates:** same category only; excludes high-risk, low-confidence or capacity-constrained vendors; ranks by
   performance, cost, capacity headroom and lead-time consistency (weights in `config.json`).

All thresholds and weights live in `config.json` (versioned; the version is shown in the app).

## Demo storyline (default seed 42, latest period 2026-09)

- **Early-warning example:** *Jade Group · Womenswear* scored ~89 in July, dropped to ~61 in August (late
  deliveries during peak season) and sits at ~75 in September, flagged as an early warning on roughly £0.7m of
  3-month spend. Open it in *Vendor drill-down* to see the drivers, then *Alternate vendors* for a shortlist.
- **Sustained underperformance:** *Everest Exports*, *Dynamo Apparel*, *Quantum Group* have been below threshold for months
  across several categories, so a strong case for the corrective-action list.
- **Disruptions:** shipping-delay events (Jun–Aug 2025, Feb–Mar 2026) affect groups of vendors by market. The drill-down
  notes when a disruption applies.
- **Fairness:** thin vendor-category pairs (1–2 POs a month) show Low/Medium data confidence, and one bad PO does not
  push them into High risk. Try *Scenario weights* with everything on defect rate to see rankings reshuffle.

## Back-test: what to expect, and its limits

On the default data the early-warning rule gets ~11% precision and ~31% recall against a base rate of ~4.6%
(about 2.4x lift), and flags 20 of 32 sustained-underperformance events early, at a median 2.5 months' lead time, versus
8% precision, 26% recall and 1.5 months for a static "score below 70" rule. Be careful how you read this:

- Vendors that genuinely **drift** down are caught well (12 of 13 in ground-truth checks), while **volatile**
  vendors with random bad months are largely unpredictable from trend alone (about 40% of events).
- Alert precision is modest by design: treat warnings as *review prompts*, not verdicts.
- Thresholds were tuned on synthetic data. **Re-calibrate on real history** before relying on them.

## Project layout

```
app.py              Streamlit scorecard (UI only; all logic lives in pipeline.py)
pipeline.py         match/clean → KPIs → score → trends → alternates → back-test
generate_data.py    reproducible synthetic source tables (with dirty data and vendor profiles)
config.json         weights, metric bounds, risk bands, early-warning and alternate rules
tests/              11 business-rule tests (reproducibility, no leakage, low-volume protection ...)
data/raw            generated source tables (git-ignored); data/processed holds scored outputs
```

## From demo to enterprise: next steps

- **Real data:** replace `load_raw()` with connectors to the ERP/PO, goods-receipt/WMS, QC and AP systems; keep the
  same column contracts. Add a data-steward screen for quarantined records.
- **Platform:** move storage to a governed lakehouse (raw / cleaned / curated layers); schedule with Airflow or
  Azure Data Factory; containerise and deploy via CI/CD.
- **Security & governance:** SSO, role-based access to commercial terms and invoices, encryption, audit trail, and an
  approval workflow for weight/threshold changes (config already carries a version).
- **UI:** the Streamlit app is ideal for the pilot. For wide rollout consider a React front end or Power BI behind
  the same scoring service (add a REST read endpoint).
- **Calibration:** category- and tier-specific targets, validated with buyers; metric definitions (what counts as "on
  time", how partial deliveries count) must be agreed before go-live.
- **Optional (not required):** an LLM may draft supplier-review narratives *from already-calculated metrics*, with
  buyer review. It should never compute scores or rankings.

## Tested with

Python 3.12, pandas 3.0, numpy 2.4. The pipeline and tests were run end to end; the Streamlit UI was exercised
with a stub (all tabs executed without errors) but **not in a live Streamlit session**, so expect minor cosmetic
tweaks on first run.
