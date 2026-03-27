# Progress Check Documentation (Draft)

## 1) Project Question
**Primary question:**
How do **regional temperature and wind conditions**, and **regional solar generation**, explain and predict **net electricity demand in Great Britain** over a selected 6‑month window?

**Operational target:**
Forecast net demand (ND) using regional weather + regional solar generation + national grid context.

**Selected time window (6 months):**
- _Fill in exact dates here (e.g., 2024-07-01 to 2024-12-31)_

---

## 2) Data Sources + Permission to Use (Evidence Links)

### Dataset A — National Grid Demand + Embedded Wind/Solar (NESO)
- **Source:** NESO Data Portal — Historic Demand Data
- **Why:** Contains ND (National Demand), TSD, embedded wind/solar generation and interconnector flows
- **Permission:** Rights listed as **NESO Open Data Licence** on the dataset page
- **Evidence links:**
```
https://www.neso.energy/data-portal/historic-demand-data
https://www.neso.energy/data-portal/neso-open-licence
```

### Dataset B — Regional Solar Generation (Sheffield Solar PV‑Live)
- **Source:** Sheffield Solar PV‑Live (University of Sheffield)
- **Why:** Regional solar PV output at GSP or PES level
- **Permission:** **CC BY 4.0** license (public reuse with attribution)
- **Evidence links:**
```
https://www.solar.sheffield.ac.uk/pvlive/regional/
https://api.solar.sheffield.ac.uk/pvlive/gdocs
```

### Dataset C — Regional Weather (Met Office DataPoint)
- **Source:** Met Office DataPoint API (observations/forecast)
- **Why:** Wind speed and temperature by station; can aggregate to region
- **Permission:** **Open Government Licence (OGL)** for DataPoint
- **Evidence links:**
```
https://www.metoffice.gov.uk/services/data/datapoint/terms-and-conditions---datapoint
```

---

## 3) Datasheets (Short)

### Dataset A: NESO Historic Demand Data
- **Owner:** National Energy System Operator (NESO)
- **Coverage:** 2020–2025 (we will use a 6‑month window)
- **Granularity:** Half‑hourly (SETTLEMENT_DATE + SETTLEMENT_PERIOD)
- **Key fields:** ND, TSD, EMBEDDED_WIND_GENERATION, EMBEDDED_SOLAR_GENERATION, interconnector flows
- **Join keys:** Time (settlement date/period)
- **Limitations:** Uses UK settlement time (BST/GMT clock changes)

### Dataset B: PV‑Live Regional Solar (GSP/PES)
- **Owner:** Sheffield Solar, University of Sheffield
- **Coverage:** 2020–2025 in our files
- **Granularity:** 30‑minute intervals (UTC/GMT)
- **Key fields:** gsp_id, gsp_name, datetime_gmt, generation_mw
- **Join keys:** GSP IDs + time
- **Limitations:** PV‑Live provides **estimated** outturns (not direct meter data)

### Dataset C: Met Office DataPoint Weather
- **Owner:** Met Office (UK Gov)
- **Coverage:** station‑level observations (hourly)
- **Granularity:** hourly (time‑stamped, UTC/GMT)
- **Key fields:** temperature, wind speed, station coordinates
- **Join keys:** station → region mapping; time alignment
- **Limitations:** Must aggregate stations into regions; ensure coverage quality

---

## 4) Current Progress Toward EDA (Honest Status)
- **Downloaded:**
  - PV‑Live regional solar data (`regional_solar_2020_2025.csv`)
  - PV‑Live labeled with GSP info (`regional_solar_2020_2025_labeled.csv`)
  - Hourly solar dataset (`solar_energy_hourly.csv`)
  - NESO Historic Demand data 2020–2025 (`data/demanddata_*.csv`)
- **Not yet downloaded as CSV:** Met Office regional weather (DataPoint API still to run)
- **Known data issue to fix:** time alignment (PV‑Live UTC vs NESO settlement time with BST)

---

## 5) Plan for Midterm Demonstration (5‑min demo)
1) Project question + why it matters (30s)
2) Data sources + license evidence (60s)
3) Pipeline demo (PV‑Live → labeled → hourly) (60s)
4) One EDA plot (regional solar seasonality) + national demand overlay (60s)
5) Baseline model plan + evaluation metric (60s)

---

## 6) Project Plan (1‑Month Gantt‑style Breakdown)

**Week 1 (Feb 3 – Feb 9)**
- Finalize weather ingestion via DataPoint
- Create region mapping for weather stations
- Normalize all timezones to UTC

**Week 2 (Feb 10 – Feb 16)**
- EDA: missingness, seasonality, regional comparisons
- Build baseline model (seasonal naive or linear regression)

**Week 3 (Feb 17 – Feb 23)**
- Feature engineering: lags, rolling stats, weather aggregates
- Train models: Random Forest, XGBoost

**Week 4 (Feb 24 – Mar 2)**
- Train LSTM (sequence model)
- Model comparison + error analysis
- Draft final report + slides

---

## 7) Team Roles (4 members)
- **Member A:** Data ingestion + licensing evidence
- **Member B:** Time alignment + data engineering
- **Member C:** EDA + visualization
- **Member D:** Modeling + evaluation

---

## 8) Risks & Mitigation (Bullet‑proof)
- **Risk:** Weather data not yet aggregated by region → **Mitigation:** station‑to‑region mapping, documented method
- **Risk:** BST/GMT mismatch → **Mitigation:** standardize all timestamps to UTC before joining
- **Risk:** PV‑Live estimates vs actual → **Mitigation:** state clearly in limitations and do not claim exact generation

---

## 9) Deliverables for Canvas Submission
- This document (question + sources + permissions + datasheets + plan)
- Evidence links for licensing (included above)
- Summary table of datasets + join keys

