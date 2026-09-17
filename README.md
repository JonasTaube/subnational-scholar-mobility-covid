# Reconfiguration, not collapse: pandemic-era patterns of subnational scholarly migration worldwide

Analysis code and derived outputs for a manuscript on how the COVID-19 pandemic
reconfigured **subnational** (admin-1 region) scholarly migration worldwide.

Everything here is **descriptive and associational**. We deliberately do not claim
causal identification; see *Identification limits* below for the reason.

---

## 1. Data

All source data come from **SMD 2.0** (*Scholarly Migration Database*, version 2.0),
which derives region-to-region flows of researchers from affiliation changes in
bibliometric records.

| Item | Value |
|---|---|
| Working paper | MPIDR Working Paper WP-2026-029 |
| Working paper DOI | https://doi.org/10.4054/MPIDR-WP-2026-029 |
| Data record DOI | https://doi.org/10.5281/zenodo.20487917 |
| Record title | *Replication data for: Bilateral flows and rates of subnational and international migration of scholars worldwide by gender and field* |
| Creators | Theile, Tom; Akbaritabar, Aliakbar; Zagheni, Emilio |
| Licence | CC-BY 4.0 |

The predecessor database (SMD 1.0, https://doi.org/10.1038/s41597-024-03655-9)
is country-level only, international-only, and stops at 2020; SMD 2.0 is the first
release with subnational resolution, internal flows, and coverage through 2022.

**The raw files are not redistributed here** (they are ~33 MB of Parquet and are
better obtained from the source above, which keeps the DOI citation intact).
Download the four `*_subnational_*` Parquet files for both sources into

```
01_原始数据/smd2_repo/data_input/2026_V2/
```

### ⚠️ File-convention trap (read this before writing your own loader)

The two subnational files in SMD 2.0 use **different area-code prefixes**:

* region-level counts file (`*_subnational_gender_and_field.parquet`) → **ISO3** prefix, e.g. `CHN.BJ`
* bilateral flows file (`*_subnational_flows_gender_and_field.parquet`) → **ISO2** prefix, e.g. `CN.BJ`

Joining them on the raw `area` string silently returns **zero** rows for every
country whose ISO2 ≠ ISO3. Our `analysis_robustness.py` hit exactly this and the
China case returned an empty frame. All scripts here therefore select country
blocks with ISO2 prefixes when reading the flows file.

### Verified coverage facts

| | OpenAlex-based | Scopus-based |
|---|---|---|
| admin-1 regions | 2,073 | 2,065 |
| countries | 196 | 214 |
| counts span | 1994–2025 (2025 partial) | 1994–2025 (2025 partial) |
| **bilateral flows span** | **1997–2022** | **1997–2022** |
| gender | m / f / u / all | m / f / u / all |
| field | 3 macro-groups | 3 macro-groups |

The bilateral flows file **ends in 2022**, which is what forces the 2022 endpoint
used throughout the manuscript.

---

## 2. Scripts

Run in this order. Each writes into `03_分析与图表/results/` and/or `figures/`.

| Script | What it does |
|---|---|
| `analysis_core.py` | Loads the panel, builds the balanced sample, estimates the region-FE year path and the continuous-treatment DID, produces Figures 1–5 and `headline.json` |
| `analysis_robustness.py` | Scopus replication; 2022 endpoint; region-size × year, field-composition, country × year FE, region linear trends; trajectory denominator / endpoint sensitivities; class diagnostics; coverage diagnostics; China check → `robustness.json` |
| `analysis_robustness2.py` | Primary trajectory classification with a **2022-only** post window, mean/median/IQR and sign shares, absolute net-loss targeting stats; country × year FE diagnostics; cross-source comparability; regenerates Figure 3 → `robustness_round2.json` |
| `analysis_robustness3.py` | Clustering-level robustness: identical linear predictors re-estimated with a region-level vs a country-level sandwich → `robustness_round3.json` |
| `verify_revision_facts.py` | Standalone re-check of sample sizes, the 2022 total, and DID significance |

```bash
python analysis_core.py
python analysis_robustness.py
python analysis_robustness2.py
python analysis_robustness3.py
```

Environment: Python 3.13, `numpy`, `pandas`, `pyarrow`, `matplotlib`.
No econometrics package is used — fixed effects are absorbed by alternating
demeaning and standard errors are hand-computed, so the sandwich construction is
visible in the source rather than hidden behind a library call.

---

## 3. Design

### Sample

Regions with a mean 2015–2019 researcher population ≥ 20, restricted to a
**balanced 2015–2024 panel**:

* OpenAlex: **1,711 regions / 191 countries / 17,110 region-years**
* Scopus: 1,436 regions (for the trajectory replication)

### Exposure

`Expose_i` = the region's **2015–2019 international share of in-migration**
(international in-migration ÷ all in-migration), standardized. Across regions
SD = **0.261**. High-exposure regions are the internationally connected hubs that
a border-closure shock should hit hardest.

### Estimation

**Outcome:** `log1p(number_of_inmigrations)` — a log transform because the
in-migration counts are heavily right-skewed.

| Spec | Linear predictor | Fixed effects | Clustering |
|---|---|---|---|
| S1 baseline DID | year dummies + `Expose × year dummies` | region | region (country reported too) |
| S2 country × year | `Expose × year dummies` | region **+ country × year** | region (country reported too) |
| S3 multi-region only | year dummies + `Expose × year dummies` | region | region, sample = countries with ≥ 2 regions |

2019 is the omitted year. Coefficients are per one SD of exposure.

### Identification limits — why nothing here is causal

1. **There is a single common shock date.** Event time is therefore collinear
   with calendar year, so a two-way fixed-effects *event study* is not identified.
   We report the region-FE year path (spec A) purely descriptively, and the
   `Expose × year` coefficients (spec B) as a **differential association**.
2. **Country × year fixed effects absorb the gradient.** Under S2 the post-period
   coefficients collapse to ≈ 0 and the *pre-period* coefficients grow (e.g.
   +0.082 in 2015). The exposure gradient is therefore **chiefly between-country
   and partly pre-existing**, not a within-country pandemic effect. This is the
   single most important robustness result in the paper.
3. **Clustering level matters.** Re-estimating spec S1 with a country-level
   sandwich leaves the point estimates unchanged but widens the SEs enough that
   2021 (p = 0.115) and 2024 (p = 0.090) fall below conventional thresholds;
   2022 (p = 0.013) and 2023 (p = 0.005) survive. The precise significance of the
   later years should not be over-read.
4. Region fixed effects absorb only **time-invariant** region-level confounding;
   they say nothing about time-varying confounders such as funding, visa policy,
   or institute-level hiring freezes.

### Known data caveats

* **Right truncation.** Affiliation-change capture is incomplete for 2023–2024.
  Those years are shown for transparency but not interpreted. The primary
  trajectory classification therefore uses a **2022-only** post window.
* **Zero population denominators.** `padded_population_of_researchers` is zero
  for some regions from 2022, which makes the official `netmigrationrate`
  undefined. We use a **fixed denominator**: net migration ÷ mean 2015–2019
  population × 1,000.
* **Cross-source level gap.** OpenAlex counts 2019 in-migrations ≈ **5.1×**
  Scopus counts, and the two sources disagree in sign in 2020 (OpenAlex
  international −3.6%, Scopus +12.4%). Only 554 of 2,073 OpenAlex regions (26.7%)
  appear in the Scopus region set, and the correlation of region-level log change
  2019→2022 across the 489 shared regions with population ≥ 20 is only **0.203**.
  Scopus is used for **directional validation only**, never for level comparison.

### Trajectory classification

Each region is assigned from its mean fixed-denominator net rate in three windows —
pre-COVID (2015–2019), shock (2020–2021), post (2022 primary; 2022–2024 alternative):

| Condition | Class |
|---|---|
| shock < pre and post ≥ pre | Temporary decline, rebounded |
| shock < pre and post < pre | Relative persistent decline |
| shock ≥ pre and post ≥ pre | Relative persistent improvement |
| shock ≥ pre and post < pre | Improvement then reversal |

Class names are deliberately **relative**: they describe movement of a region's
own rate versus its own pre-pandemic level, not whether a region is a net receiver
or sender in absolute terms. The diagnostics file makes this concrete — e.g. 37%
of regions in the *Relative persistent improvement* class had a **negative**
pre-pandemic net rate, and 13% were still negative in 2022.

---

## 4. Outputs

```
results/
  headline.json                      headline aggregates (flows, growth rates, 2022 total)
  fig1_global_flows.csv              annual global in-/out-/net migration
  fig2_event_study_*.csv             region-FE year paths
  fig2_did_exposure.csv              baseline DID coefficients
  fig3_trajectory_counts.csv         trajectory class counts
  fig4_intl_internal.csv             international vs internal split
  fig5_china.csv  china_province_change.csv
  robustness.json                    main robustness battery
  robust_*.csv                       trajectory sensitivity variants
  robustness_round2.json             primary classification, country×year diagnostics, cross-source
  robustness_round3.json             clustering-level robustness
  traj_primary_2022_*.csv            primary classification (post window = 2022)
  traj_alt_2022_2024_*.csv           alternative classification (post window = 2022–2024)
figures/
  fig1..fig5, .pdf (vector, embedded fonts) + .png (300 dpi)
```

Figure conventions: Times New Roman, `pdf.fonttype = 42` (editable embedded
TrueType), single-column width 7.2 in, no titles inside the figure area
(captions live in the manuscript).

---

## 5. Citation

If you use this code, please cite the SMD 2.0 data record
(https://doi.org/10.5281/zenodo.20487917) and the working paper
(https://doi.org/10.4054/MPIDR-WP-2026-029) alongside the manuscript.

Code: MIT. Derived tables and figures: CC-BY 4.0.
