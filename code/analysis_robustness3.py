# -*- coding: utf-8 -*-
"""
Round-3 addendum: proper clustering-level robustness.

The baseline DID uses region fixed effects and region-clustered SEs.
Reviewer asked whether, under country x year FE, SEs should be clustered at
the country level. Answering that requires holding the SPEC fixed and changing
only the clustering level -- so we re-estimate the identical linear predictor
with a country-level sandwich.

Specs (all: y = log1p(number_of_inmigrations), balanced 2015-2024 panel):
  S1  FE = region                 X = year dummies + Expose x year dummies
  S2  FE = region + country x year X = Expose x year dummies
  S3  FE = region                 X = year dummies + Expose x year dummies (multi-region countries only)
Each estimated with both region-level and country-level clustering.
Also: level gradient with country FE only (no region FE) for reference.
"""
import json, os, math
import numpy as np
import pandas as pd

# Set SMD2_BASE to your working copy, or keep this checkout's layout
import os
BASE = os.environ.get("SMD2_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "01_原始数据", "smd2_repo", "data_input", "2026_V2")
RES = os.path.join(BASE, "03_分析与图表", "results")
out = {}


def idx(arr):
    _, rinv = np.unique(arr, return_inverse=True)
    return rinv, int(rinv.max()) + 1


def within_demean(a, rinv, G):
    sums = np.zeros((G, a.shape[1])); np.add.at(sums, rinv, a)
    cnts = np.bincount(rinv, minlength=G).astype(float)
    return a - sums[rinv] / cnts[rinv, None]


def reg_cluster(fe_groups, cluster, X, y, extra_fe=None):
    """Absorb fe_groups (+ extra_fe) by alternating demeaning; cluster the
    sandwich at `cluster` (CR0). fe_groups and cluster may differ."""
    Xd = np.asarray(X, float); yd = np.asarray(y, float)[:, None]
    fe = [(idx(g)[0], idx(g)[1]) for g in fe_groups] + \
         [(idx(g)[0], idx(g)[1]) for g in (extra_fe or [])]
    for _ in range(80):
        for r_g, G_g in fe:
            Xd = within_demean(Xd, r_g, G_g); yd = within_demean(yd, r_g, G_g)
    ydv = yd.ravel(); XtX = Xd.T @ Xd
    beta = np.linalg.lstsq(XtX, Xd.T @ ydv, rcond=None)[0]
    e = ydv - Xd @ beta
    rinv, G = idx(cluster)
    order = np.argsort(rinv); rs = rinv[order]; Xs = Xd[order]; es = e[order]
    K = Xd.shape[1]; meat = np.zeros((K, K)); start = 0
    for g in range(G):
        end = start + int((rs == g).sum())
        sg = Xs[start:end].T @ es[start:end]
        meat += np.outer(sg, sg); start = end
    N = Xd.shape[0]
    c = G / (G - 1) * (N - 1) / max(N - K, 1)
    V = c * np.linalg.pinv(XtX) @ meat @ np.linalg.pinv(XtX)
    return beta, np.sqrt(np.abs(np.diag(V)))


def load_panel(src="openalex"):
    rt = pd.read_parquet(os.path.join(DATA, f"{src}_scholarlymigration_subnational_gender_and_field.parquet"))
    fl = pd.read_parquet(os.path.join(DATA, f"{src}_scholarlymigration_subnational_flows_gender_and_field.parquet"))
    rta = rt[(rt.gender == "all") & (rt.field == "all")].copy()
    fla = fl[(fl.gender == "all") & (fl.field == "all")].copy()
    fla["is_intl"] = fla.area_from.str.split(".").str[0] != fla.area_to.str.split(".").str[0]
    pre_pop = (rta[(rta.year >= 2015) & (rta.year <= 2019)]
               .groupby("area")["padded_population_of_researchers"].mean())
    keep = pre_pop[pre_pop >= 20].index
    pan = rta[["year", "area", "iso3code", "number_of_inmigrations"]].copy()
    pan = pan[pan.area.isin(keep)].copy()
    pan["y_in"] = np.log1p(pan.number_of_inmigrations)
    fp = fla[(fla.year >= 2015) & (fla.year <= 2019)]
    pan["share"] = pan.area.map(fp[fp.is_intl].groupby("area_to").n_migrations.sum() /
                                fp.groupby("area_to").n_migrations.sum())
    YRS = tuple(range(2015, 2025))
    pan = pan[pan.year.isin(YRS) & pan.share.notna()].copy()
    cnt = pan.groupby("area").year.count()
    pan = pan[pan.area.isin(cnt[cnt == len(YRS)].index)].copy()
    sd = pan.groupby("area").share.first().std()
    pan["sh"] = pan.share / sd
    return pan, sd, float(sd)


pan, sd_series, sd_val = load_panel("openalex")
YRS = sorted(pan.year.unique()); YD = [str(y) for y in YRS if y != 2019]
Dyear = pd.get_dummies(pan.year, prefix="Y", prefix_sep="").reindex(
    columns=["Y" + b for b in YD], fill_value=0).astype(float)
Dint = pan.sh.values[:, None] * Dyear.values
X_main = np.hstack([Dyear.values, Dint])
cy = (pan.iso3code.astype(str) + "_" + pan.year.astype(str)).values

out["exposure_sd"] = round(sd_val, 4)
out["panel_n"] = int(len(pan)); out["panel_regions"] = int(pan.area.nunique())

# S1: baseline, clustering by region vs country
b1r, s1r = reg_cluster([pan.area.values], pan.area.values, X_main, pan.y_in.values)
b1c, s1c = reg_cluster([pan.area.values], pan.iso3code.values, X_main, pan.y_in.values)
# S2: country x year FE, clustering by region vs country
b2r, s2r = reg_cluster([pan.area.values], pan.area.values, Dint, pan.y_in.values, extra_fe=[cy])
b2c, s2c = reg_cluster([pan.area.values], pan.iso3code.values, Dint, pan.y_in.values, extra_fe=[cy])
# S3: multi-region countries only, clustered by region vs country
per_country = pan.groupby("iso3code").area.nunique()
multi = per_country[per_country >= 2].index
sub = pan[pan.iso3code.isin(multi)].copy()
Xs = np.hstack([Dyear.loc[sub.index].values, sub.sh.values[:, None] * Dyear.loc[sub.index].values])
b3r, s3r = reg_cluster([sub.area.values], sub.area.values, Xs, sub.y_in.values)
b3c, s3c = reg_cluster([sub.area.values], sub.iso3code.values, Xs, sub.y_in.values)
out["multi_only_n_regions"] = int(sub.area.nunique())


def pack(beta, se, year_dummies=True):
    k = len(YD)
    b = beta[k:] if year_dummies else beta
    s = se[k:] if year_dummies else se
    return {y: [round(float(v), 4), round(float(sv), 4),
                round(float(2 * (1 - 0.5 * (1 + math.erf(abs(v / sv) / math.sqrt(2))))), 4)]
            for y, v, sv in zip(YD, b, s)}


out["S1_baseline_region_clustered"] = pack(b1r, s1r)
out["S1_baseline_country_clustered"] = pack(b1c, s1c)
out["S2_ctryyear_region_clustered"] = pack(b2r, s2r, year_dummies=False)
out["S2_ctryyear_country_clustered"] = pack(b2c, s2c, year_dummies=False)
out["S3_multionly_region_clustered"] = pack(b3r, s3r)
out["S3_multionly_country_clustered"] = pack(b3c, s3c)

# level gradient with country FE only (no region FE), country-clustered
bl, sl = reg_cluster([pan.iso3code.values], pan.iso3code.values, X_main, pan.y_in.values)
out["levelgradient_countryFE_country_clustered"] = pack(bl, sl)

with open(os.path.join(RES, "robustness_round3.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=str)
for k, v in out.items():
    if isinstance(v, dict):
        print(f"--- {k}")
        for y, t in v.items():
            print(f"    {y}: beta={t[0]:+.4f} se={t[1]:.4f} p={t[2]:.4f}")
    else:
        print(f"{k}: {v}")
print("DONE")
