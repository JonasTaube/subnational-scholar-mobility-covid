# -*- coding: utf-8 -*-
"""
Robustness pack for SMD 2.0 COVID paper:
 1) Scopus replication (aggregates, shock path, DID, trajectories)
 2) Endpoint sensitivity (2015-2022)
 3) Controls: region size x year, field composition x year, country x year FE, region linear trends
 4) Trajectory denominator / endpoint sensitivities + agreement
 5) Trajectory class diagnostics (levels & signs) for renamed classes
 6) Coverage diagnostics OpenAlex vs Scopus
 7) China international inflow series check
Outputs: results/robust_*.csv|json
"""
import json, os
import numpy as np
import pandas as pd

# Set SMD2_BASE to your working copy, or keep this checkout's layout
import os
BASE = os.environ.get("SMD2_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "01_原始数据", "smd2_repo", "data_input", "2026_V2")
RES = os.path.join(BASE, "03_分析与图表", "results")
out = {}

# ---------------- helpers ----------------
def idx(arr):
    _, rinv = np.unique(arr, return_inverse=True)
    return rinv, int(rinv.max()) + 1

def within_demean(a, rinv, G):
    sums = np.zeros((G, a.shape[1]))
    np.add.at(sums, rinv, a)
    cnts = np.bincount(rinv, minlength=G).astype(float)
    return a - sums[rinv] / cnts[rinv, None]

def reg_cluster(reg, X, y, extra_fe=None, detrend=None):
    """OLS with region FE (+ optional extra FE groups, optional region linear trends),
    cluster-robust (CR0) SE by region."""
    Xd = np.asarray(X, float); yd = np.asarray(y, float)[:, None]
    rinv, G = idx(reg)
    groups = [(rinv, G)] + [(idx(g)[0], idx(g)[1]) for g in (extra_fe or [])]
    for _ in range(60):
        for r_g, G_g in groups:
            Xd = within_demean(Xd, r_g, G_g); yd = within_demean(yd, r_g, G_g)
    if detrend is not None:
        t = np.asarray(detrend, float)  # detrend carries the year vector
        for arr_name in ("X", "y"):
            pass
        # within-region projection on [1, t]
        t_col = t[:, None]
        for M in (Xd, yd):
            s_t = np.zeros((G, 1)); s_tt = np.zeros((G, 1)); s_a = np.zeros((G, M.shape[1])); s_ta = np.zeros((G, M.shape[1]))
            np.add.at(s_t, rinv, t_col); np.add.at(s_tt, rinv, t_col ** 2)
            np.add.at(s_a, rinv, M);    np.add.at(s_ta, rinv, t_col * M)
            n = np.bincount(rinv, minlength=G).astype(float)[:, None]
            s_t, s_tt, s_a, s_ta = s_t / n, s_tt / n, s_a / n, s_ta / n
            var_t = s_tt - s_t ** 2
            cov = s_ta - s_t * s_a
            slope = cov / var_t
            pred = s_a[rinv] + slope[rinv] * (t_col - s_t[rinv])
            M -= pred[rinv]
        Xd = Xd
        yd = yd
    ydv = yd.ravel()
    XtX = Xd.T @ Xd
    beta = np.linalg.lstsq(XtX, Xd.T @ ydv, rcond=None)[0]
    e = ydv - Xd @ beta
    order = np.argsort(rinv); rs = rinv[order]; Xs = Xd[order]; es = e[order]
    K = Xd.shape[1]; meat = np.zeros((K, K)); start = 0
    for g in range(G):
        end = start + int((rs == g).sum())
        sg = Xs[start:end].T @ es[start:end]
        meat += np.outer(sg, sg)
        start = end
    N = Xd.shape[0]
    c = G / (G - 1) * (N - 1) / max(N - K, 1)
    V = c * np.linalg.pinv(XtX) @ meat @ np.linalg.pinv(XtX)
    return beta, np.sqrt(np.abs(np.diag(V)))

def load(src):
    rt = pd.read_parquet(os.path.join(DATA, f"{src}_scholarlymigration_subnational_gender_and_field.parquet"))
    fl = pd.read_parquet(os.path.join(DATA, f"{src}_scholarlymigration_subnational_flows_gender_and_field.parquet"))
    return rt, fl

def build_panel(src, YRS=(2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)):
    rt, fl = load(src)
    rta = rt[(rt.gender == "all") & (rt.field == "all")].copy()
    fla = fl[(fl.gender == "all") & (fl.field == "all")].copy()
    fla["is_intl"] = fla.area_from.str.split(".").str[0] != fla.area_to.str.split(".").str[0]
    pre_pop = (rta[(rta.year >= 2015) & (rta.year <= 2019)]
               .groupby("area")["padded_population_of_researchers"].mean())
    keep = pre_pop[pre_pop >= 20].index
    pan = rta[["year", "area", "iso3code", "padded_population_of_researchers",
               "number_of_inmigrations", "number_of_outmigrations",
               "netmigration", "netmigrationrate"]].copy()
    pan = pan[pan.area.isin(keep)].copy()
    pan["y_in"] = np.log1p(pan.number_of_inmigrations)
    pan["y_out"] = np.log1p(pan.number_of_outmigrations)
    fp = fla[(fla.year >= 2015) & (fla.year <= 2019)]
    intl_in = fp[fp.is_intl].groupby("area_to").n_migrations.sum()
    all_in = fp.groupby("area_to").n_migrations.sum()
    exposure = (intl_in / all_in)
    pan["share"] = pan.area.map(exposure)
    pop19 = rta[rta.year == 2019].set_index("area").padded_population_of_researchers
    pan["pop19"] = pan.area.map(pop19)
    pan["pop_pre"] = pan.area.map(pre_pop)
    pan = pan[pan.year.isin(YRS) & pan.share.notna()].copy()
    cnt = pan.groupby("area").year.count()
    bal = cnt[cnt == len(list(YRS))].index
    pan = pan[pan.area.isin(bal)].copy()
    sd = pan.groupby("area").share.first().std()
    pan["sh"] = pan.share / sd
    return pan, pre_pop, rta, fla, sd

def path_and_did(pan, label, extra_fe=None, detrend=None):
    YRS = sorted(pan.year.unique()); REF = 2019
    YD = [str(y) for y in YRS if y != REF]
    D = pd.get_dummies(pan.year, prefix="Y", prefix_sep="").reindex(
        columns=["Y" + b for b in YD], fill_value=0).astype(float)
    res = {}
    bA, sA = reg_cluster(pan.area.values, D.values, pan.y_in.values,
                         extra_fe=extra_fe, detrend=detrend)
    res["path_in"] = {b: (round(float(v), 4), round(float(s), 4)) for b, v, s in zip(YD, bA, sA)}
    Xb = np.hstack([D.values, (pan.sh.values[:, None] * D.values)])
    eb = extra_fe
    if extra_fe is not None and label == "countryxyear":
        Xb = (pan.sh.values[:, None] * D.values)  # year effects absorbed by country-year FE
    bb, sb = reg_cluster(pan.area.values, Xb, pan.y_in.values,
                         extra_fe=extra_fe, detrend=detrend)
    k = 0 if Xb.shape[1] == len(YD) else len(YD)
    res["did"] = {b: (round(float(v), 4), round(float(s), 4)) for b, v, s in zip(YD, bb[k:], sb[k:])}
    out[label] = res
    return res

def classify(pan, pre_pop, post_years=(2022, 2023, 2024), denom="pre"):
    p = pan.copy()
    if denom == "pre":
        d = p.area.map(pre_pop)
    elif denom == "pop19":
        d = p.pop19
    p["nr"] = p.netmigration / d * 1000
    def wm(y0, y1):
        return p[(p.year >= y0) & (p.year <= y1)].groupby("area").nr.mean()
    pre = wm(2015, 2019); shk = wm(2020, 2021); post = wm(post_years[0], post_years[-1])
    t = pd.concat([pre, shk, post], axis=1, keys=["pre", "shock", "post"]).dropna()
    def cl(r):
        if r.shock < r.pre and r.post >= r.pre:  return "Temporary decline, rebounded"
        if r.shock < r.pre and r.post < r.pre:   return "Relative persistent decline"
        if r.shock >= r.pre and r.post >= r.pre: return "Relative persistent improvement"
        return "Improvement then reversal"
    t["class"] = t.apply(cl, axis=1)
    return t

# ================= 1. OpenAlex baseline + sensitivities =================
print("loading OpenAlex...")
pan_oa, pre_pop_oa, rta_oa, fla_oa, sd_oa = build_panel("openalex")
print("openalex panel regions:", pan_oa.area.nunique())
path_and_did(pan_oa, "oa_baseline")

# endpoint 2022
pan_22 = pan_oa[pan_oa.year <= 2022].copy()
path_and_did(pan_22, "oa_end2022")

# size control (pre population, region size x year)
YRS = sorted(pan_oa.year.unique()); YD = [str(y) for y in YRS if y != 2019]
D = pd.get_dummies(pan_oa.year, prefix="Y", prefix_sep="").reindex(
    columns=["Y" + b for b in YD], fill_value=0).astype(float)
lp = np.log(pan_oa.pop_pre.values)[:, None]
X = np.hstack([D.values, pan_oa.sh.values[:, None] * D.values, (lp * D.values)])
b, s = reg_cluster(pan_oa.area.values, X, pan_oa.y_in.values)
out["oa_size_control"] = {b_: (round(float(v), 4), round(float(se), 4))
                          for b_, v, se in zip(YD, b[len(YD):2 * len(YD)], s[len(YD):2 * len(YD)])}

# field composition x year
rta = pd.read_parquet(os.path.join(DATA, "openalex_scholarlymigration_subnational_gender_and_field.parquet"))
rtf = rta[(rta.gender == "all") & (rta.field != "all") & (rta.year.between(2015, 2019))]
fld = rtf.groupby(["area", "field"]).number_of_inmigrations.sum().unstack(fill_value=0)
fld = fld.div(fld.sum(axis=1).replace(0, np.nan), axis=0)
fld.columns = ["f_" + c for c in fld.columns]
fld = fld.dropna()
pan_f = pan_oa.join(fld, on="area").dropna(subset=list(fld.columns))
Xf = np.hstack([D.loc[pan_f.index].values,
                pan_f.sh.values[:, None] * D.loc[pan_f.index].values]
               + [pan_f[c].values[:, None] * D.loc[pan_f.index].values for c in fld.columns[:2]])
bf, sf = reg_cluster(pan_f.area.values, Xf, pan_f.y_in.values)
out["oa_field_control"] = {b_: (round(float(v), 4), round(float(se), 4))
                           for b_, v, se in zip(YD, bf[len(YD):2 * len(YD)], sf[len(YD):2 * len(YD)])}
out["oa_field_control_note"] = f"sample n={pan_f.area.nunique()} regions (field shares available)"

# country x year FE (year effects absorbed)
cy = pan_oa.iso3code.astype(str) + "_" + pan_oa.year.astype(str)
Dcy = pan_oa.sh.values[:, None] * pd.get_dummies(pan_oa.year, prefix="Y", prefix_sep="").reindex(
    columns=["Y" + b for b in YD], fill_value=0).astype(float).values
bcy, scy = reg_cluster(pan_oa.area.values, Dcy, pan_oa.y_in.values, extra_fe=[cy.values])
out["oa_country_year"] = {b_: (round(float(v), 4), round(float(se), 4))
                          for b_, v, se in zip(YD, bcy, scy)}

# region linear trends
path_and_did(pan_oa, "oa_region_trends", detrend=pan_oa.year.values.astype(float))

# ================= 2. Scopus replication =================
print("loading Scopus...")
pan_sc, pre_pop_sc, rta_sc, fla_sc, sd_sc = build_panel("scopus")
print("scopus panel regions:", pan_sc.area.nunique())
path_and_did(pan_sc, "sc_baseline")
g_sc = rta_sc[(rta_sc.gender == "all") & (rta_sc.field == "all")].groupby("year").number_of_inmigrations.sum()
out["scopus_totals"] = {int(y): int(g_sc.get(y, 0)) for y in (2019, 2020, 2021, 2022)}
fy_sc = fla_sc.groupby(["year", "is_intl"]).n_migrations.sum().unstack()
fy_sc.columns = ["internal", "international"]
out["scopus_flows"] = {int(y): {"internal": int(fy_sc.loc[y, "internal"]),
                                "international": int(fy_sc.loc[y, "international"])}
                       for y in (2019, 2020, 2021, 2022) if y in fy_sc.index}
t_sc = classify(pan_sc, pre_pop_sc)
out["scopus_traj"] = t_sc["class"].value_counts().to_dict()
out["scopus_n_regions"] = int(len(t_sc))

# ================= 3. Trajectory diagnostics & sensitivities (OpenAlex) =================
tb = classify(pan_oa, pre_pop_oa)
tb.to_csv(os.path.join(RES, "robust_traj_baseline.csv"))
diag = tb.groupby("class").agg(
    n=("pre", "size"),
    mean_pre=("pre", "mean"), mean_shock=("shock", "mean"), mean_post=("post", "mean"),
    share_pre_negative=("pre", lambda s: float((s < 0).mean())),
    share_post_negative=("post", lambda s: float((s < 0).mean())),
).round(3)
diag["share"] = (diag.n / diag.n.sum()).round(3)
diag.to_csv(os.path.join(RES, "robust_traj_class_diagnostics.csv"))
out["traj_diagnostics"] = diag.reset_index().to_dict("records")

# sensitivities
variants = {
    "denom_pop19": classify(pan_oa, pre_pop_oa, denom="pop19"),
    "post_2022_only": classify(pan_oa, pre_pop_oa, post_years=(2022, 2022)),
}
for k, v in variants.items():
    v.to_csv(os.path.join(RES, f"robust_traj_{k}.csv"))
    agree = (v["class"].reindex(tb.index) == tb["class"]).mean()
    out[f"traj_{k}"] = {"counts": v["class"].value_counts().to_dict(),
                        "n": int(len(v)), "agreement_with_baseline": round(float(agree), 3)}

# ================= 4. Coverage diagnostics =================
def coverage(rta, fla, name):
    areas = rta[(rta.gender == "all") & (rta.field == "all")]
    n_reg = areas.area.nunique()
    yrs = [int(areas.year.min()), int(areas.year.max())]
    n_country = areas.iso3code.nunique()
    per_country = areas.groupby("iso3code").area.nunique()
    tot19 = int(areas[areas.year == 2019].number_of_inmigrations.sum())
    fa = fla[(fla.gender == "all") & (fla.field == "all")] if "gender" in fla.columns else fla
    fl_yrs = [int(fa.year.min()), int(fa.year.max())]
    return {"source": name, "n_regions": int(n_reg), "years_counts": yrs, "years_flows": fl_yrs,
            "n_countries": int(n_country),
            "countries_gt20_regions": int((per_country > 20).sum()),
            "median_regions_per_country": float(per_country.median()),
            "inmigrations_2019": tot19,
            "regions_1_to_3": int((per_country <= 3).sum())}
out["coverage"] = [coverage(rta_oa, fla_oa, "OpenAlex"), coverage(rta_sc, fla_sc, "Scopus")]

# ================= 5. China international inflow check =================
cn_rt = rta_oa[(rta_oa.gender == "all") & (rta_oa.field == "all") & (rta_oa.iso3code == "CHN")]
cn_y = cn_rt.groupby("year").agg(inflow=("number_of_inmigrations", "sum"),
                                 outflow=("number_of_outmigrations", "sum"),
                                 net=("netmigration", "sum"))
cn_y.to_csv(os.path.join(RES, "robust_china_annual.csv"))
cn_fl = fla_oa[(fla_oa.area_to.str.startswith("CN.")) | (fla_oa.area_from.str.startswith("CN."))]
ci = (cn_fl[cn_fl.area_to.str.startswith("CN.")]
      .groupby(["year", "is_intl"]).n_migrations.sum().unstack())
ci.columns = ["internal_in", "international_in"]
ci.to_csv(os.path.join(RES, "robust_china_flows.csv"))
i19, i20, i21, i22 = [int(ci.loc[y, "international_in"]) for y in (2019, 2020, 2021, 2022)]
out["china_check"] = {
    "flows_intl_in": {"2019": i19, "2020": i20, "2021": i21, "2022": i22,
                      "pct_2020_vs_2019": round(100 * (i20 - i19) / i19, 1),
                      "pct_2021_vs_2019": round(100 * (i21 - i19) / i19, 1),
                      "pct_2022_vs_2019": round(100 * (i22 - i19) / i19, 1)},
    "rates_intl_in_2019": int(cn_y.loc[2019, "inflow"]), "rates_intl_in_2020": int(cn_y.loc[2020, "inflow"]),
    "rates_intl_in_2021": int(cn_y.loc[2021, "inflow"]), "rates_intl_in_2022": int(cn_y.loc[2022, "inflow"]),
    "rates_in_2015": int(cn_y.loc[2015, "inflow"]),
}
out["china_annual_rates"] = {int(y): int(v) for y, v in cn_y["inflow"].items()}

with open(os.path.join(RES, "robustness.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print("DONE")
