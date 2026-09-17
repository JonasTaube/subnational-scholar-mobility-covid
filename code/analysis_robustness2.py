# -*- coding: utf-8 -*-
"""
Round-2 addendum:
 A) Primary trajectory classification with post window = 2022 only (avoids truncation)
    + full diagnostics (mean/median/IQR, sign shares, absolute net losses)
 B) country x year FE diagnostics: within-country exposure variation, multi-region countries,
    excluding single-region countries, country-clustered SEs
 C) Cross-source (OpenAlex vs Scopus) comparability diagnostics
 D) Regenerate Figure 3 with the primary classification
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Set SMD2_BASE to your working copy, or keep this checkout's layout
import os
BASE = os.environ.get("SMD2_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "01_原始数据", "smd2_repo", "data_input", "2026_V2")
RES = os.path.join(BASE, "03_分析与图表", "results")
FIG = os.path.join(BASE, "03_分析与图表", "figures")
for f in ["C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf"]:
    if os.path.exists(f): font_manager.fontManager.addfont(f)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "Times New Roman",
                     "font.size": 9, "axes.linewidth": 0.8, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 150})
out = {}

def idx(arr):
    _, rinv = np.unique(arr, return_inverse=True)
    return rinv, int(rinv.max()) + 1

def within_demean(a, rinv, G):
    sums = np.zeros((G, a.shape[1])); np.add.at(sums, rinv, a)
    cnts = np.bincount(rinv, minlength=G).astype(float)
    return a - sums[rinv] / cnts[rinv, None]

def reg_cluster(reg, X, y, extra_fe=None):
    Xd = np.asarray(X, float); yd = np.asarray(y, float)[:, None]
    rinv, G = idx(reg)
    groups = [(rinv, G)] + [(idx(g)[0], idx(g)[1]) for g in (extra_fe or [])]
    for _ in range(60):
        for r_g, G_g in groups:
            Xd = within_demean(Xd, r_g, G_g); yd = within_demean(yd, r_g, G_g)
    ydv = yd.ravel(); XtX = Xd.T @ Xd
    beta = np.linalg.lstsq(XtX, Xd.T @ ydv, rcond=None)[0]
    e = ydv - Xd @ beta
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

def load(src):
    rt = pd.read_parquet(os.path.join(DATA, f"{src}_scholarlymigration_subnational_gender_and_field.parquet"))
    fl = pd.read_parquet(os.path.join(DATA, f"{src}_scholarlymigration_subnational_flows_gender_and_field.parquet"))
    return rt, fl

def build(src, YRS=tuple(range(2015, 2025)), balanced=True):
    rt, fl = load(src)
    rta = rt[(rt.gender == "all") & (rt.field == "all")].copy()
    fla = fl[(fl.gender == "all") & (fl.field == "all")].copy()
    fla["is_intl"] = fla.area_from.str.split(".").str[0] != fla.area_to.str.split(".").str[0]
    pre_pop = (rta[(rta.year >= 2015) & (rta.year <= 2019)]
               .groupby("area")["padded_population_of_researchers"].mean())
    keep = pre_pop[pre_pop >= 20].index
    pan = rta[["year", "area", "iso3code", "padded_population_of_researchers", "number_of_inmigrations",
               "number_of_outmigrations", "netmigration", "netmigrationrate"]].copy()
    pan = pan[pan.area.isin(keep)].copy()
    pan["y_in"] = np.log1p(pan.number_of_inmigrations)
    fp = fla[(fla.year >= 2015) & (fla.year <= 2019)]
    intl_in = fp[fp.is_intl].groupby("area_to").n_migrations.sum()
    all_in = fp.groupby("area_to").n_migrations.sum()
    pan["share"] = pan.area.map(intl_in / all_in)
    pan["pop_pre"] = pan.area.map(pre_pop)
    if balanced:
        pan = pan[pan.year.isin(YRS) & pan.share.notna()].copy()
        cnt = pan.groupby("area").year.count()
        bal = cnt[cnt == len(list(YRS))].index
        pan = pan[pan.area.isin(bal)].copy()
    else:
        pan = pan[pan.share.notna() | True].copy()
    sd = pan.groupby("area").share.first().std()
    pan["sh"] = pan.share / sd
    return pan, pre_pop, rta, fla, sd

print("=== A) primary classification: post window = 2022 only ===")
rt_oa, fl_oa = load("openalex")
rta = rt_oa[(rt_oa.gender == "all") & (rt_oa.field == "all")].copy()
pre_pop = (rta[(rta.year >= 2015) & (rta.year <= 2019)]
           .groupby("area")["padded_population_of_researchers"].mean())
keep = pre_pop[pre_pop >= 20].index
p = rta[rta.area.isin(keep)].copy()
p["nr"] = p.netmigration / p.area.map(pre_pop) * 1000
def wm(y0, y1):
    return p[(p.year >= y0) & (p.year <= y1)].groupby("area").nr.mean()
pre, shk = wm(2015, 2019), wm(2020, 2021)
post22 = wm(2022, 2022)
post24 = wm(2022, 2024)
def cl(r):
    if r.shock < r.pre and r.post >= r.pre:  return "Temporary decline, rebounded"
    if r.shock < r.pre and r.post < r.pre:   return "Relative persistent decline"
    if r.shock >= r.pre and r.post >= r.pre: return "Relative persistent improvement"
    return "Improvement then reversal"
def build_traj(post, tag):
    t = pd.concat([pre, shk, post], axis=1, keys=["pre", "shock", "post"]).dropna()
    t["class"] = t.apply(cl, axis=1)
    d = t.groupby("class").agg(
        n=("pre", "size"), mean_pre=("pre", "mean"), mean_shock=("shock", "mean"), mean_post=("post", "mean"),
        med_pre=("pre", "median"), med_post=("post", "median"),
        q1_post=("post", lambda s: s.quantile(0.25)), q3_post=("post", lambda s: s.quantile(0.75)),
        share_pre_neg=("pre", lambda s: float((s < 0).mean())),
        share_post_neg=("post", lambda s: float((s < 0).mean())),
    ).round(2)
    d["share"] = (d.n / d.n.sum()).round(3)
    t.to_csv(os.path.join(RES, f"traj_{tag}_region_level.csv"))
    d.to_csv(os.path.join(RES, f"traj_{tag}_diagnostics.csv"))
    return t, d

t22, d22 = build_traj(post22, "primary_2022")
t24, d24 = build_traj(post24, "alt_2022_2024")
out["primary_counts"] = t22["class"].value_counts().to_dict()
out["alt_counts"] = t24["class"].value_counts().to_dict()
out["primary_n"] = int(len(t22))
order = ["Relative persistent improvement", "Relative persistent decline", "Improvement then reversal", "Temporary decline, rebounded"]
out["primary_diag"] = d22.reindex(order).reset_index().to_dict("records")
out["alt_diag"] = d24.reindex(order).reset_index().to_dict("records")

# absolute net loss (policy targeting)
p22 = p[(p.year == 2022)].set_index("area")
post_neg = t22[t22["post"] < 0]
loss = p22.loc[post_neg.index, "netmigration"]
out["absolute"] = {
    "n_regions_negative_post_rate": int(len(post_neg)),
    "share_of_classified": round(len(post_neg) / len(t22), 3),
    "n_regions_negative_post_net_flow": int((loss < 0).sum()),
    "total_net_loss_2022_among_negatives": int(-loss[loss < 0].sum()),
    "of_which_improvement_class": int((post_neg["class"] == "Relative persistent improvement").sum()),
    "of_which_decline_class": int((post_neg["class"] == "Relative persistent decline").sum()),
}
# top absolute losers (for illustration)
top = loss[loss < 0].sort_values().head(10)
out["top_absolute_losers_2022"] = [{"area": a, "net_2022": int(v)} for a, v in top.items()]

print("=== B) country x year FE diagnostics ===")
pan_oa, pre_pop_oa, rta_oa, fla_oa, sd_oa = build("openalex")
per_country = pan_oa.groupby("iso3code").area.nunique()
out["cy_diag"] = {
    "n_regions": int(pan_oa.area.nunique()), "n_countries": int(pan_oa.iso3code.nunique()),
    "countries_with_ge2_regions": int((per_country >= 2).sum()),
    "share_regions_in_multiregion_countries": round(
        float(per_country[per_country >= 2].sum() / per_country.sum()), 3),
    "countries_with_1_region": int((per_country == 1).sum()),
}
wc = pan_oa.groupby(["iso3code", "area"]).share.first().groupby("iso3code").std().dropna()
out["cy_diag"]["within_country_exposure_sd_median"] = round(float(wc.median()), 4)
out["cy_diag"]["within_country_exposure_sd_q1"] = round(float(wc.quantile(0.25)), 4)
out["cy_diag"]["within_country_exposure_sd_q3"] = round(float(wc.quantile(0.75)), 4)

YRS = sorted(pan_oa.year.unique()); YD = [str(y) for y in YRS if y != 2019]
YRS2 = (2020, 2021, 2022, 2023, 2024)
D = pd.get_dummies(pan_oa.year, prefix="Y", prefix_sep="").reindex(
    columns=["Y" + b for b in YD], fill_value=0).astype(float)
Xb = np.hstack([D.values, pan_oa.sh.values[:, None] * D.values])
# exclude single-region countries
multi = per_country[per_country >= 2].index
sub = pan_oa[pan_oa.iso3code.isin(multi)].copy()
Dsub = D.loc[sub.index]
Xsub = np.hstack([Dsub.values, sub.sh.values[:, None] * Dsub.values])
bb, sb = reg_cluster(sub.area.values, Xsub, sub.y_in.values)
out["multi_country_only_did"] = {b: (round(float(v), 4), round(float(s), 4))
                                for b, v, s in zip(YD, bb[len(YD):], sb[len(YD):])}
out["multi_country_only_n"] = int(sub.area.nunique())

# country-clustered SEs
b_bc, s_bc = reg_cluster(pan_oa.iso3code.values, Xb, pan_oa.y_in.values)
out["baseline_country_clustered"] = {b: (round(float(v), 4), round(float(s), 4))
                                     for b, v, s in zip(YD, b_bc[len(YD):], s_bc[len(YD):])}
cy = pan_oa.iso3code.astype(str) + "_" + pan_oa.year.astype(str)
Dcy = pan_oa.sh.values[:, None] * D.values
b_cy, s_cy_reg = reg_cluster(pan_oa.area.values, Dcy, pan_oa.y_in.values, extra_fe=[cy.values])
b_cyc, s_cyc = reg_cluster(pan_oa.iso3code.values, Dcy, pan_oa.y_in.values, extra_fe=[cy.values])
out["country_year_region_clustered"] = {b: (round(float(v), 4), round(float(s), 4)) for b, v, s in zip(YD, b_cy, s_cy_reg)}
out["country_year_country_clustered"] = {b: (round(float(v), 4), round(float(s), 4)) for b, v, s in zip(YD, b_cyc, s_cyc)}

print("=== C) cross-source comparability ===")
rt_sc, fl_sc = load("scopus")
rta_sc = rt_sc[(rt_sc.gender == "all") & (rt_sc.field == "all")].copy()
a_oa = set(rta.area.unique()); a_sc = set(rta_sc.area.unique())
out["cross_source"] = {
    "regions_openalex": len(a_oa), "regions_scopus": len(a_sc),
    "regions_shared": len(a_oa & a_sc),
    "share_openalex_also_scopus": round(len(a_oa & a_sc) / len(a_oa), 3),
    "countries_openalex": int(rta.iso3code.nunique()), "countries_scopus": int(rta_sc.iso3code.nunique()),
}
o19 = rta[rta.year == 2019].set_index("area").number_of_inmigrations
o22 = rta[rta.year == 2022].set_index("area").number_of_inmigrations
s19 = rta_sc[rta_sc.year == 2019].set_index("area").number_of_inmigrations
s22 = rta_sc[rta_sc.year == 2022].set_index("area").number_of_inmigrations
common = sorted(set(o19.index) & set(s19.index) & set(o22.index) & set(s22.index))
common = [a for a in common if pre_pop.get(a, 0) >= 20]
d_oa = (np.log1p(o22[common].astype(float)) - np.log1p(o19[common].astype(float)))
d_sc = (np.log1p(s22[common].astype(float)) - np.log1p(s19[common].astype(float)))
r = float(np.corrcoef(d_oa.values, d_sc.values)[0, 1])
out["cross_source"]["n_regions_common_pop20"] = len(common)
out["cross_source"]["corr_region_log_change_2019_2022"] = round(r, 3)
# level ratio
out["cross_source"]["event_ratio_2019"] = round(float(rta[rta.year == 2019].number_of_inmigrations.sum() /
                                                       rta_sc[rta_sc.year == 2019].number_of_inmigrations.sum()), 2)

print("=== D) Figure 3 regeneration (primary = 2022-only post window) ===")
cols = {"Relative persistent improvement": "#27ae60", "Temporary decline, rebounded": "#2471a3",
        "Improvement then reversal": "#e67e22", "Relative persistent decline": "#c0392b"}
plot_order = ["Relative persistent improvement", "Temporary decline, rebounded", "Improvement then reversal", "Relative persistent decline"]
cnt = t22["class"].value_counts()
p["cl"] = p.area.map(t22["class"])
mt = (p[p.year.between(2015, 2024)][["year", "area", "nr", "cl"]].dropna()
      .groupby(["cl", "year"]).nr.mean().reset_index())
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axes[0]
ax.barh(range(len(plot_order)), [cnt.get(o, 0) for o in plot_order], color=[cols[o] for o in plot_order])
ax.set_yticks(range(len(plot_order))); ax.set_yticklabels(plot_order, fontsize=8)
ax.set_xlabel("Number of subnational regions"); ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
for o in plot_order:
    dd = mt[mt["cl"] == o]
    ax.plot(dd.year, dd.nr, lw=1.4, color=cols[o], label=o)
ax.axhline(0, color="#888888", lw=0.7)
ax.axvspan(2020, 2021.9, color="#f0c94833", zorder=0)
ax.set_xlabel("Year"); ax.set_ylabel("Mean net migration rate*")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_trajectories.pdf")); fig.savefig(os.path.join(FIG, "fig3_trajectories.png"), dpi=300)
plt.close(fig)

with open(os.path.join(RES, "robustness_round2.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print("DONE")
