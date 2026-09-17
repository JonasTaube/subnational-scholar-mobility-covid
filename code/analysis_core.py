# -*- coding: utf-8 -*-
"""
SMD 2.0 core analysis: COVID shock x global subnational scholar mobility
Outputs: results CSVs + headline JSON + 5 journal-style figures (PDF vector + PNG)
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
RES  = os.path.join(BASE, "03_分析与图表", "results")
FIG  = os.path.join(BASE, "03_分析与图表", "figures")
for d in (RES, FIG): os.makedirs(d, exist_ok=True)

# ---------- matplotlib journal style ----------
for f in ["C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf"]:
    if os.path.exists(f): font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "Times New Roman", "font.size": 9,
    "axes.linewidth": 0.8, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 150,
})
C_IN, C_OUT, C_NET = "#c0392b", "#2471a3", "#7d3c98"
C_INTL, C_INT = "#c0392b", "#2471a3"
SHADE = (2020, 2021.9)

def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)

# ================= 1. LOAD =================
print("loading rates...")
rt = pd.read_parquet(os.path.join(DATA, "openalex_scholarlymigration_subnational_gender_and_field.parquet"))
rt = rt[(rt.gender == "all") & (rt.field == "all")].copy()
rt["country_prefix"] = rt["area"].str.split(".").str[0]

print("loading flows...")
fl = pd.read_parquet(os.path.join(DATA, "openalex_scholarlymigration_subnational_flows_gender_and_field.parquet"))
fl = fl[(fl.gender == "all") & (fl.field == "all")].copy()
fl["pref_from"] = fl["area_from"].str.split(".").str[0]
fl["pref_to"]   = fl["area_to"].str.split(".").str[0]
fl["is_international"] = fl["pref_from"] != fl["pref_to"]

headline = {}

# ================= 2. FIG1: global flows time series =================
g = rt.groupby("year").agg(inflow=("number_of_inmigrations", "sum"),
                           outflow=("number_of_outmigrations", "sum")).reset_index()
g = g[g.year >= 1998]
g.to_csv(os.path.join(RES, "fig1_global_flows.csv"), index=False)

i19 = g.loc[g.year == 2019, "inflow"].iloc[0]
i20 = g.loc[g.year == 2020, "inflow"].iloc[0]
i21 = g.loc[g.year == 2021, "inflow"].iloc[0]
i22 = g.loc[g.year == 2022, "inflow"].iloc[0]
headline["intl_note"] = "rates-file totals (all migration events, region level)"
headline["global_inflow_2019"] = int(i19); headline["global_inflow_2020"] = int(i20)
headline["global_inflow_2021"] = int(i21)
headline["pct_change_2019_2020"] = round(100 * (i20 - i19) / i19, 1)
headline["pct_change_2019_2021"] = round(100 * (i21 - i19) / i19, 1)

fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.axvspan(*SHADE, color="#f0c94833", zorder=0)
ax.plot(g.year, g.inflow / 1e3, color=C_IN,  lw=1.4, label="In-migrations")
ax.plot(g.year, g.outflow / 1e3, color=C_OUT, lw=1.4, ls="--", label="Out-migrations")
ax.set_xlabel("Year"); ax.set_ylabel("Migration events (thousands)")
ax.legend(frameon=False, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); savefig(fig, "fig1_global_flows")

# ================= 3. FIG4: international vs internal (flows file) =================
fy = fl.groupby(["year", "is_international"])["n_migrations"].sum().unstack()
fy.columns = ["internal", "international"]
fy = fy[fy.index >= 1998]
fy.to_csv(os.path.join(RES, "fig4_intl_internal.csv"))

f19i = fy.loc[2019, "international"]; f20i = fy.loc[2020, "international"]
f19n = fy.loc[2019, "internal"];       f20n = fy.loc[2020, "internal"]
headline["intl_flows_2019"] = int(f19i); headline["intl_flows_2020"] = int(f20i)
headline["intl_pct_drop_2020"] = round(100 * (f20i - f19i) / f19i, 1)
headline["internal_pct_change_2020"] = round(100 * (f20n - f19n) / f19n, 1)
headline["intl_flows_2022_over_2019"] = round(fy.loc[2022, "international"] / f19i, 2)

fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.axvspan(*SHADE, color="#f0c94833", zorder=0)
ax.plot(fy.index, fy.international / 1e3, color=C_INTL, lw=1.4, label="International (cross-country)")
ax.plot(fy.index, fy.internal / 1e3, color=C_INT, lw=1.4, ls="--", label="Internal (within-country)")
ax.set_xlabel("Year"); ax.set_ylabel("Region-to-region flows (thousands)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); savefig(fig, "fig4_intl_internal")

# ================= 4. FIG2: shock path + event-study DID by exposure =================
# NOTE 1: single common shock date -> event time == calendar year -> two-way FE not
#         identified. Design: (A) region-FE year path; (B) event-study DID with
#         Expose_i = pre-2019 international inflow share. Cluster SE by region.
# NOTE 2: SMD 2.0 population denominators are zero for 2022+ -> NMR undefined for
#         recovery years. Outcomes therefore use log migration COUNTS (available
#         throughout); recovery is identified in count terms.
panel = rt[["year", "area", "iso3code", "padded_population_of_researchers",
            "number_of_inmigrations", "number_of_outmigrations", "netmigrationrate"]].copy()
pre_pop = (panel[(panel.year >= 2015) & (panel.year <= 2019)]
           .groupby("area")["padded_population_of_researchers"].mean())
keep = pre_pop[pre_pop >= 20].index
panel = panel[panel.area.isin(keep)].copy()
panel["y_in"] = np.log1p(panel.number_of_inmigrations)
panel["y_out"] = np.log1p(panel.number_of_outmigrations)

fl_pre = fl[(fl.year >= 2015) & (fl.year <= 2019)]
intl_in = fl_pre[fl_pre.is_international].groupby("area_to").n_migrations.sum()
all_in = fl_pre.groupby("area_to").n_migrations.sum()
exposure = (intl_in / all_in)
panel["share"] = panel.area.map(exposure)
YRS = list(range(2015, 2025))
panel = panel[panel.year.isin(YRS) & panel.share.notna()].copy()
# balanced panel: region observed in all 10 years (required for year dummies under FE)
cnt = panel.groupby("area").year.count()
bal = cnt[cnt == len(YRS)].index
panel = panel[panel.area.isin(bal)].copy()
sd_share = panel.groupby("area").share.first().std()
panel["sh"] = panel.share / sd_share  # scaled exposure
REF = 2019

def reg_cluster(df, X, y):
    """OLS with region FE (within transformation), cluster-robust SE by region."""
    reg = df.area.values
    def dem(a):
        a = np.asarray(a, float); one_d = (a.ndim == 1)
        if one_d: a = a[:, None]
        _, rinv = np.unique(reg, return_inverse=True)
        G = int(rinv.max()) + 1
        sums = np.zeros((G, a.shape[1])); np.add.at(sums, rinv, a)
        cnts = np.bincount(rinv, minlength=G).astype(float)
        a = a - sums[rinv] / cnts[rinv, None]
        return (a.ravel() if one_d else a), rinv, G
    Xd, rinv, G = dem(X)
    yd, _, _ = dem(y)
    XtX = Xd.T @ Xd
    beta = np.linalg.lstsq(XtX, Xd.T @ yd, rcond=None)[0]
    e = yd - Xd @ beta
    order = np.argsort(rinv)
    rs = rinv[order]; Xs = Xd[order]; es = e[order]
    K = Xd.shape[1]; meat = np.zeros((K, K)); start = 0
    for g in range(G):
        end = start + int((rs == g).sum())
        xg = Xs[start:end]; eg = es[start:end]
        sg = xg.T @ eg            # cluster score vector
        meat += np.outer(sg, sg)  # CR0 meat: sum_g s_g s_g'
        start = end
    N = Xd.shape[0]
    c = G / (G - 1) * (N - 1) / max(N - K, 1)
    V = c * np.linalg.pinv(XtX) @ meat @ np.linalg.pinv(XtX)
    return beta, np.sqrt(np.abs(np.diag(V)))

YD = [str(y) for y in YRS if y != REF]  # ref = 2019
# --- Model A: shock path (inflows) ---
D = pd.get_dummies(panel.year, prefix="Y", prefix_sep="").reindex(columns=["Y" + b for b in YD], fill_value=0).astype(float)
bA, sA = reg_cluster(panel, D.values, panel.y_in.values)
es_all = pd.DataFrame({"bin": YD, "beta": bA, "se": sA})
es_all.to_csv(os.path.join(RES, "fig2_event_study_all.csv"), index=False)
bA2, sA2 = reg_cluster(panel, D.values, panel.y_out.values)
es_out = pd.DataFrame({"bin": YD, "beta": bA2, "se": sA2})
es_out.to_csv(os.path.join(RES, "fig2_event_study_outflows.csv"), index=False)

# --- Model B: DID by exposure (inflows) ---
Di = (panel.sh.values[:, None] * D.values)  # share x year dummies (ref 2019)
Xb = np.hstack([D.values, Di])
bb, sb = reg_cluster(panel, Xb, panel.y_in.values)
kB = len(YD)
did = pd.DataFrame({"bin": YD, "beta": bb[kB:], "se": sb[kB:]})
did.to_csv(os.path.join(RES, "fig2_did_exposure.csv"), index=False)
headline["es_in_2020"] = round(float(es_all.loc[es_all.bin == "2020", "beta"].iloc[0]), 3)
headline["es_in_2021"] = round(float(es_all.loc[es_all.bin == "2021", "beta"].iloc[0]), 3)
headline["es_in_2024"] = round(float(es_all.loc[es_all.bin == "2024", "beta"].iloc[0]), 3)
headline["es_out_2020"] = round(float(es_out.loc[es_out.bin == "2020", "beta"].iloc[0]), 3)
headline["es_out_2021"] = round(float(es_out.loc[es_out.bin == "2021", "beta"].iloc[0]), 3)
headline["did_2020_per_sd"] = round(float(did.loc[did.bin == "2020", "beta"].iloc[0]), 3)
headline["did_2021_per_sd"] = round(float(did.loc[did.bin == "2021", "beta"].iloc[0]), 3)
headline["did_2024_per_sd"] = round(float(did.loc[did.bin == "2024", "beta"].iloc[0]), 3)
headline["exposure_sd"] = round(float(sd_share), 3)
headline["n_regions_es"] = int(panel.area.nunique())

xnum = np.arange(len(YD))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axes[0]
ax.axhline(0, color="#888888", lw=0.7)
ax.errorbar(xnum, es_all.beta, yerr=1.96 * es_all.se, fmt="o-", ms=3.5,
            lw=1.2, color="#1a1a1a", capsize=2, label="In-migrations")
ax.errorbar(xnum, es_out.beta, yerr=1.96 * es_out.se, fmt="s--", ms=3,
            lw=1.1, color="#7d3c98", capsize=2, label="Out-migrations")
ax.set_xticks(xnum); ax.set_xticklabels(YD, rotation=45)
ax.set_ylabel("log(count) deviation from 2019")
ax.set_xlabel("Year (2019 = reference)")
ax.legend(frameon=False, fontsize=7)
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
ax.axhline(0, color="#888888", lw=0.7)
ax.errorbar(xnum, did.beta, yerr=1.96 * did.se, fmt="s-", ms=3.5,
            lw=1.2, color=C_INTL, capsize=2)
ax.set_xticks(xnum); ax.set_xticklabels(YD, rotation=45)
ax.set_ylabel("DID coefficient (per 1 SD exposure)")
ax.set_xlabel("Year (2019 = reference)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); savefig(fig, "fig2_event_study")

# ================= 5. FIG3: trajectory classification =================
p = rt[rt.area.isin(keep)].copy()
def wmean(df, col, y0, y1):
    s = df[(df.year >= y0) & (df.year <= y1)].groupby("area")[col].mean()
    return s
pre  = wmean(p, "netmigrationrate", 2015, 2019)
# fixed-denominator net rate (population denominators vanish 2022+ in SMD 2.0):
# netrate* = netmigration / (mean population 2015-2019) * 1000
poppre = pre_pop
p["netrate2"] = p.netmigration / p.area.map(poppre) * 1000
shk  = wmean(p, "netrate2", 2020, 2021)
pre  = wmean(p, "netrate2", 2015, 2019)
post = wmean(p, "netrate2", 2022, 2024)
t = pd.concat([pre, shk, post], axis=1, keys=["pre", "shock", "post"]).dropna()
t = t[t.index.map(lambda a: pre_pop.get(a, 0) >= 20)]

def classify(r):
    if r.shock < r.pre and r.post >= r.pre:  return "Temporary shock, rebounded"
    if r.shock < r.pre and r.post < r.pre:   return "Persistent loser"
    if r.shock >= r.pre and r.post >= r.pre: return "Persistent gainer"
    return "Boom-bust"
t["class"] = t.apply(classify, axis=1)
t.to_csv(os.path.join(RES, "trajectories_region_level.csv"))
cnt = t["class"].value_counts()
cnt.to_csv(os.path.join(RES, "fig3_trajectory_counts.csv"))
headline["n_regions_classified"] = int(len(t))
headline["traj_counts"] = cnt.to_dict()

# mean trajectories per class (2015-2024), fixed-denominator rate
mt = (p[p.year.between(2015, 2024)][["year", "area", "netrate2"]]
      .merge(t[["class"]], left_on="area", right_index=True)
      .groupby(["class", "year"]).netrate2.mean().reset_index())

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axes[0]
order = ["Persistent gainer", "Temporary shock, rebounded", "Boom-bust", "Persistent loser"]
cols = {"Persistent gainer": "#27ae60", "Temporary shock, rebounded": "#2471a3",
        "Boom-bust": "#e67e22", "Persistent loser": "#c0392b"}
ax.barh(range(len(order)), [cnt.get(o, 0) for o in order], color=[cols[o] for o in order])
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
ax.set_xlabel("Number of subnational regions")
ax.invert_yaxis(); ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
for o in order:
    d = mt[mt["class"] == o]
    ax.plot(d.year, d.netrate2, lw=1.4, color=cols[o], label=o)
ax.axhline(0, color="#888888", lw=0.7)
ax.axvspan(*SHADE, color="#f0c94833", zorder=0)
ax.set_xlabel("Year"); ax.set_ylabel("Mean net migration rate*")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); savefig(fig, "fig3_trajectories")

# ================= 6. FIG5: China spotlight =================
cn_fl = fl[(fl.pref_from == "CN") | (fl.pref_to == "CN")]
cn_in  = cn_fl[cn_fl.pref_to == "CN"].groupby(["year", "is_international"]).n_migrations.sum().unstack()
cn_in.columns = ["internal_in", "international_in"]
cn_out = cn_fl[cn_fl.pref_from == "CN"].groupby(["year", "is_international"]).n_migrations.sum().unstack()
cn_out.columns = ["internal_out", "international_out"]
cn = cn_in.join(cn_out).loc[2015:2025]
cn.to_csv(os.path.join(RES, "fig5_china.csv"))
ci19 = cn.loc[2019, "international_in"]; ci20 = cn.loc[2020, "international_in"]
headline["china_intl_in_2019"] = int(ci19)
headline["china_intl_in_2020"] = int(ci20)
headline["china_intl_in_pct_change"] = round(100 * (ci20 - ci19) / ci19, 1)
headline["china_internal_in_pct_change_2020"] = round(100 * (cn.loc[2020, "internal_in"] - cn.loc[2019, "internal_in"]) / cn.loc[2019, "internal_in"], 1)

fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.axvspan(*SHADE, color="#f0c94833", zorder=0)
ax.plot(cn.index, cn.international_in / 1e3, color=C_INTL, lw=1.4, label="International inflows")
ax.plot(cn.index, cn.internal_in / 1e3, color=C_INT, lw=1.4, ls="--", label="Internal (inter-provincial) inflows")
ax.set_xlabel("Year"); ax.set_ylabel("Inflows to China regions (thousands)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); savefig(fig, "fig5_china")

# China province-level pre/post change
cnp = rt[rt.iso3code == "CHN"]
cpre = wmean(cnp, "netmigrationrate", 2015, 2019)
cpost = wmean(cnp, "netmigrationrate", 2022, 2024)
cch = pd.concat([cpre, cpost], axis=1, keys=["pre_NMR_2015_2019", "post_NMR_2022_2024"])
cch["change"] = cch.post_NMR_2022_2024 - cch.pre_NMR_2015_2019
cch = cch.sort_values("change")
cch.to_csv(os.path.join(RES, "china_province_change.csv"))

# ================= 7. substitution check =================
sub = (fl[(fl.year.isin([2019, 2020]))]
       .groupby(["area_to", "year", "is_international"]).n_migrations.sum().unstack()
       .fillna(0))
sub.columns = ["internal", "international"]
wid = sub.unstack("year")
d_intl = (wid[("international", 2020)] - wid[("international", 2019)])
d_int  = (wid[("internal", 2020)] - wid[("internal", 2019)])
valid = d_intl.index.map(lambda a: pre_pop.get(a, 0) >= 20)
r_sub = np.corrcoef(d_intl[valid.values], d_int[valid.values])[0, 1]
headline["corr_delta_intl_vs_delta_internal_2020"] = round(float(r_sub), 3)

with open(os.path.join(RES, "headline.json"), "w", encoding="utf-8") as f:
    json.dump(headline, f, ensure_ascii=False, indent=2)
print(json.dumps(headline, ensure_ascii=False, indent=2))
print("DONE")
