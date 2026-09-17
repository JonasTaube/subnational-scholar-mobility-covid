# -*- coding: utf-8 -*-
"""Verify sample sizes, 2022 totals, DID significance, and 1714 vs 1711 gap."""
import json, os
import numpy as np
import pandas as pd

# Set SMD2_BASE to your working copy, or keep this checkout's layout
import os
BASE = os.environ.get("SMD2_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "01_原始数据", "smd2_repo", "data_input", "2026_V2")
RES  = os.path.join(BASE, "03_分析与图表", "results")

rt = pd.read_parquet(os.path.join(DATA, "openalex_scholarlymigration_subnational_gender_and_field.parquet"))
rt_all = rt[(rt.gender == "all") & (rt.field == "all")]
fl = pd.read_parquet(os.path.join(DATA, "openalex_scholarlymigration_subnational_flows_gender_and_field.parquet"))
fl_all = fl[(fl.gender == "all") & (fl.field == "all")]

out = {}
out["rt_unique_areas_allall"] = int(rt_all.area.nunique())
out["rt_year_min_max"] = [int(rt_all.year.min()), int(rt_all.year.max())]
out["rt_year_min_max_anygender"] = [int(rt.year.min()), int(rt.year.max())]
out["rt_unique_areas_any"] = int(rt.area.nunique())
out["fl_year_min_max"] = [int(fl_all.year.min()), int(fl_all.year.max())]
u_from = fl_all.area_from.nunique(); u_to = fl_all.area_to.nunique()
out["fl_unique_area_from"] = int(u_from); out["fl_unique_area_to"] = int(u_to)

# fig1 2022 total
g = pd.read_csv(os.path.join(RES, "fig1_global_flows.csv"))
out["total_in_2022"] = int(g.loc[g.year == 2022, "inflow"].iloc[0])
out["total_out_2022"] = int(g.loc[g.year == 2022, "outflow"].iloc[0])
fy = pd.read_csv(os.path.join(RES, "fig4_intl_internal.csv"), index_col=0)
out["sum_check_2019"] = {"intl": int(fy.loc[2019, "international"]), "internal": int(fy.loc[2019, "internal"]),
                         "sum": int(fy.loc[2019, "international"] + fy.loc[2019, "internal"])}
out["sum_check_2022"] = {"intl": int(fy.loc[2022, "international"]), "internal": int(fy.loc[2022, "internal"]),
                         "sum": int(fy.loc[2022, "international"] + fy.loc[2022, "internal"])}

# reconstruct samples
rt_all = rt_all.copy()
pre_pop = (rt_all[(rt_all.year >= 2015) & (rt_all.year <= 2019)]
           .groupby("area")["padded_population_of_researchers"].mean())
keep = pre_pop[pre_pop >= 20].index
out["n_keep_pop20"] = int(len(keep))

fl_all = fl_all.copy()
fl_all["pref_from"] = fl_all["area_from"].str.split(".").str[0]
fl_all["pref_to"] = fl_all["area_to"].str.split(".").str[0]
fl_all["is_intl"] = fl_all["pref_from"] != fl_all["pref_to"]
fl_pre = fl_all[(fl_all.year >= 2015) & (fl_all.year <= 2019)]
intl_in = fl_pre[fl_pre.is_intl].groupby("area_to").n_migrations.sum()
all_in = fl_pre.groupby("area_to").n_migrations.sum()
exposure = intl_in / all_in

panel = rt_all[["year", "area", "number_of_inmigrations"]].copy()
panel["share"] = panel.area.map(exposure)
YRS = list(range(2015, 2025))
panel = panel[panel.year.isin(YRS) & panel.share.notna()]
cnt = panel.groupby("area").year.count()
bal = cnt[cnt == len(YRS)].index
out["n_balanced_regression"] = int(len(bal))

# trajectory sample (replicate analysis_core.py)
p = rt_all[rt_all.area.isin(keep)].copy()
p["netrate2"] = p.netmigration / p.area.map(pre_pop) * 1000
def wmean(df, col, y0, y1):
    return df[(df.year >= y0) & (df.year <= y1)].groupby("area")[col].mean()
pre = wmean(p, "netrate2", 2015, 2019)
shk = wmean(p, "netrate2", 2020, 2021)
post = wmean(p, "netrate2", 2022, 2024)
t = pd.concat([pre, shk, post], axis=1, keys=["pre", "shock", "post"]).dropna()
t = t[t.index.map(lambda a: pre_pop.get(a, 0) >= 20)]
out["n_trajectory"] = int(len(t))
extra = sorted(set(t.index) - set(bal))
out["traj_not_in_reg"] = extra
for a in extra:
    yrs_obs = sorted(panel[panel.area == a].year.tolist()) if a in set(panel.area) else []
    miss_years = [y for y in YRS if y not in yrs_obs]
    out.setdefault("extra_detail", {})[a] = {
        "share_missing": bool(pd.isna(exposure.get(a, np.nan))),
        "years_missing_in_panel": miss_years,
    }

# DID significance
did = pd.read_csv(os.path.join(RES, "fig2_did_exposure.csv"))
did["t"] = did.beta / did.se
did["ci_lo"] = did.beta - 1.96 * did.se
did["ci_hi"] = did.beta + 1.96 * did.se
out["did_table"] = did.round(4).to_dict("records")
es_in = pd.read_csv(os.path.join(RES, "fig2_event_study_all.csv"))
es_in["t"] = es_in.beta / es_in.se
es_out = pd.read_csv(os.path.join(RES, "fig2_event_study_outflows.csv"))
es_out["t"] = es_out.beta / es_out.se
out["es_in_table"] = es_in.round(4).to_dict("records")
out["es_out_table"] = es_out.round(4).to_dict("records")

print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
