# -*- coding: utf-8 -*-
"""Full cross-matrix of the two trajectory sensitivities: denominator x post window."""
import json, os
import numpy as np
import pandas as pd

# Set SMD2_BASE to your working copy, or keep this checkout's layout
import os
BASE = os.environ.get("SMD2_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "01_原始数据", "smd2_repo", "data_input", "2026_V2")
RES = os.path.join(BASE, "03_分析与图表", "results")
out = {}

rt = pd.read_parquet(os.path.join(DATA, "openalex_scholarlymigration_subnational_gender_and_field.parquet"))
fl = pd.read_parquet(os.path.join(DATA, "openalex_scholarlymigration_subnational_flows_gender_and_field.parquet"))
rta = rt[(rt.gender == "all") & (rt.field == "all")].copy()
pre_pop = (rta[(rta.year >= 2015) & (rta.year <= 2019)]
           .groupby("area")["padded_population_of_researchers"].mean())
keep = pre_pop[pre_pop >= 20].index
p = rta[rta.area.isin(keep)].copy()
p["nr"] = p.netmigration / p.area.map(pre_pop) * 1000
p19 = p[p.year == 2019][["area", "padded_population_of_researchers"]].set_index("area")[
    "padded_population_of_researchers"]
p["nr19"] = p.netmigration / p.area.map(p19) * 1000

wm = lambda col, y0, y1: p[(p.year >= y0) & (p.year <= y1)].groupby("area")[col].mean()


def cl(r):
    if r.shock < r.pre and r.post >= r.pre:  return "temp_rebound"
    if r.shock < r.pre and r.post < r.pre:   return "persistent_decline"
    if r.shock >= r.pre and r.post >= r.pre: return "persistent_improvement"
    return "reversal"


def run(col, post_years):
    pre, shk = wm(col, 2015, 2019), wm(col, 2020, 2021)
    post = (wm(col, post_years[0], post_years[1]) if len(post_years) == 2
            else wm(col, post_years[0], post_years[0]))
    t = pd.concat([pre, shk, post], axis=1, keys=["pre", "shock", "post"]).dropna()
    t["class"] = t.apply(cl, axis=1)
    return t


COMBOS = {
    "denom_pre_window2022": ("nr", (2022,)),
    "denom_pre_window2022_2024": ("nr", (2022, 2024)),
    "denom_2019_window2022": ("nr19", (2022,)),
    "denom_2019_window2022_2024": ("nr19", (2022, 2024)),
}
frames = {}
for k, (col, pw) in COMBOS.items():
    t = run(col, pw); frames[k] = t
    vc = t["class"].value_counts().to_dict()
    out[k] = {"n": int(len(t)), "counts": {a: int(b) for a, b in vc.items()},
              "shares": {a: round(b / len(t), 3) for a, b in vc.items()}}
    print(k, out[k]["n"], vc)


def agree(a, b):
    idx = frames[a].index.intersection(frames[b].index)
    return (frames[a].loc[idx, "class"] == frames[b].loc[idx, "class"]).mean(), len(idx)


for a, b in [("denom_pre_window2022", "denom_2019_window2022"),
             ("denom_pre_window2022", "denom_pre_window2022_2024"),
             ("denom_pre_window2022_2024", "denom_2019_window2022_2024")]:
    r, n = agree(a, b)
    out[f"agreement {a} vs {b}"] = [round(float(r), 4), int(n)]
    print(f"agreement {a} vs {b}: {r:.4f} (n={n})")

# Scopus under the primary window, for comparability with the new primary
rts = pd.read_parquet(os.path.join(DATA, "scopus_scholarlymigration_subnational_gender_and_field.parquet"))
rtas = rts[(rts.gender == "all") & (rts.field == "all")].copy()
pp_s = (rtas[(rtas.year >= 2015) & (rtas.year <= 2019)]
        .groupby("area")["padded_population_of_researchers"].mean())
ks = pp_s[pp_s >= 20].index
ps = rtas[rtas.area.isin(ks)].copy()
ps["nr"] = ps.netmigration / ps.area.map(pp_s) * 1000
g = lambda y0, y1: ps[(ps.year >= y0) & (ps.year <= y1)].groupby("area").nr.mean()
ts = pd.concat([g(2015, 2019), g(2020, 2021), g(2022, 2022)],
               axis=1, keys=["pre", "shock", "post"]).dropna()
ts["class"] = ts.apply(cl, axis=1)
vc = ts["class"].value_counts().to_dict()
out["scopus_window2022"] = {"n": int(len(ts)), "counts": {a: int(b) for a, b in vc.items()}}
print("scopus primary window:", len(ts), vc)

with open(os.path.join(RES, "robustness_round4.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("DONE")
