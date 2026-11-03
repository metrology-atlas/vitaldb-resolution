"""Tabular screens of the VitalDB clinical and lab tables, one subcommand per metrology-atlas study.

Each subcommand reads clinical_data.csv and/or lab_data.csv from the release and writes OUT/<study>.json with the
numbers the study quotes (and prints them). They run in seconds.

    python tools/screens.py <study> --root <VitalDB release folder> --out <folder>
    python tools/screens.py all --root ... --out ...

Times in clinical_data.csv are seconds from the start of the recording (casestart = 0); admission and discharge
(adm, dis) are midnights of their dates on the same scale, so -adm modulo one day is the recording's start time of
day and the date arithmetic below follows from it. Case 1563's admission and discharge, and case 4476's
anaesthesia end, are 1900 placeholder dates (near -3.69e9 s) and are left out where stated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86400.0
PLACEHOLDER_S = -3.0e9  # dates defaulting to 1900 sit near -3.69e9 s


def q(v, ps=(5, 25, 50, 75, 95)) -> dict | None:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    return {f"p{p}": round(float(np.percentile(v, p)), 4) for p in ps} | {"n": int(v.size)}


def clinical(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "clinical_data.csv", encoding="utf-8-sig")


def clinical_str(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "clinical_data.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)


def lab(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "lab_data.csv")


def share(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


# ------------------------------------------------------------------------------------------------ studies

def dates_admission_discharge_los(root: Path) -> dict:
    """Admission and discharge dates: whole days, admission lead time, length of stay, discharge day."""
    c = clinical(root)
    ok = (c["adm"] > PLACEHOLDER_S) & (c["dis"] > PLACEHOLDER_S)
    d = c[ok]
    los_s = d["dis"] - d["adm"]
    tod = np.mod(-d["adm"], DAY)                      # recording start, seconds after midnight
    lead = np.floor(-d["adm"] / DAY).astype(int)       # days from admission date to surgery date
    los = (los_s / DAY)
    postop = np.round((d["dis"] + tod) / DAY).astype(int)   # discharge date minus the recording's date, days
    lc = lead.value_counts()
    pc = postop.value_counts()
    return {
        "cases": int(len(c)), "excluded_placeholder": [int(x) for x in c.loc[~ok, "caseid"]],
        "los_whole_days": int((np.mod(los_s, DAY) == 0).sum()), "of": int(len(d)),
        "admitted_days_before": {str(k): int(lc.get(k, 0)) for k in range(0, 5)},
        "admitted_share": {str(k): share(int(lc.get(k, 0)), len(d)) for k in range(0, 3)},
        "los_days": q(los),
        "discharged_day_of_surgery": int((postop == 0).sum()),
        "discharge_postop_day_counts": {str(k): int(pc.get(k, 0)) for k in range(0, 15)},
        "los_over_115_days": int((los > 115).sum()), "los_max_days": float(los.max()),
        "postop_over_115_days": int((postop > 115).sum()), "postop_max_days": int(postop.max()),
    }


def lab_preop_pao2_decimal(root: Path) -> dict:
    """Pre-operative PaO2: share recorded as a whole number (no decimal, or .0)."""
    s = clinical_str(root)["preop_pao2"]
    s = s[s != ""]
    whole = s.map(lambda x: "." not in x or x.endswith(".0"))
    return {"values": int(len(s)), "whole_number": int(whole.sum()), "share_whole": share(int(whole.sum()), len(s)),
            "expected_at_0_1": 0.10,
            "estimated_share_recorded_whole": round((whole.mean() - 0.10) / 0.90, 4)}


def timestamp_clock_reconstruction(root: Path) -> dict:
    """-adm as whole minutes; recording start time of day; charted events on 5-minute clock marks.

    -adm is a whole number of minutes, so it gives the recording's start rounded down to the minute; the seconds
    (0-59) are not known. A charted event lies on a 5-minute clock mark if some start second makes it so: its time
    of day from the rounded start, modulo 300 s, is above 240 s or exactly 0."""
    c = clinical(root)
    all_neg = (-c["adm"]).to_numpy()
    all_whole = np.isclose(np.mod(all_neg, 60), 0) | np.isclose(np.mod(all_neg, 60), 60)
    ok = c["adm"] > PLACEHOLDER_S
    neg = (-c.loc[ok, "adm"]).to_numpy()
    whole_min = np.isclose(np.mod(neg, 60), 0) | np.isclose(np.mod(neg, 60), 60)
    tod = np.mod(neg, DAY)
    tod_w = tod[whole_min]
    hours = pd.Series((tod_w // 3600).astype(int)).value_counts()
    out = {"cases": int(len(c)), "adm_whole_minutes_all_cases": int(all_whole.sum()),
           "not_whole_minutes": [int(x) for x in c["caseid"].to_numpy()[~all_whole]],
           "with_real_adm": int(ok.sum()), "start_times_used": int(whole_min.sum()),
           "start_hour_counts": {f"{h:02d}": int(hours.get(h, 0)) for h in range(24)},
           "start_tod_h": q(tod_w / 3600.0),
           "start_tod_hhmm": {k: f"{int(v // 1):02d}:{int(round(v % 1 * 60)):02d}" for k, v in q(tod_w / 3600.0).items() if k != "n"}}
    for ev in ("anestart", "opstart", "opend", "aneend"):
        t = c.loc[ok, ev].to_numpy()
        good = whole_min & np.isfinite(t) & (t > PLACEHOLDER_S)
        clock = np.mod(tod[good] + t[good], 300)
        on = (clock > 240) | np.isclose(clock, 0, atol=1e-6)
        out[f"{ev}_on_5_min"] = {"n": int(good.sum()), "on": int(on.sum()), "share": share(int(on.sum()), int(good.sum()))}
    return out


def timestamp_imputed_and_conformance(root: Path) -> dict:
    """Imputed op ends, anaesthesia ends with floating-point noise, placeholder dates, event order, outliers."""
    c = clinical(root)
    gap = c["caseend"] - c["opend"]
    imp = (gap >= 598) & (gap <= 604)
    gc = gap[imp].round().astype(int).value_counts()
    ae = c["aneend"]
    noisy = clinical_str(root)["aneend"].str.contains(".", regex=False) & (ae > PLACEHOLDER_S)   # e.g. 5117.999999
    d_ae = (ae - c["opend"])[noisy].round()
    mult = d_ae[np.isclose(np.mod(d_ae, 300), 0)]
    placeholder = {col: [int(x) for x in c.loc[c[col] < PLACEHOLDER_S, "caseid"]]
                   for col in ("casestart", "caseend", "anestart", "aneend", "opstart", "opend", "adm", "dis")}
    real = ae > PLACEHOLDER_S
    out = {
        "cases": int(len(c)),
        "opend_imputed": int(imp.sum()), "opend_imputed_share": share(int(imp.sum()), len(c)),
        "imputed_gap_counts_s": {str(k): int(v) for k, v in sorted(gc.items())},
        "aneend_float_noise": int(noisy.sum()),
        "aneend_noise_minus_opend_s_counts": {str(int(k)): int(v) for k, v in mult.value_counts().head(8).items()},
        "aneend_noise_with_imputed_opend": int((noisy & imp).sum()),
        "placeholder_dates": placeholder,
        "opend_after_aneend": int(((c["opend"] > ae) & real).sum()),
        "opend_equals_aneend": int(((c["opend"] == ae) & real).sum()),
        "opstart_before_anestart": int((c["opstart"] < c["anestart"]).sum()),
        "opend_before_opstart": int((c["opend"] < c["opstart"]).sum()),
        "aneend_before_anestart": int(((ae < c["anestart"]) & real).sum()),
    }
    late = ((ae - c["caseend"]) / 60.0)
    big = c.loc[(late > 60) & real, ["caseid"]].assign(min_after_end=late[(late > 60) & real].round(1))
    out["aneend_more_than_60_min_after_recording_end"] = big.to_dict("records")
    for col in ("department", "ane_type", "optype", "emop"):
        g = c.groupby(col).apply(lambda x: (int(((x["caseend"] - x["opend"]).between(598, 604)).sum()), int(len(x))))
        out[f"imputed_by_{col}"] = {str(k): {"imputed": v[0], "of": v[1], "share": share(v[0], v[1])} for k, v in g.items()}
    return out


def value_bolus_drug_doses(root: Path) -> dict:
    """Bolus drug totals: the unit each drug's values sit on, common totals, rocuronium and propofol checks."""
    c = clinical(root)
    drugs = {"intraop_eph": (5,), "intraop_ca": (300, 150), "intraop_ftn": (50,), "intraop_phe": (10,),
             "intraop_rocu": (10,), "intraop_ppf": (10, 50), "intraop_mdz": (1,), "intraop_vecu": (1,),
             "intraop_epi": (1,)}
    out = {}
    for col, units in drugs.items():
        v = c[col].dropna()
        nz = v[v > 0]
        out[col] = {"nonzero": int(len(nz)), "top_values": {str(k): int(n) for k, n in nz.value_counts().head(6).items()},
                    "on_unit": {str(u): share(int((np.isclose(np.mod(nz, u), 0)).sum()), len(nz)) for u in units}}
    dur_h = (c["caseend"] - c["casestart"]) / 3600.0     # case (recording) duration
    good = (c["intraop_rocu"] > 0) & (dur_h > 0)
    out["rocuronium_spearman"] = {
        "duration": round(float(pd.Series(c.loc[good, "intraop_rocu"]).corr(dur_h[good], method="spearman")), 3),
        "weight": round(float(c.loc[good, "intraop_rocu"].corr(c.loc[good, "weight"], method="spearman")), 3),
        "n": int(good.sum())}
    by_ane = c.groupby("ane_type").apply(lambda x: (int((x["intraop_rocu"] == 0).sum()), int(len(x))))
    out["rocuronium_zero_by_ane_type"] = {str(k): {"zero": v[0], "of": v[1]} for k, v in by_ane.items()}
    gen = c["ane_type"] == "General"
    out["propofol_zero_general"] = {"zero": int((c.loc[gen, "intraop_ppf"] == 0).sum()), "of": int(gen.sum())}
    return out


def value_intraop_colloid_containers(root: Path) -> dict:
    """Intra-operative colloid volumes by operation group, against container sizes."""
    c = clinical(root)
    v = c["intraop_colloid"]
    nz = c[v > 0]
    real = c["aneend"] > PLACEHOLDER_S
    dur_h = ((c["aneend"] - c["anestart"]) / 3600.0).where(real)      # anaesthesia duration
    name = c["opname"].fillna("")
    liver = name.str.lower().str.contains("liver transplant")
    kidney = name.isin(["Kidney transplantation", "Donor nephrectomy", "Nephrectomy"])
    grp = np.where(liver, "liver transplant", np.where(kidney, "kidney surgery", "other"))
    out = {"values": int(v.notna().sum()), "zeros": int((v == 0).sum()), "nonzero": int((v > 0).sum()),
           "max_ml": float(v.max()),
           "median_anaesthesia_h": {"colloid": round(float(dur_h[v > 0].median()), 2), "none": round(float(dur_h[v == 0].median()), 2)},
           "median_ebl_ml": {"colloid": float(c.loc[v > 0, "intraop_ebl"].median()), "none": float(c.loc[v == 0, "intraop_ebl"].median())},
           "groups": {}}
    for g in ("other", "liver transplant", "kidney surgery"):
        sel = (grp == g)
        x = v[sel & (v > 0)]
        out["groups"][g] = {"cases": int(sel.sum()), "nonzero": int(len(x)),
                            "counts": {str(int(k) if float(k).is_integer() else k): int(n) for k, n in x.value_counts().sort_index().items()}}
    kt = name == "Kidney transplantation"
    x = v[kt & (v > 0)]
    out["kidney_transplant"] = {"nonzero": int(len(x)), "counts": {str(k): int(n) for k, n in x.value_counts().items()}}
    return out


def value_registry_unmodelled(root: Path) -> dict:
    """Red-cell and plasma units, age, ICU days."""
    c = clinical(root)
    s = clinical_str(root)
    rbc, ffp, ebl = c["intraop_rbc"], c["intraop_ffp"], c["intraop_ebl"]
    nz = rbc > 0
    flag = (rbc >= 50) & (ebl < 1000)
    age_num = pd.to_numeric(s["age"], errors="coerce")
    ages = age_num.dropna()
    whole = ages[np.isclose(ages, np.round(ages))]
    los_days = (c["dis"] - c["adm"]) / DAY
    ok = (c["adm"] > PLACEHOLDER_S)
    return {
        "rbc_nonzero": int(nz.sum()), "rbc_counts_1_5": int(rbc[nz].between(1, 5).sum()),
        "ffp_nonzero": int((ffp > 0).sum()), "ffp_range": [float(ffp[ffp > 0].min()), float(ffp[ffp > 0].max())],
        "rbc_vs_ebl_spearman": round(float(rbc[nz].corr(ebl[nz], method="spearman")), 3),
        "rbc_flagged": c.loc[flag, ["caseid", "intraop_rbc", "intraop_ebl"]].to_dict("records"),
        "rbc_large_not_flagged": c.loc[(rbc >= 50) & ~flag, ["caseid", "intraop_rbc", "intraop_ebl"]].to_dict("records"),
        "age_numbers": int(len(ages)), "age_coded": {str(k): int(n) for k, n in s.loc[age_num.isna(), "age"].value_counts().items()},
        "age_end_0_or_5": share(int(np.isin(np.mod(whole, 10), [0, 5]).sum()), int(len(whole))),
        "age_fractional": [float(x) for x in sorted(ages[~np.isclose(ages, np.round(ages))])],
        "icu_days": {"zero_share": share(int((c["icu_days"] == 0).sum()), int(c["icu_days"].notna().sum())),
                     "one_share": share(int((c["icu_days"] == 1).sum()), int(c["icu_days"].notna().sum())),
                     "max": float(c["icu_days"].max())},
        "icu_exceeds_hospital_stay": int((ok & (c["icu_days"] > los_days)).sum()),
    }


# ------------------------------------------------------------------------------------------------ lab helpers

def lab_sources(L: pd.DataFrame) -> pd.Series:
    """Source of each lab result from the panel sharing its case and timestamp: 'bg' where the panel has a pH
    (the theatre blood-gas analyser), 'cbc' where it has Hb, WBC or platelets (the central CBC analyser)."""
    names = L.groupby(["caseid", "dt"])["name"].agg(set)
    bg = names.map(lambda x: "ph" in x)
    cbc = names.map(lambda x: bool(x & {"hb", "wbc", "plt"}))
    key = pd.MultiIndex.from_frame(L[["caseid", "dt"]])
    return pd.Series(np.where(bg.reindex(key).to_numpy(), "bg", np.where(cbc.reindex(key).to_numpy(), "cbc", "other")),
                     index=L.index)


def decimals(v: pd.Series) -> pd.Series:
    """Decimal places as stored (results are written without trailing zeros)."""
    def nd(x):
        t = repr(float(x)).rstrip("0").rstrip(".")
        return len(t.split(".")[1]) if "." in t else 0
    return v.map(nd)


def finest(v: pd.Series) -> dict:
    """Finest resolution (the most decimal places held by at least 1% of values) and the share one step coarser."""
    d = decimals(v)
    vc = d.value_counts(normalize=True)
    f = int(max(k for k, s_ in vc.items() if s_ >= 0.01))
    return {"n": int(len(v)), "finest_decimals": f, "coarser_share": round(float((d < f).mean()), 4)}


def heap_ratio(v: pd.Series, base: int, odd_only: bool = False, w: int = 4) -> dict:
    """Preference for multiples of BASE among whole-number values: at each multiple m (only odd multiples of 5 when
    ODD_ONLY), the expected count from a quadratic fit to the counts at m-w..m+w, leaving out m and other multiples
    of 5. Ratio = sum observed / sum expected; z = (O - E) / sqrt(E)."""
    x = v[np.isclose(v, np.round(v))].round().astype(int)
    cnt = x.value_counts()
    lo, hi = int(x.min()), int(x.max())
    O = E = 0.0
    n = 0
    for m in range((lo // base + 1) * base, hi, base):
        if odd_only and m % 10 == 0:
            continue
        xs = [m + d for d in range(-w, w + 1) if d and (m + d) % 5]
        ys = [cnt.get(t, 0) for t in xs]
        e = max(float(np.polyval(np.polyfit(xs, ys, 2), m)), 0.5)
        O += cnt.get(m, 0)
        E += e
        n += 1
    return {"screened": int(len(x)), "multiples": n, "observed": int(O), "expected": round(E, 1),
            "ratio": round(O / E, 3), "z": round((O - E) / np.sqrt(E), 1)}


def lab_haematocrit_two_analysers(root: Path) -> dict:
    """Haematocrit by source; last digits; blood gas before CBC within 15 min; resolutions of shared analytes."""
    L = lab(root)
    L["src"] = lab_sources(L)
    h = L[L["name"] == "hct"]
    out = {"hct": int(len(h)), "by_source": {k: int(v) for k, v in h["src"].value_counts().items()}}
    cbc, bg = h[h["src"] == "cbc"], h[h["src"] == "bg"]
    tenths = np.round(cbc["result"] * 10).astype(int) % 10
    out["cbc_decimal_digit_share"] = {str(k): round(float(v), 4) for k, v in tenths.value_counts(normalize=True).sort_index().items()}
    out["bg_whole_percent"] = int(np.isclose(bg["result"], np.round(bg["result"])).sum())
    units = np.round(bg["result"]).astype(int) % 10
    out["bg_last_digit_share"] = {str(k): round(float(v), 4) for k, v in units.value_counts(normalize=True).sort_index().items()}
    out["bg_x5"] = heap_ratio(bg["result"], 5, odd_only=True)
    out["bg_x10"] = heap_ratio(bg["result"], 10)
    # a blood gas preceding a CBC in the same case by up to 15 min: each CBC with the latest blood gas before it
    pairs = []
    bgs = bg.sort_values(["caseid", "dt"])
    for cid, g in cbc.groupby("caseid"):
        b2 = bgs[bgs["caseid"] == cid]
        if b2.empty:
            continue
        t = b2["dt"].to_numpy()
        for dt, val in zip(g["dt"], g["result"]):
            j = np.searchsorted(t, dt, side="right") - 1
            if j >= 0 and dt - t[j] <= 900:
                pairs.append((float(b2["result"].iloc[j]), val))
    d = np.array([a - b for a, b in pairs])
    out["bg_then_cbc_15min"] = {"pairs": int(d.size), "mean_bg_minus_cbc": round(float(d.mean()), 2), "sd": round(float(d.std(ddof=1)), 2)}
    out["shared_analytes"] = {}
    for nm in ("gluc", "na", "k", "ica", "lac"):
        x = L[L["name"] == nm]
        out["shared_analytes"][nm] = {s_: finest(x.loc[x["src"] == s_, "result"]) for s_ in ("bg", "cbc", "other")
                                      if (x["src"] == s_).sum() >= 50}
    return out


def lab_computed_value_grids(root: Path) -> dict:
    """PT % and fibrinogen: heaping ratios, integers that cannot occur, and the pre-operative PT %."""
    L = lab(root)
    c = clinical(root)
    out = {}
    for nm in ("pt%", "fib"):
        v = L.loc[L["name"] == nm, "result"]
        x = v[np.isclose(v, np.round(v))].round().astype(int)
        cnt = x.value_counts()
        out[nm] = {"x10": heap_ratio(v, 10), "x5_odd": heap_ratio(v, 5, odd_only=True)}
        if nm == "pt%":
            out[nm]["counts_96_104"] = {str(k): int(cnt.get(k, 0)) for k in range(96, 105)}
        else:
            low = []
            for k in range(150, 451):
                nb = [cnt.get(k + d, 0) for d in range(-5, 6) if d]   # local average: the 10 integers within 5
                if np.mean(nb) > 0 and cnt.get(k, 0) < 0.3 * np.mean(nb):
                    low.append(k)
            out[nm]["integers_150_450_under_30pct_of_neighbours"] = {"n": len(low), "values": low}
    pre = c["preop_pt"].dropna()
    pc = pre.round().astype(int).value_counts()
    out["preop_pt"] = {"n": int(len(pre)), "x10": heap_ratio(pre, 10), "counts_96_104": {str(k): int(pc.get(k, 0)) for k in range(96, 105)}}
    return out


def heaping_screen_all_variables(root: Path) -> dict:
    """The quoted figures of the all-variable heaping screen: heaped clinical values, age, lab coarser shares,
    whole-number analytes, total bilirubin's low range, lactate and the reporting limits."""
    c = clinical(root)
    s = clinical_str(root)
    L = lab(root)
    out = {"clinical": {}}
    def endshare(col, mod, rem=(0,), positive=True):
        v = c[col].dropna()
        v = v[v > 0] if positive else v
        return {"n": int(len(v)), "share": round(float(np.isin(np.mod(v, mod), rem).mean()), 4)}
    out["clinical"]["urine_end_in_0"] = endshare("intraop_uo", 10)
    out["clinical"]["ebl_multiple_50"] = endshare("intraop_ebl", 50)
    out["clinical"]["crystalloid_multiple_50"] = endshare("intraop_crystalloid", 50)
    out["clinical"]["colloid_multiple_50"] = endshare("intraop_colloid", 50)
    ht = c["height"].dropna()
    out["clinical"]["height_whole_cm"] = {"n": int(len(ht)), "share": round(float(np.isclose(ht, np.round(ht)).mean()), 4), "expected": 0.10}
    wt = c["weight"].dropna()
    out["clinical"]["weight_whole_kg"] = {"n": int(len(wt)), "share": round(float(np.isclose(wt, np.round(wt)).mean()), 4)}
    age = pd.to_numeric(s["age"], errors="coerce").dropna()
    aw = age[np.isclose(age, np.round(age))].round().astype(int)
    out["clinical"]["age_end_0_or_5"] = round(float(np.isin(aw % 10, [0, 5]).mean()), 4)
    out["clinical"]["age_last_digit_share"] = {str(k): round(float(v), 4) for k, v in (aw % 10).value_counts(normalize=True).sort_index().items()}
    for col in ("intraop_rbc", "intraop_ffp", "icu_days"):
        v = c[col].dropna(); v = v[v > 0]
        out["clinical"][f"{col}_end_0_or_5"] = {"n": int(len(v)), "share": round(float(np.isin(v.round() % 10, [0, 5]).mean()), 4)}
    los = ((c["dis"] - c["adm"]) / DAY)[c["adm"] > PLACEHOLDER_S]
    out["clinical"]["los_end_0_or_5"] = round(float(np.isin(los.round() % 10, [0, 5]).mean()), 4)
    L["src"] = lab_sources(L)
    out["lab_coarser"] = {nm: finest(g["result"]) for nm, g in L.groupby("name")}
    out["whole_number_analytes"] = {}
    for nm, g in L.groupby("name"):
        if out["lab_coarser"][nm]["finest_decimals"] == 0 and len(g) > 500:
            out["whole_number_analytes"][nm] = {"x10": heap_ratio(g["result"], 10), "x5_odd": heap_ratio(g["result"], 5, odd_only=True)}
    tb = L.loc[(L["name"] == "tbil"), "result"]
    low = tb[(tb >= 0.2 - 1e-9) & (tb <= 1.0 + 1e-9)]
    lc = (tb[np.isclose(tb * 10, np.round(tb * 10))] * 10).round().astype(int).value_counts().sort_index()
    rel = {}
    for k in range(2, 11):
        nb = [lc.get(k + d, 0) for d in (-1, 1)]
        nb = [x for x in nb if x]
        rel[f"{k / 10:.1f}"] = {"count": int(lc.get(k, 0)), "vs_neighbour_mean": round(lc.get(k, 0) / np.mean(nb) - 1, 3) if nb else None}
    out["tbil_low_range"] = {"n": int(len(low)), "by_value": rel}
    la = L[L["name"] == "lac"]
    out["lactate"] = {"n": int(len(la)), "with_ph_panel": int((la["src"] == "bg").sum()),
                      "at_15": int(np.isclose(la["result"], 15).sum()), "at_15_cases": int(la.loc[np.isclose(la["result"], 15), "caseid"].nunique()),
                      "max_other_count_ge_10": int(la.loc[(la["result"] >= 10) & ~np.isclose(la["result"], 15), "result"].value_counts().max())}
    ap = L[L["name"] == "aptt"]
    es = L[L["name"] == "esr"]
    out["aptt_400"] = {"results": int(np.isclose(ap["result"], 400).sum()), "cases": int(ap.loc[np.isclose(ap["result"], 400), "caseid"].nunique())}
    out["esr_2"] = {"results": int(np.isclose(es["result"], 2).sum()), "cases": int(es.loc[np.isclose(es["result"], 2), "caseid"].nunique()),
                    "below_2": int((es["result"] < 2).sum())}
    return out


def timestamp_lab_times_meaning(root: Path) -> dict:
    """Lab panel timestamps: panels drawn together, order of draw, lag after a blood gas, time of day, minute."""
    L = lab(root)
    c = clinical(root)
    L["src"] = lab_sources(L)
    panels = L.groupby(["caseid", "dt"])["name"].agg(set).reset_index()
    def kind(x):
        if "ph" in x: return "bg"
        if x & {"hb", "wbc", "plt"}: return "cbc"
        if x & {"ptinr", "ptsec", "pt%", "aptt", "fib"}: return "coag"
        if x & {"na", "k", "cr", "bun", "alb", "ast", "alt", "tbil", "gluc"}: return "chem"
        return "other"
    panels["kind"] = panels["name"].map(kind)
    out = {"case_timestamp_pairs": int(len(panels)), "panels": {k: int(v) for k, v in panels["kind"].value_counts().items()}}
    # CBC and chemistry stamped close together: for each CBC the nearest chemistry panel in the same case
    d = []
    first = {"cbc": 0, "chem": 0, "same": 0}
    for cid, g in panels.groupby("caseid"):
        a = g.loc[g["kind"] == "cbc", "dt"].to_numpy()
        b = np.sort(g.loc[g["kind"] == "chem", "dt"].to_numpy())
        if not a.size or not b.size:
            continue
        for t in a:
            j = np.searchsorted(b, t)
            cand = [b[k] for k in (j - 1, j) if 0 <= k < b.size]
            nb = min(cand, key=lambda x: abs(x - t))
            if abs(nb - t) <= 600:
                d.append(nb - t)
                first["same" if nb == t else ("cbc" if t < nb else "chem")] += 1
    d = np.array(d, dtype=float)
    out["cbc_chem_pairs_within_10min"] = {"n": int(d.size), "within_60s": round(float((np.abs(d) <= 60).mean()), 3),
                                          "median_chem_minus_cbc_s": float(np.median(d)), "first": first}
    # lag from a blood gas to the next central-lab panel, intra-operative (to 30 min after anaesthesia end)
    ae = c.set_index("caseid")["aneend"]
    lag = {}
    for k in ("cbc", "coag", "chem"):
        v = []
        for cid, g in panels.groupby("caseid"):
            end = ae.get(cid, np.nan)
            if not np.isfinite(end) or end < PLACEHOLDER_S:
                continue
            bgt = np.sort(g.loc[g["kind"] == "bg", "dt"].to_numpy())
            for t in g.loc[(g["kind"] == k) & (g["dt"] >= 0) & (g["dt"] <= end + 1800), "dt"]:
                j = np.searchsorted(bgt, t, side="right") - 1
                if j >= 0 and t - bgt[j] <= 3600:
                    v.append((t - bgt[j]) / 60.0)
        v = np.array(v)
        lag[k] = {"n": int(v.size), "median_min": round(float(np.median(v)), 1), "within_5": round(float((v <= 5).mean()), 3),
                  "within_15": round(float((v <= 15).mean()), 3)}
    out["lag_after_blood_gas"] = lag
    # time of day and minute of the hour of CBCs outside the case (ward), from the reconstructed clock
    tod0 = np.mod(-c.set_index("caseid")["adm"], DAY)
    cb = panels[panels["kind"] == "cbc"].copy()
    cb["tod"] = np.mod(cb["caseid"].map(tod0) + cb["dt"], DAY)
    ward = cb[(cb["dt"] < 0) | (cb["dt"] > cb["caseid"].map(c.set_index("caseid")["caseend"]))]
    hrs = (ward["tod"] // 3600).astype(int).value_counts().sort_index()
    out["ward_cbc_hour_counts"] = {f"{int(k):02d}": int(v) for k, v in hrs.items()}
    mins = ((ward["tod"] % 3600) // 60).astype(int).value_counts().sort_index()
    out["ward_cbc_minute_counts"] = {str(int(k)): int(v) for k, v in mins.items()}
    return out


def timestamp_derived_intervals(root: Path) -> dict:
    """Intervals between charted events against the difference of the events' posterior means (vitaldb_resolution's
    timestamp models), the interval's posterior SD (the two events taken as independent given the device
    references), and the charted intervals' share on multiples of 15 and 30 minutes."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # the package in this repository
    import vitaldb_resolution as vr
    reg = vr.load()
    c = clinical(root)
    rows = c.to_dict("records")
    ev = {}
    for e in ("anestart", "opstart", "opend", "aneend"):
        m, sd, ok = [], [], []
        for r in rows:
            d = reg.case_distribution(r, e)
            sm = d.summary() if d.status == "ok" else None
            m.append(sm["mean"] if sm else np.nan)
            sd.append(sm["sd"] if sm else np.nan)
        ev[e] = (np.array(m, float), np.array(sd, float))
    pairs = {"operation": ("opstart", "opend"), "anaesthesia": ("anestart", "aneend"),
             "emergence": ("opend", "aneend"), "induction_to_incision": ("anestart", "opstart")}
    out = {"cases": int(len(c)), "intervals": {}}
    for nm, (a, b) in pairs.items():
        charted = (c[b] - c[a]).to_numpy(float)
        post = ev[b][0] - ev[a][0]
        sd = np.sqrt(ev[a][1] ** 2 + ev[b][1] ** 2)
        good = np.isfinite(post) & np.isfinite(sd) & (c[a] > PLACEHOLDER_S).to_numpy() & (c[b] > PLACEHOLDER_S).to_numpy()
        bias = (charted - post)[good] / 60.0
        pos = charted[(charted > 0) & (c[a] > PLACEHOLDER_S).to_numpy() & (c[b] > PLACEHOLDER_S).to_numpy()]
        m5 = pos[np.isclose(np.mod(pos, 300), 0)]
        out["intervals"][nm] = {
            "n": int(good.sum()), "charted_minus_posterior_min": q(bias), "mean_bias_min": round(float(np.mean(bias)), 2),
            "posterior_sd_min": q(sd[good] / 60.0, ps=(50, 90)),
            "multiples_of_5": int(m5.size),
            "share_on_15": share(int(np.isclose(np.mod(m5, 900), 0).sum()), int(m5.size)),
            "share_on_30": share(int(np.isclose(np.mod(m5, 1800), 0).sum()), int(m5.size))}
    return out


def heap_ratio_log(v: pd.Series, base: int, odd_only: bool = False, w: int = 4) -> dict:
    """As heap_ratio, but the quadratic is fitted to log counts (count + 0.5), so an exponential rise or fall (a
    steep flank, as near a ceiling) is followed instead of overshot."""
    x = v[np.isclose(v, np.round(v))].round().astype(int)
    cnt = x.value_counts()
    lo, hi = int(x.min()), int(x.max())
    O = E = 0.0
    n = 0
    per = {}
    for m in range((lo // base + 1) * base, hi, base):
        if odd_only and m % 10 == 0:
            continue
        xs = [m + d for d in range(-w, w + 1) if d and (m + d) % 5]
        ys = np.log(np.array([cnt.get(t, 0) for t in xs], float) + 0.5)
        e = max(float(np.exp(np.polyval(np.polyfit(xs, ys, 2), m)) - 0.5), 0.5)
        O += cnt.get(m, 0)
        E += e
        n += 1
        if cnt.get(m, 0) >= 100:
            per[str(m)] = {"observed": int(cnt.get(m, 0)), "expected": round(e, 1)}
    return {"screened": int(len(x)), "multiples": n, "observed": int(O), "expected": round(E, 1),
            "ratio": round(O / E, 3), "z": round((O - E) / np.sqrt(E), 1), "per_value_n100": per}


def lab_heaping_sao2_sodium(root: Path) -> dict:
    """The two whole-number analytes the all-variable screen flags (SaO2 at odd multiples of 5, sodium at multiples
    of 10): counts near the mode; the same ratio on each other residue (the values shifted by 1-4, a placebo for
    the fit); the ratio with a log-quadratic fit; by analyser (lab_sources); by time of day (08:00-18:00 against
    the rest, clock time from -adm modulo a day plus dt)."""
    c = clinical(root)
    L = lab(root)
    L["src"] = lab_sources(L)
    start = (-c.set_index("caseid")["adm"]).mod(DAY)
    L["hour"] = ((L["caseid"].map(start) + L["dt"]).mod(DAY) / 3600.0)
    out = {}
    for nm, base, odd, rng in (("sao2", 5, True, (85, 101)), ("na", 10, False, (125, 152))):
        g = L[L["name"] == nm]
        v = g["result"]
        cnt = v[np.isclose(v, np.round(v))].round().astype(int).value_counts()
        r = {"n": int(len(v)), "counts": {str(k): int(cnt.get(k, 0)) for k in range(*rng)},
             "quadratic": heap_ratio(v, base, odd_only=odd),
             "placebo_shift": {str(k): heap_ratio(v - k, base, odd_only=odd)["ratio"] for k in (1, 2, 3, 4)},
             "log_quadratic": heap_ratio_log(v, base, odd_only=odd),
             "log_placebo_shift": {str(k): heap_ratio_log(v - k, base, odd_only=odd)["ratio"] for k in (1, 2, 3, 4)}}
        r["by_source"] = {s_: {"n": int(len(gg)), "quadratic": heap_ratio(gg["result"], base, odd_only=odd)["ratio"],
                               "log_quadratic": heap_ratio_log(gg["result"], base, odd_only=odd)["ratio"]}
                          for s_, gg in g.groupby("src") if len(gg) >= 500}
        day = g["hour"].between(8, 18, inclusive="left")
        r["by_time"] = {k: {"n": int(len(gg)), "quadratic": heap_ratio(gg["result"], base, odd_only=odd)["ratio"],
                            "log_quadratic": heap_ratio_log(gg["result"], base, odd_only=odd)["ratio"]}
                        for k, gg in (("08-18", g[day]), ("18-08", g[~day])) if len(gg) >= 500}
        out[nm] = r
    return out


STUDIES = {
    "dates-admission-discharge-los": dates_admission_discharge_los,
    "lab-preop-pao2-decimal": lab_preop_pao2_decimal,
    "timestamp-clock-reconstruction": timestamp_clock_reconstruction,
    "timestamp-imputed-and-conformance": timestamp_imputed_and_conformance,
    "value-bolus-drug-doses": value_bolus_drug_doses,
    "value-intraop-colloid-containers": value_intraop_colloid_containers,
    "value-registry-unmodelled": value_registry_unmodelled,
    "lab-haematocrit-two-analysers": lab_haematocrit_two_analysers,
    "lab-computed-value-grids": lab_computed_value_grids,
    "heaping-screen-all-variables": heaping_screen_all_variables,
    "lab-heaping-sao2-sodium": lab_heaping_sao2_sodium,
    "timestamp-lab-times-meaning": timestamp_lab_times_meaning,
    "timestamp-derived-intervals": timestamp_derived_intervals,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study", choices=sorted(STUDIES) + ["all"])
    ap.add_argument("--root", required=True, type=Path, help="the VitalDB release folder (clinical_data.csv, lab_data.csv)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name in (sorted(STUDIES) if a.study == "all" else [a.study]):
        res = {"study": name, "release": str(a.root.name)} | STUDIES[name](a.root)
        (a.out / f"{name}.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        print(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:3000])


if __name__ == "__main__":
    main()
