#!/usr/bin/env python3
"""A6P4 - Explainable LinUCB next-best-action (NBA) recommendation for utility
energy-efficiency programs.  Runs every experiment of the paper:

  * preprocessing (median imputation, IQR winsorising, standardisation; fitted per training fold)
  * eligibility rule engine (hard constraint) and its agreement with the logged `eligible` flag
  * online replay learning (Li et al. 2011) on the training stream for all policies
  * frozen-policy off-policy evaluation on held-out customers (IPS and doubly robust),
    ranking metrics (HR@1, NDCG@3), kWh saved per contact
  * ablation, alpha sensitivity, explanation fidelity / stability / sign agreement
  * 5-fold customer-disjoint cross-validation (mean +- std) + paired bootstrap on pooled out-of-fold estimates
Outputs: ../results/*.csv, ../results/metrics.json, ../figures/*.png (300 dpi) and ../results/*.jpg
Usage:   python3 run_all.py
"""
import os, json, time, subprocess, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, Ridge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "A6P4_energy_program_offers.csv")
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
SEED = 2026
K_FOLDS = 5
ARMS = ["heat_pump_rebate", "smart_thermostat", "home_weatherization", "ev_tou_rate",
        "demand_response_enrollment", "led_lighting_kit", "solar_battery_incentive", "home_energy_audit"]
ARM_SHORT = ["HeatPump", "Thermostat", "Weatherise", "EV-TOU", "DemandResp", "LED kit", "Solar+Batt", "Audit"]
NA = len(ARMS)
PROP = 1.0 / NA                       # logging propensity (uniform randomised policy)
ALPHA_GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
LAMBDA = 1.0
EPS = 0.1

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "figure.dpi": 100})
GREY = "#7f7f7f"; ACC = "#c0392b"
COL = {"Logging (uniform)": "#bbbbbb", "Rule-based": "#8c8c8c", "Popularity": "#5a5a5a",
       "Epsilon-greedy": "#2f6f9f", "LinUCB (proposed)": ACC, "LinUCB-kWh": "#e67e22"}


# ---------------------------------------------------------------- data & rules
def load():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    m = df.timestamp.dt.month
    df["winter"] = m.isin([11, 12, 1, 2]).astype(int)
    df["summer"] = m.isin([6, 7, 8]).astype(int)
    df["season"] = np.select([df.winter == 1, df.summer == 1], ["winter", "summer"], "shoulder")
    df["arm"] = df.offer.map({a: i for i, a in enumerate(ARMS)}).astype(int)
    return df


def eligibility(df):
    """Programme eligibility rules (hard constraints), one column per arm."""
    own = df.owner_occupied.values == 1
    E = np.ones((len(df), NA), dtype=bool)
    E[:, 0] = own & (df.heating_fuel.values != "heat_pump")
    E[:, 1] = df.smart_thermostat.values == 0
    E[:, 2] = own
    E[:, 3] = df.has_ev.values == 1
    E[:, 6] = own & (df.home_type.values != "apartment")
    return E


NUM = ["home_age_years", "floor_area_m2", "digital_engagement", "annual_kwh", "peak_usage_share",
       "recent_bill_change_pct"]
WINS = ["annual_kwh", "floor_area_m2", "home_age_years"]


class Prep:
    """Fold-wise preprocessing: median imputation, IQR winsorising, z-scoring."""
    def fit(self, d):
        self.med = d[NUM].median()
        self.lim = {}
        for c in WINS:
            q1, q3 = d[c].quantile([0.25, 0.75]); iqr = q3 - q1
            self.lim[c] = (q1 - 3 * iqr, q3 + 3 * iqr)
        z = self._clean(d)
        self.mu = z[NUM].mean(); self.sd = z[NUM].std() + 1e-9
        return self

    def _clean(self, d):
        z = d.copy()
        z[NUM] = z[NUM].fillna(self.med)
        for c in WINS:
            z[c] = z[c].clip(*self.lim[c])
        return z

    def transform(self, d):
        z = self._clean(d)
        s = (z[NUM] - self.mu) / self.sd
        F = pd.DataFrame(index=d.index)
        F["bias"] = 1.0
        F["home_age"] = s.home_age_years; F["floor_area"] = s.floor_area_m2
        F["engagement"] = s.digital_engagement; F["annual_kwh"] = s.annual_kwh
        F["peak_share"] = s.peak_usage_share; F["bill_change"] = s.recent_bill_change_pct
        F["owner"] = d.owner_occupied
        F["apartment"] = (d.home_type == "apartment").astype(float)
        F["townhouse"] = (d.home_type == "townhouse").astype(float)
        F["inc_low"] = (d.income_band == "low").astype(float)
        F["inc_high"] = (d.income_band == "high").astype(float)
        F["fuel_elec_res"] = (d.heating_fuel == "electric_resistance").astype(float)
        F["fuel_oil"] = (d.heating_fuel == "oil").astype(float)
        F["fuel_heat_pump"] = (d.heating_fuel == "heat_pump").astype(float)
        F["has_ev"] = d.has_ev; F["has_solar"] = d.has_rooftop_solar
        F["has_thermostat"] = d.smart_thermostat
        F["prior_part"] = np.log1p(d.prior_program_participations)
        F["fatigue"] = np.clip(d.contacts_last_30d - 2, 0, None)
        F["ch_app"] = (d.channel == "app_push").astype(float)
        F["ch_email"] = (d.channel == "email").astype(float)
        F["ch_call"] = (d.channel == "call_center").astype(float)
        F["app_x_eng"] = F.ch_app * F.engagement
        F["winter"] = d.winter; F["summer"] = d.summer
        return F.astype(float)


FEATURES = None  # filled at run time
GROUPS = {  # feature groups used by the ablation
    "customer": ["home_age", "floor_area", "engagement", "annual_kwh", "peak_share", "bill_change", "owner",
                 "apartment", "townhouse", "inc_low", "inc_high", "fuel_elec_res", "fuel_oil", "fuel_heat_pump",
                 "has_ev", "has_solar", "has_thermostat", "prior_part", "fatigue", "app_x_eng"],
    "situational": ["ch_app", "ch_email", "ch_call", "app_x_eng", "winter", "summer"],
}


# ---------------------------------------------------------------- policies
class LinUCB:
    """Disjoint LinUCB (one ridge model per arm) with eligibility masking.
    score_a(x) = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)."""
    name = "LinUCB"

    def __init__(self, d, alpha, lam=LAMBDA, mask=True, cols=None, seed=0):
        self.alpha, self.mask = alpha, mask
        self.cols = cols                       # indices of active features (ablation)
        dd = d if cols is None else len(cols)
        self.Ainv = np.stack([np.eye(dd) / lam for _ in range(NA)])
        self.b = np.zeros((NA, dd))
        self.rng = np.random.default_rng(seed)

    def _x(self, X):
        return X if self.cols is None else X[..., self.cols]

    def theta(self):
        return np.einsum("aij,aj->ai", self.Ainv, self.b)

    def scores(self, X, E, explore=True):
        X = self._x(np.atleast_2d(X))
        th = self.theta()
        mean = X @ th.T                                                     # n x A
        if explore and self.alpha > 0:
            var = np.einsum("ni,aij,nj->na", X, self.Ainv, X)
            mean = mean + self.alpha * np.sqrt(np.maximum(var, 0))
        if self.mask:
            mean = np.where(E, mean, -np.inf)
        return mean

    def select(self, x, e):
        s = self.scores(x, e[None, :])[0]
        s = s + 1e-9 * self.rng.random(NA)                                  # random tie-breaking
        return int(np.argmax(s))

    def update(self, x, a, r):
        x = self._x(x)
        Ai = self.Ainv[a]; Ax = Ai @ x
        self.Ainv[a] = Ai - np.outer(Ax, Ax) / (1.0 + x @ Ax)               # Sherman-Morrison
        self.b[a] += r * x


class EpsGreedy:
    name = "Epsilon-greedy"

    def __init__(self, eps=EPS, seed=0):
        self.eps = eps; self.n = np.zeros(NA); self.s = np.zeros(NA); self.rng = np.random.default_rng(seed)

    def means(self):
        return np.where(self.n > 0, self.s / np.maximum(self.n, 1), 1.0)    # optimistic for unseen arms

    def scores(self, X, E, explore=False):
        return np.where(E, np.tile(self.means(), (len(E), 1)), -np.inf)

    def select(self, x, e):
        idx = np.flatnonzero(e)
        if self.rng.random() < self.eps:
            return int(self.rng.choice(idx))
        m = self.means()[idx] + 1e-9 * self.rng.random(len(idx))
        return int(idx[np.argmax(m)])

    def update(self, x, a, r):
        self.n[a] += 1; self.s[a] += r


class Popularity:
    """Ranks offers by their historical (logged) acceptance rate among eligible customers."""
    name = "Popularity"

    def fit_log(self, d, E):
        el = E[np.arange(len(d)), d.arm.values]
        self.rate = np.array([d.accepted.values[(d.arm.values == a) & el].mean() for a in range(NA)])
        return self

    def scores(self, X, E, explore=False):
        return np.where(E, np.tile(self.rate, (len(E), 1)), -np.inf)


class RuleBased:
    """Hand-written utility marketing playbook (priority cascade), eligibility-aware."""
    name = "Rule-based"

    def scores_df(self, d, E):
        n = len(d); S = np.zeros((n, NA))
        hf = d.heating_fuel.values; inc = d.income_band.values
        age = d.home_age_years.fillna(d.home_age_years.median()).values
        # priority points: higher = earlier in the cascade
        S[:, 3] += 60 * (d.has_ev.values == 1)
        S[:, 0] += 50 * np.isin(hf, ["oil", "electric_resistance"])
        S[:, 2] += 40 * ((inc == "low") & (age > 40))
        S[:, 6] += 35 * ((inc == "high") & (d.has_rooftop_solar.values == 1))
        S[:, 4] += 30 * (d.summer.values == 1)
        S[:, 1] += 20
        S[:, 7] += 10 * (d.recent_bill_change_pct.values > 10)
        S[:, 5] += 5
        S += 1e-3 * np.arange(NA)[::-1]                                     # deterministic tie order
        return np.where(E, S, -np.inf)


# ---------------------------------------------------------------- evaluation helpers
def replay(policy, X, E, arm, r, record_every=200):
    """Li et al. (2011) replay: an event counts only when the policy matches the logged arm."""
    matched = 0; tot = 0.0; curve = []
    for t in range(len(X)):
        a = policy.select(X[t], E[t])
        if a == arm[t]:
            policy.update(X[t], a, r[t]); matched += 1; tot += r[t]
        if (t + 1) % record_every == 0:
            curve.append(tot / max(matched, 1))
    return tot / max(matched, 1), matched, np.array(curve)


def ranks_from_scores(S):
    order = np.argsort(-S, axis=1, kind="stable")
    R = np.empty_like(order); R[np.arange(len(S))[:, None], order] = np.arange(1, NA + 1)
    return order[:, 0], R


def offpolicy(choice, arm, acc, kwh, qacc, qkwh):
    """IPS and doubly-robust value estimates of a deterministic policy (per-row terms returned)."""
    n = len(arm); idx = np.arange(n)
    hit = (choice == arm).astype(float)
    ips_a = hit * acc / PROP; ips_k = hit * kwh / PROP
    dr_a = qacc[idx, choice] + hit * (acc - qacc[idx, arm]) / PROP
    dr_k = qkwh[idx, choice] + hit * (kwh - qkwh[idx, arm]) / PROP
    return ips_a, ips_k, dr_a, dr_k


def ranking(R, arm, acc):
    pos = acc == 1
    rk = R[np.arange(len(arm)), arm][pos]
    hr1 = (rk == 1).astype(float)
    ndcg3 = np.where(rk <= 3, 1.0 / np.log2(rk + 1), 0.0)
    return hr1, ndcg3


def fit_direct_model(Xtr, Etr, dtr, Xte, Ete):
    """Reward model q(x,a) for DR: per-arm logistic acceptance x per-arm ridge savings-given-acceptance."""
    qa = np.zeros((len(Xte), NA)); qk = np.zeros((len(Xte), NA))
    arm = dtr.arm.values; acc = dtr.accepted.values; kwh = dtr.annual_kwh_savings.values
    for a in range(NA):
        m = (arm == a) & Etr[:, a]
        y = acc[m]
        if y.sum() >= 5:
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr[m][:, 1:], y)
            pa = lr.predict_proba(Xte[:, 1:])[:, 1]
        else:
            pa = np.full(len(Xte), y.mean() if len(y) else 0.0)
        mk = m & (acc == 1)
        if mk.sum() >= 10:
            rg = Ridge(alpha=10.0).fit(Xtr[mk][:, 1:], kwh[mk])
            sk = np.maximum(rg.predict(Xte[:, 1:]), 0)
        else:
            sk = np.full(len(Xte), kwh[mk].mean() if mk.any() else 0.0)
        qa[:, a] = np.where(Ete[:, a], pa, 0.0); qk[:, a] = qa[:, a] * sk
    return qa, qk


# ---------------------------------------------------------------- main experiment
def main():
    global FEATURES
    t0 = time.time()
    df = load()
    E_all = eligibility(df)
    rule_agree = float((E_all[np.arange(len(df)), df.arm.values] == (df.eligible.values == 1)).mean())
    print(f"eligibility rule agreement with logged flag: {rule_agree:.4f}")

    # ---------- dataset summary tables
    miss = df[["home_age_years", "floor_area_m2", "digital_engagement"]].isna().mean() * 100
    q1, q3 = df.annual_kwh.quantile([.25, .75]); out_kwh = int((df.annual_kwh > q3 + 3 * (q3 - q1)).sum())
    q1, q3 = df.floor_area_m2.quantile([.25, .75]); out_fa = int((df.floor_area_m2 > q3 + 3 * (q3 - q1)).sum())
    summ = pd.DataFrame([
        ["Interactions / customers", f"{len(df):,} / {df.customer_id.nunique():,}"],
        ["Period", f"{df.timestamp.min():%b %Y} - {df.timestamp.max():%b %Y}"],
        ["Offers (arms) / channels", f"{NA} / {df.channel.nunique()}"],
        ["Context features (d)", "26 incl. bias"],
        ["Overall acceptance rate", f"{100*df.accepted.mean():.2f}%"],
        ["Eligible share of logged offers", f"{100*df.eligible.mean():.1f}%"],
        ["Acceptance when eligible", f"{100*df.accepted[df.eligible==1].mean():.2f}%"],
        ["Mean savings if accepted (kWh/yr)", f"{df.annual_kwh_savings[df.accepted==1].mean():.0f}"],
        ["Missing (age / area / engagement)", " / ".join(f"{v:.1f}%" for v in miss.values)],
        ["IQR outliers (kWh / area)", f"{out_kwh} / {out_fa}"],
    ], columns=["Property", "Value"])
    summ.to_csv(os.path.join(RES, "table_dataset_summary.csv"), index=False)
    per_arm = df.groupby("arm").agg(eligible=("eligible", "mean"), accepted=("accepted", "mean"),
                                    kwh=("annual_kwh_savings", "mean")).reset_index()
    per_arm["offer"] = [ARMS[a] for a in per_arm.arm]
    per_arm["accept_if_eligible"] = [df.accepted[(df.arm == a) & (df.eligible == 1)].mean() for a in per_arm.arm]
    per_arm.to_csv(os.path.join(RES, "table_dataset_per_offer.csv"), index=False)

    gkf = GroupKFold(n_splits=K_FOLDS)
    rng_perm = np.random.default_rng(SEED)
    groups = df.customer_id.values
    # shuffle customer ids -> randomised but reproducible GroupKFold assignment
    ucust = np.unique(groups); perm = dict(zip(ucust, rng_perm.permutation(len(ucust))))
    gnum = np.array([perm[g] for g in groups])

    rows = []            # per-fold per-policy metrics
    curves = {}          # replay learning curves
    oof = {}             # pooled out-of-fold per-row terms for bootstrap
    abl_rows = []; sens_rows = []
    thetas = []; expl_rows = []; local_example = None; mix_rows = []

    for k, (tr, te) in enumerate(gkf.split(df, groups=gnum)):
        dtr, dte = df.iloc[tr], df.iloc[te]                     # dtr keeps chronological order
        prep = Prep().fit(dtr)
        Ftr = prep.transform(dtr); Fte = prep.transform(dte)
        FEATURES = list(Ftr.columns)
        Xtr, Xte = Ftr.values, Fte.values
        Etr, Ete = E_all[tr], E_all[te]
        arm_tr, arm_te = dtr.arm.values, dte.arm.values
        acc_tr, acc_te = dtr.accepted.values.astype(float), dte.accepted.values.astype(float)
        kwh_tr, kwh_te = dtr.annual_kwh_savings.values, dte.annual_kwh_savings.values
        d = Xtr.shape[1]
        qa, qk = fit_direct_model(Xtr, Etr, dtr, Xte, Ete)

        def evaluate(name, S, extra=None, store=True):
            choice, R = ranks_from_scores(S)
            ia, ik, da, dk = offpolicy(choice, arm_te, acc_te, kwh_te, qa, qk)
            hr, nd = ranking(R, arm_te, acc_te)
            elig_ok = E_all[te][np.arange(len(te)), choice].mean()
            row = dict(fold=k, policy=name, ips_accept=ia.mean(), dr_accept=da.mean(), ips_kwh=ik.mean(),
                       dr_kwh=dk.mean(), hr1=hr.mean(), ndcg3=nd.mean(), eligible_rate=elig_ok)
            if extra: row.update(extra)
            if store:
                o = oof.setdefault(name, {"ia": [], "ik": [], "da": [], "dk": [], "hr": [], "nd": []})
                for key, v in zip(["ia", "ik", "da", "dk", "hr", "nd"], [ia, ik, da, dk, hr, nd]):
                    o[key].append(v)
            return row, choice

        # logging policy (uniform random over 8 offers): on-policy value = empirical mean
        rows.append(dict(fold=k, policy="Logging (uniform)", ips_accept=acc_te.mean(), dr_accept=acc_te.mean(),
                         ips_kwh=kwh_te.mean(), dr_kwh=kwh_te.mean(), hr1=np.nan, ndcg3=np.nan,
                         eligible_rate=E_all[te][np.arange(len(te)), arm_te].mean(), replay_ctr=acc_tr.mean()))
        # rule-based
        rb = RuleBased()
        Str = rb.scores_df(dtr, Etr); ch_tr = Str.argmax(1); m = ch_tr == arm_tr
        r_, _ = evaluate("Rule-based", rb.scores_df(dte, Ete), dict(replay_ctr=acc_tr[m].mean()))
        rows.append(r_)
        # popularity (fitted on full training log)
        pop = Popularity().fit_log(dtr, Etr)
        ch_tr = pop.scores(Xtr, Etr).argmax(1); m = ch_tr == arm_tr
        r_, ch_pop = evaluate("Popularity", pop.scores(Xte, Ete), dict(replay_ctr=acc_tr[m].mean()))
        rows.append(r_)
        # epsilon-greedy (online replay)
        eg = EpsGreedy(seed=SEED + k)
        ctr, mt, cv = replay(eg, Xtr, Etr, arm_tr, acc_tr); curves.setdefault("Epsilon-greedy", []).append(cv)
        r_, _ = evaluate("Epsilon-greedy", eg.scores(Xte, Ete), dict(replay_ctr=ctr, matched=mt)); rows.append(r_)

        # alpha sensitivity (also selects alpha on the TRAINING replay CTR only)
        lin_models = {}
        for al in ALPHA_GRID:
            lu = LinUCB(d, al, seed=SEED + k)
            ctr, mt, cv = replay(lu, Xtr, Etr, arm_tr, acc_tr)
            r_, _ = evaluate(f"alpha={al}", lu.scores(Xte, Ete, explore=False), dict(replay_ctr=ctr), store=False)
            r_["alpha"] = al; sens_rows.append(r_); lin_models[al] = (lu, cv, ctr, mt)
        rows_k = pd.DataFrame([r for r in sens_rows if r["fold"] == k])
        al_star = float(rows_k.sort_values(["replay_ctr", "alpha"], ascending=[False, True]).alpha.iloc[0])
        lu, cv, ctr, mt = lin_models[al_star]
        curves.setdefault("LinUCB (proposed)", []).append(cv)
        S_lin = lu.scores(Xte, Ete, explore=False)
        r_, ch_lin = evaluate("LinUCB (proposed)", S_lin, dict(replay_ctr=ctr, matched=mt, alpha=al_star))
        rows.append(r_)
        # value-aware variant: reward = kWh savings (in MWh)
        luk = LinUCB(d, al_star, seed=SEED + k)
        ctr_k, mt_k, _ = replay(luk, Xtr, Etr, arm_tr, kwh_tr / 1000.0)
        r_, _ = evaluate("LinUCB-kWh", luk.scores(Xte, Ete, explore=False), dict(replay_ctr=np.nan, matched=mt_k))
        rows.append(r_)
        # rule-based / popularity / logging have no online curve; keep their static replay CTR as flat lines
        for nm in ["Rule-based", "Popularity"]:
            curves.setdefault(nm, []).append(None)

        # ---------- ablation (same alpha)
        idx = {f: i for i, f in enumerate(FEATURES)}
        def cols_without(names):
            return [i for f, i in idx.items() if f not in names]
        abl_cfg = [
            ("Full LinUCB", dict(alpha=al_star)),
            ("w/o eligibility mask", dict(alpha=al_star, mask=False)),
            ("w/o exploration (alpha=0)", dict(alpha=0.0)),
            ("w/o customer context", dict(alpha=al_star, cols=cols_without(GROUPS["customer"]))),
            ("w/o season/channel", dict(alpha=al_star, cols=cols_without(GROUPS["situational"]))),
            ("Context-free (bias only)", dict(alpha=al_star, cols=[idx["bias"]])),
        ]
        for nm, cfg in abl_cfg:
            if nm == "Full LinUCB":
                model = lu
            elif nm == "w/o exploration (alpha=0)":
                model = lin_models[0.0][0]
            else:
                model = LinUCB(d, cfg["alpha"], mask=cfg.get("mask", True), cols=cfg.get("cols"), seed=SEED + k)
                replay(model, Xtr, Etr, arm_tr, acc_tr)
            r_, _ = evaluate(nm, model.scores(Xte, Ete, explore=False), store=False)
            abl_rows.append(r_)

        # ---------- explanations: additive contributions theta_a * x
        th = lu.theta(); thetas.append(th)
        contrib = Xte[:, None, :] * th[None, :, :]                          # n x A x d
        csel = contrib[np.arange(len(te)), ch_lin, :]                        # contributions of recommended arm
        nonbias = np.array([f != "bias" for f in FEATURES])
        absc = np.abs(csel[:, nonbias])
        top3 = np.argsort(-absc, axis=1)[:, :3]
        # (i) additive fidelity: sum of contributions reproduces the score
        fid_err = np.abs(csel.sum(1) - S_lin[np.arange(len(te)), ch_lin]).max()
        # (ii) deletion test: neutralise top-3 vs 3 random features (set to training mean = 0 / binary -> mean)
        Xmean = Xtr.mean(0)
        rng = np.random.default_rng(SEED + 100 + k)
        nb_idx = np.flatnonzero(nonbias)
        Xd = Xte.copy(); Xr = Xte.copy()
        for i in range(len(te)):
            ft = nb_idx[top3[i]]; Xd[i, ft] = Xmean[ft]
            fr = rng.choice(nb_idx, 3, replace=False); Xr[i, fr] = Xmean[fr]
        flip_top = (lu.scores(Xd, Ete, explore=False).argmax(1) != ch_lin).mean()
        flip_rand = (lu.scores(Xr, Ete, explore=False).argmax(1) != ch_lin).mean()
        # (iii) stability: 5 nearest neighbours in standardised context
        from sklearn.neighbors import NearestNeighbors
        Z = Xte[:, 1:]
        nn = NearestNeighbors(n_neighbors=6).fit(Z)
        _, nbr = nn.kneighbors(Z)
        same_arm = []; jac = []
        for i in range(len(te)):
            for j in nbr[i, 1:]:
                s = ch_lin[j] == ch_lin[i]; same_arm.append(s)
                if s:
                    a_, b_ = set(top3[i]), set(top3[j]); jac.append(len(a_ & b_) / len(a_ | b_))
        expl_rows.append(dict(fold=k, additive_max_abs_err=fid_err, deletion_flip_top3=flip_top,
                              deletion_flip_random3=flip_rand, neighbour_same_offer=np.mean(same_arm),
                              neighbour_top3_jaccard=np.mean(jac)))
        if k == 0:
            # pick an example customer: winter, electric resistance, owner
            cand = np.flatnonzero((dte.heating_fuel.values == "electric_resistance") & (dte.winter.values == 1) &
                                  (dte.owner_occupied.values == 1) & (ch_lin == 0))
            i = int(cand[0]) if len(cand) else 0
            s_sorted = np.argsort(-S_lin[i]); second = int(s_sorted[1])
            local_example = dict(row=dte.iloc[i][["interaction_id", "home_type", "heating_fuel", "income_band",
                                                  "owner_occupied", "has_ev", "channel", "season"]].to_dict(),
                                 rec=int(ch_lin[i]), second=second, contrib_rec=csel[i].tolist(),
                                 contrib_second=contrib[i, second].tolist(), score=S_lin[i].tolist(),
                                 eligible=Ete[i].tolist())
        for nm, ch in [("LinUCB (proposed)", ch_lin), ("Popularity", ch_pop)]:
            for s in ["winter", "shoulder", "summer"]:
                msk = dte.season.values == s
                cnt = np.bincount(ch[msk], minlength=NA) / msk.sum()
                mix_rows.append(dict(fold=k, policy=nm, season=s, **{ARMS[a]: cnt[a] for a in range(NA)}))
        print(f"fold {k}: alpha*={al_star}  done  ({time.time()-t0:.0f}s)")

    # ---------------------------------------------------------------- aggregate & save
    R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "main_comparison_per_fold.csv"), index=False)
    order = ["Logging (uniform)", "Rule-based", "Popularity", "Epsilon-greedy", "LinUCB (proposed)", "LinUCB-kWh"]
    mets = ["replay_ctr", "ips_accept", "dr_accept", "ips_kwh", "dr_kwh", "hr1", "ndcg3"]
    agg = R.groupby("policy")[mets].agg(["mean", "std"]).reindex(order)
    tab = pd.DataFrame(index=order)
    for m_ in mets:
        sc = 100 if m_ in ("replay_ctr", "ips_accept", "dr_accept") else 1
        dec = 1 if m_ in ("ips_kwh", "dr_kwh") else (2 if sc == 100 else 3)
        tab[m_] = [("-" if np.isnan(agg.loc[p, (m_, "mean")]) else
                    f"{sc*agg.loc[p,(m_,'mean')]:.{dec}f} ± {sc*agg.loc[p,(m_,'std')]:.{dec}f}") for p in order]
    tab.index.name = "policy"; tab.to_csv(os.path.join(RES, "table_main_comparison.csv"))
    print(tab.to_string())

    A = pd.DataFrame(abl_rows); A.to_csv(os.path.join(RES, "ablation_per_fold.csv"), index=False)
    aorder = [c[0] for c in abl_cfg]
    aagg = A.groupby("policy")[["ips_accept", "dr_accept", "dr_kwh", "hr1", "eligible_rate"]].agg(["mean", "std"]).reindex(aorder)
    atab = pd.DataFrame(index=aorder)
    for m_, sc, dec in [("ips_accept", 100, 2), ("dr_accept", 100, 2), ("dr_kwh", 1, 1), ("hr1", 1, 3), ("eligible_rate", 100, 1)]:
        atab[m_] = [f"{sc*aagg.loc[p,(m_,'mean')]:.{dec}f} ± {sc*aagg.loc[p,(m_,'std')]:.{dec}f}" for p in aorder]
    atab.index.name = "variant"; atab.to_csv(os.path.join(RES, "table_ablation.csv"))
    print(atab.to_string())

    S = pd.DataFrame(sens_rows); S.to_csv(os.path.join(RES, "sensitivity_alpha_per_fold.csv"), index=False)
    sagg = S.groupby("alpha")[["replay_ctr", "dr_accept", "ips_accept", "dr_kwh"]].agg(["mean", "std"])
    sagg.to_csv(os.path.join(RES, "table_sensitivity_alpha.csv"))
    X_ = pd.DataFrame(expl_rows); X_.to_csv(os.path.join(RES, "explanation_metrics_per_fold.csv"), index=False)
    M = pd.DataFrame(mix_rows); Mg = M.groupby(["policy", "season"])[ARMS].mean(); Mg.to_csv(os.path.join(RES, "offer_mix_by_season.csv"))

    # pooled out-of-fold paired bootstrap: LinUCB vs best baseline (by DR acceptance)
    base_names = ["Rule-based", "Popularity", "Epsilon-greedy"]
    best_base = max(base_names, key=lambda b: R[R.policy == b].dr_accept.mean())
    rngb = np.random.default_rng(SEED)
    boot = {}
    for key in ["da", "dk", "ia"]:
        a_ = np.concatenate(oof["LinUCB (proposed)"][key]); b_ = np.concatenate(oof[best_base][key])
        diff = a_ - b_; n = len(diff)
        bs = np.array([diff[rngb.integers(0, n, n)].mean() for _ in range(2000)])
        boot[key] = dict(mean_diff=float(diff.mean()), ci_low=float(np.percentile(bs, 2.5)),
                         ci_high=float(np.percentile(bs, 97.5)), p_one_sided=float((bs <= 0).mean()))
    pd.DataFrame(boot).T.to_csv(os.path.join(RES, "bootstrap_linucb_vs_best_baseline.csv"))

    # sign agreement of learned coefficients with the known data-generating drivers
    TH = np.mean(thetas, axis=0); fi = {f: i for i, f in enumerate(FEATURES)}
    expected = [(0, "fuel_elec_res", +1), (0, "fuel_oil", +1), (0, "winter", +1), (1, "engagement", +1),
                (1, "winter", +1), (2, "home_age", +1), (2, "inc_low", +1), (2, "winter", +1),
                (3, "peak_share", +1), (3, "inc_high", -1), (4, "summer", +1), (4, "has_thermostat", +1),
                (4, "engagement", +1), (5, "inc_low", +1), (6, "inc_high", +1), (6, "has_solar", +1),
                (7, "bill_change", +1), (7, "home_age", +1)] + [(a, "fatigue", -1) for a in range(NA)]
    sign_ok = [np.sign(TH[a, fi[f]]) == s for a, f, s in expected]
    sign_fold = [np.mean([np.sign(th[a, fi[f]]) == s for a, f, s in expected]) for th in thetas]
    pd.DataFrame(TH, index=ARMS, columns=FEATURES).to_csv(os.path.join(RES, "linucb_coefficients_mean.csv"))

    metrics = dict(
        n_rows=len(df), n_customers=int(df.customer_id.nunique()), eligibility_rule_agreement=rule_agree,
        folds=K_FOLDS, alpha_selected_per_fold=[float(r) for r in R[R.policy == "LinUCB (proposed)"].alpha],
        best_baseline=best_base,
        main={p: {m_: [float(agg.loc[p, (m_, "mean")]), float(agg.loc[p, (m_, "std")])] for m_ in mets} for p in order},
        ablation={p: {m_: [float(aagg.loc[p, (m_, "mean")]), float(aagg.loc[p, (m_, "std")])]
                      for m_ in ["ips_accept", "dr_accept", "dr_kwh", "hr1", "eligible_rate"]} for p in aorder},
        alpha_sensitivity={str(a): {m_: float(sagg.loc[a, (m_, "mean")]) for m_ in ["replay_ctr", "dr_accept", "dr_kwh"]}
                           for a in ALPHA_GRID},
        bootstrap_vs_best_baseline=boot,
        explanations={c: [float(X_[c].mean()), float(X_[c].std())] for c in X_.columns if c != "fold"},
        sign_agreement_mean_theta=float(np.mean(sign_ok)), sign_agreement_per_fold=[float(s) for s in sign_fold],
        n_expected_signs=len(expected), local_example=local_example,
        runtime_s=None,
    )

    # ---------------------------------------------------------------- figures
    make_figures(R, curves, S, TH, local_example, Mg, order)
    metrics["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(RES, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: metrics[k] for k in ["best_baseline", "bootstrap_vs_best_baseline", "explanations",
                                              "sign_agreement_mean_theta", "alpha_selected_per_fold", "runtime_s"]}, indent=1))


def save_fig(fig, name):
    p = os.path.join(FIG, name + ".png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    Image.open(p).convert("RGB").save(os.path.join(RES, name + ".jpg"), quality=92)


def make_figures(R, curves, S, TH, ex, Mg, order):
    # Fig 1 flowchart (Graphviz)
    dot = os.path.join(HERE, "fig1_flowchart.dot")
    png = os.path.join(FIG, "fig1_flowchart.png")
    subprocess.run(["dot", "-Tpng", "-Gdpi=300", dot, "-o", png], check=True)
    Image.open(png).convert("RGB").save(os.path.join(RES, "fig1_flowchart.jpg"), quality=92)
    shutil.copy(dot, os.path.join(RES, "fig1_flowchart.dot"))

    # Fig 2 replay learning curves
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(1, 100) * 200
    for nm in ["Epsilon-greedy", "LinUCB (proposed)"]:
        cs = np.array([c for c in curves[nm]]); L = min(len(c) for c in cs); cs = np.array([c[:L] for c in cs]) * 100
        xx = (np.arange(1, L + 1) * 200) / 1000
        mu, sd = cs.mean(0), cs.std(0)
        ax.plot(xx, mu, color=COL[nm], lw=1.4, label=nm); ax.fill_between(xx, mu - sd, mu + sd, color=COL[nm], alpha=.2, lw=0)
    for nm, ls in [("Rule-based", "--"), ("Popularity", "-."), ("Logging (uniform)", ":")]:
        v = 100 * R[R.policy == nm].replay_ctr.mean()
        ax.axhline(v, color=COL[nm] if nm != "Logging (uniform)" else "#555", ls=ls, lw=1.1, label=f"{nm} (static)")
    ax.set_xlabel("Training-stream position (thousand interactions)")
    ax.set_ylabel("Replay acceptance rate (%)")
    ax.set_ylim(0, 42); ax.grid(alpha=.3, lw=.4)
    ax.legend(loc="upper center", bbox_to_anchor=(.45, -.22), frameon=False, ncol=2, columnspacing=1.0)
    save_fig(fig, "fig2_replay_learning")

    # Fig 3 comparison bars (DR acceptance and DR kWh) with error bars
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.2))
    labs = ["Uniform", "Rules", "Popularity", "ε-greedy", "LinUCB", "LinUCB-kWh"]
    for ax, m_, sc, yl in [(axs[0], "dr_accept", 100, "DR acceptance rate (%)"),
                           (axs[1], "dr_kwh", 1, "DR savings (kWh/yr per offer)")]:
        mu = [sc * R[R.policy == p][m_].mean() for p in order]; sd = [sc * R[R.policy == p][m_].std() for p in order]
        ax.bar(range(len(order)), mu, yerr=sd, capsize=2, color=[COL[p] for p in order], edgecolor="k", lw=.4,
               error_kw=dict(lw=.7))
        ax.set_xticks(range(len(order))); ax.set_xticklabels(labs, rotation=55, ha="right", rotation_mode="anchor", fontsize=6.5)
        ax.set_ylabel(yl); ax.grid(axis="y", alpha=.3, lw=.4)
    fig.tight_layout(w_pad=.8)
    save_fig(fig, "fig3_policy_comparison")

    # Fig 4 alpha sensitivity
    g = S.groupby("alpha")
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    a = np.array(sorted(S.alpha.unique()))
    for m_, nm, c, mk in [("replay_ctr", "Replay acceptance (training stream)", "#2f6f9f", "o"),
                          ("dr_accept", "DR acceptance (held-out)", ACC, "s")]:
        mu = 100 * g[m_].mean().values; sd = 100 * g[m_].std().values
        ax.errorbar(a, mu, yerr=sd, color=c, marker=mk, ms=3.5, lw=1.2, capsize=2, label=nm)
    ax.set_xlabel(r"Exploration weight $\alpha$ (Eq. 4)"); ax.set_ylabel("Acceptance rate (%)")
    ax.set_ylim(32, 46); ax.grid(alpha=.3, lw=.4); ax.legend(frameon=False, loc="lower left")
    save_fig(fig, "fig4_alpha_sensitivity")

    # Fig 5 coefficient heatmap (global explanation)
    show = ["fuel_elec_res", "fuel_oil", "winter", "summer", "home_age", "inc_low", "inc_high", "peak_share",
            "engagement", "app_x_eng", "has_thermostat", "has_solar", "bill_change", "prior_part", "fatigue"]
    fi = {f: i for i, f in enumerate(FEATURES)}
    H = TH[:, [fi[f] for f in show]] * 100
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    vm = np.abs(H).max()
    im = ax.imshow(H, cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
    ax.set_yticks(range(NA)); ax.set_yticklabels(ARM_SHORT, fontsize=7)
    ax.set_xticks(range(len(show))); ax.set_xticklabels(show, rotation=65, ha="right", fontsize=6.5)
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02); cb.set_label(r"$\theta_{a,j}$ (pp acceptance per unit)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    save_fig(fig, "fig5_coefficient_heatmap")

    # Fig 6 local explanation example
    cr = np.array(ex["contrib_rec"]) * 100; cs_ = np.array(ex["contrib_second"]) * 100
    diff = cr - cs_
    top = np.argsort(-np.abs(diff))[:9]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    y = np.arange(len(top))[::-1]
    ax.barh(y + .2, cr[top], height=.4, color=ACC, label=f"{ARM_SHORT[ex['rec']]} (recommended)")
    ax.barh(y - .2, cs_[top], height=.4, color=GREY, label=f"{ARM_SHORT[ex['second']]} (runner-up)")
    ax.set_yticks(y); ax.set_yticklabels([FEATURES[i] for i in top], fontsize=7)
    ax.axvline(0, color="k", lw=.5); ax.set_xlabel(r"Contribution $\theta_{a,j}x_j$ (pp acceptance)")
    ax.legend(frameon=False, loc="lower right"); ax.grid(axis="x", alpha=.3, lw=.4)
    save_fig(fig, "fig6_local_explanation")

    # Fig 7 offer mix by season
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.2), sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, pol in zip(axs, ["Popularity", "LinUCB (proposed)"]):
        sub = Mg.loc[pol].reindex(["winter", "shoulder", "summer"]) * 100
        bottom = np.zeros(3)
        for a in range(NA):
            ax.bar(range(3), sub[ARMS[a]].values, bottom=bottom, color=cmap(a), edgecolor="k", lw=.3,
                   label=ARM_SHORT[a] if pol == "Popularity" else None)
            bottom += sub[ARMS[a]].values
        ax.set_xticks(range(3)); ax.set_xticklabels(["Winter", "Shoulder", "Summer"], fontsize=6.5)
        ax.set_title(pol.replace(" (proposed)", ""), fontsize=7.5)
    axs[0].set_ylabel("Share of recommendations (%)")
    fig.legend(loc="lower center", ncol=4, fontsize=6, frameon=False, bbox_to_anchor=(.5, -.13))
    fig.tight_layout()
    save_fig(fig, "fig7_offer_mix")


if __name__ == "__main__":
    main()
