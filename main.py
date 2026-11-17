#!/usr/bin/env python3
"""GTA VI launch-day Twitch viewer forecaster (shape-library blend).

Idea: the comparables are a library of full launch-day shapes. At the
checkpoint we know only the target's hour-1 print and its 48 hours of
pre-release history, so we match the target against the library on that
partial information alone, in two complementary ways, and blend the
neighbours' realised full-day curves with a small per-hour regression.

  * Library rows: one per comparable launch (the GTA VI category's own
    pre-release row is excluded -- it is not a launch). Each row carries
    its full day curve (hours 1..24, log viewers) plus its partial shape:
    the pre-release median level, the hour-1 jump over that median, and
    the demeaned 24-hour diurnal profile of the pre-release window.
  * Match 1 ("level", 4 neighbours): standardised distance on
    (pre-release median, hour-1 jump). Matches launches of similar
    pre-release size whose hour 1 jumped by a similar amount.
  * Match 2 ("profile", 3 neighbours): the same level distance plus the
    distance between demeaned diurnal profiles, divided by 4. This also
    matches launches whose pre-release audience has the same day/night
    phase -- a proxy for the audience's timezone mix, which is what
    shapes the launch day's rise into the evening.
  * Neighbour forecast: per hour, the inverse-distance weighted mean of
    the neighbours' log levels and of their log ratios to their own
    hour 1, mixed by a weight that fades from pure ratio at hour 2 to
    pure level by hour 18. Early hours track hour 1 closely; the evening
    level is nearly independent of hour 1, so late hours use absolute
    levels. Spread = neighbours' log-level IQR blended with the whole
    library's IQR, floored at 0.30 in log space.
  * Blend with regression: the two neighbour forecasts are averaged and
    blended 50/50 (75/25 in favour of the regression at hours 2-4, where
    hour 1 dominates) with a per-hour OLS regression of log viewers on
    (log hour-1 print, log pre-release median). The regression is the
    stabiliser: pure nearest-neighbour matching is noisy with only ~37
    library rows, and the blend's leave-one-out loss beats either part
    alone. Final spread = mean of the neighbour spread and the
    regression's residual spread, times 0.9.

Validation (see REPORT.md in the work folder): leave-one-comparable-out
over the 37 usable comparables gives mean pinball 0.418 (median 0.349,
worst 1.37, 90% coverage 0.90), against 0.434 for the regression alone,
0.453-0.487 for pure neighbour matching, and 0.93 for the house baseline
on the same held-out launches; the 4 practice cases score 0.209 overall
(per case 0.205 / 0.179 / 0.230 / 0.223), against 0.387 for the house
baseline. Pure shape matching does NOT beat the regression on its own --
the honest finding of this concept -- it only wins as half of the blend,
where it adds the evening-shape information the regression misses
(per-hour LOCO gains of ~0.04-0.05 at hours 17-24).

Deterministic, stdlib only, no network, no randomness.
"""

import json
import math
import os
import statistics as st
from datetime import datetime, timedelta, timezone

# Normal quantiles for the five forecast levels q05..q95.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
HOURS = range(2, 25)

# Constants below were set from leave-one-out evidence only (see
# REPORT.md); practice scores were never used for selection.
PROFILE_DIVISOR = 4.0   # scales the diurnal-profile distance
N_LEVEL = 4             # neighbours for the level match
N_PROFILE = 3           # neighbours for the profile+level match
REG_W = 0.5             # regression weight in the blend, hours 5..24
REG_W_EARLY = 0.75      # regression weight in the blend, hours 2..4
SPREAD_FLOOR_NN = 0.30  # log-space spread floor, neighbour part
SPREAD_FLOOR_REG = 0.15  # log-space spread floor, regression part
SPREAD_MULT = 0.9       # final spread calibration: at 1.0 the LOCO 90%
                        # coverage is 0.93 (bands too wide); at 0.9 it is
                        # 0.897, i.e. the nominal 0.90, and the LOCO loss
                        # is marginally lower too (0.418 vs 0.421)
FALLBACK_JUMP = 3.2     # typical jump if a case has no pre-release data
MIN_PRE = 6             # library rows need at least this many pre points


def iso_to_utc(text):
    """Parse an ISO-8601 timestamp to an aware UTC datetime."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def day_curve(hourly, day_start):
    """Map hour (1..24) -> viewers for one launch day's hourly points."""
    out = {}
    for pt in hourly or []:
        if pt.get("viewers") is None:
            continue
        try:
            h = int((iso_to_utc(pt["from"]) - day_start).total_seconds() // 3600) + 1
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= h <= 24:
            out[h] = float(pt["viewers"])
    return out


def pre_window(hourly, day_start):
    """48 hourly log-viewer values ending at day_start; None where missing."""
    vals = [None] * 48
    start = day_start - timedelta(hours=48)
    for pt in hourly or []:
        if pt.get("viewers") is None:
            continue
        try:
            k = int((iso_to_utc(pt["from"]) - start).total_seconds() // 3600)
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= k < 48:
            v = float(pt["viewers"])
            if v > 0:
                vals[k] = math.log(v)
    return vals


def partial_features(hourly, day_start, curve, p1):
    """Partial-shape features from pre-checkpoint data only.

    hourly: the launch's hourly points; curve: its day curve; p1: hour-1
    print (may be None, in which case curve[1] is used).
    """
    pv = pre_window(hourly, day_start)
    avail = [v for v in pv if v is not None]
    lnpre48 = st.median(avail) if avail else None
    if p1 is None:
        p1 = curve.get(1)
    lnp1 = math.log(max(1.0, p1)) if p1 else None
    # Demeaned 24-hour diurnal profile: mean of the two pre-release days
    # per hour of day, minus its own mean. Needs at least 12 of 24 slots.
    prof = None
    slots = []
    for j in range(24):
        pair = [v for v in (pv[j], pv[j + 24]) if v is not None]
        slots.append(sum(pair) / len(pair) if pair else None)
    have = [v for v in slots if v is not None]
    if len(have) >= 12:
        mean = sum(have) / len(have)
        prof = [(v - mean) if v is not None else None for v in slots]
    jump = (lnp1 - lnpre48) if (lnp1 is not None and lnpre48 is not None) else None
    return {
        "pv": pv,
        "n_pre": len(avail),
        "prof": prof,
        "lnpre48": lnpre48,
        "lnp1": lnp1,
        "jump": jump,
    }


def library_rows(case):
    """Library rows from a case's comparables: (name, features, log curve)."""
    rows = []
    for comp in case.get("comparables") or []:
        name = comp.get("name") or ""
        if "Grand Theft Auto VI" in name:
            continue  # the target category's own history is not a launch
        try:
            ds = iso_to_utc(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        curve = day_curve(comp.get("hourly"), ds)
        if 1 not in curve or curve[1] < 1:
            continue
        feats = partial_features(comp.get("hourly"), ds, curve, curve[1])
        if feats["lnpre48"] is None or feats["n_pre"] < MIN_PRE:
            continue
        logcurve = {h: math.log(max(1.0, v)) for h, v in curve.items()}
        rows.append((name, feats, logcurve))
    return rows


def target_view(case):
    """Partial-shape features for the launch being forecast."""
    ds = iso_to_utc(case["day_start"])
    curve = day_curve(case.get("history_hourly"), ds)
    p1 = None
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers") is not None:
            p1 = float(row["viewers"])
            break
    if p1 is None:
        p1 = curve.get(1)
    feats = partial_features(case.get("history_hourly"), ds, curve, p1)
    if feats["lnp1"] is None:
        return None
    if feats["lnpre48"] is None:
        # No pre-release history at all: fall back to the library's
        # typical jump so the level distance still has something to use.
        feats["lnpre48"] = feats["lnp1"] - FALLBACK_JUMP
        feats["jump"] = FALLBACK_JUMP
    return feats


def _feature_scales(rows):
    """Population sd of the level features across the library (for scaling)."""
    out = {}
    for key in ("lnpre48", "jump"):
        vals = [f[key] for _, f, _ in rows if f.get(key) is not None]
        out[key] = st.pstdev(vals) if len(vals) > 1 and st.pstdev(vals) > 0 else 1.0
    return out


def _level_distance(a, b, scales):
    if a.get("lnpre48") is None or b.get("lnpre48") is None:
        return None
    if a.get("jump") is None or b.get("jump") is None:
        return None
    d_pre = (a["lnpre48"] - b["lnpre48"]) / scales["lnpre48"]
    d_jump = (a["jump"] - b["jump"]) / scales["jump"]
    return math.sqrt(d_pre * d_pre + d_jump * d_jump)


def _profile_distance(a, b):
    """Euclidean distance between demeaned diurnal profiles, or None."""
    pa, pb = a.get("prof"), b.get("prof")
    if pa is None or pb is None:
        return None
    d2, n = 0.0, 0
    for x, y in zip(pa, pb):
        if x is not None and y is not None:
            d2 += (x - y) ** 2
            n += 1
    if n < 12:
        return None
    return math.sqrt(d2 / n) * math.sqrt(24)


def _match_distance(kind, a, b, scales):
    d_level = _level_distance(a, b, scales)
    if d_level is None:
        return None
    if kind == "level":
        return d_level
    d_prof = _profile_distance(a, b)
    if d_prof is None:
        return d_level  # profile unavailable: fall back to level only
    return d_level + d_prof / PROFILE_DIVISOR


def _iqr_sigma(vals):
    """Robust log-space sd from the interquartile range."""
    if len(vals) < 4:
        return 0.5
    s = sorted(vals)
    return (s[int(0.75 * (len(s) - 1))] - s[int(0.25 * (len(s) - 1))]) / 1.349


def neighbour_mix(train, feats, kind, k, scales):
    """Neighbour ensemble: per hour (mu, sigma) in log space, or None.

    mu mixes the neighbours' log levels with their log ratios to their
    own hour 1; the ratio weight fades from 1 at hour 2 to 0 at hour 18.
    """
    scored = []
    for name, f, logcurve in train:
        d = _match_distance(kind, feats, f, scales)
        if d is not None:
            scored.append((d, logcurve))
    if len(scored) < 2:
        return None
    scored.sort(key=lambda t: t[0])
    nb = scored[:k]
    dmed = st.median([d for d, _ in nb]) or 1.0
    weights = [1.0 / (0.25 * dmed + d + 1e-9) for d, _ in nb]
    out = {}
    for h in HOURS:
        lv, rv, wv = [], [], []
        for w, (_, logcurve) in zip(weights, nb):
            if h in logcurve and 1 in logcurve:
                lv.append(logcurve[h])
                rv.append(logcurve[h] - logcurve[1])
                wv.append(w)
        if not lv:
            return None
        wsum = sum(wv)
        mu_level = sum(v * w for v, w in zip(lv, wv)) / wsum
        mu_ratio = feats["lnp1"] + sum(v * w for v, w in zip(rv, wv)) / wsum
        wr = max(0.0, min(1.0, (18 - h) / 16.0))
        mu = wr * mu_ratio + (1.0 - wr) * mu_level
        global_lv = [lc[h] for _, _, lc in train if h in lc]
        sd = 0.7 * _iqr_sigma(lv) + 0.3 * _iqr_sigma(global_lv)
        out[h] = (mu, max(SPREAD_FLOOR_NN, sd))
    return out


def _ols(X, y, ridge=1e-6):
    """Least squares via normal equations (pure python, tiny matrices)."""
    n, p = len(X), len(X[0])
    A = [[sum(X[k][i] * X[k][j] for k in range(n)) + (ridge if i == j else 0.0)
          for j in range(p)] for i in range(p)]
    b = [sum(X[k][i] * y[k] for k in range(n)) for i in range(p)]
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            continue
        for r in range(p):
            if r != col:
                fac = M[r][col] / M[col][col]
                for c in range(col, p + 1):
                    M[r][c] -= fac * M[col][c]
    return [M[i][p] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(p)]


def regression_forecast(train, feats):
    """Per-hour OLS of log viewers on (log hour-1, log pre-release median).

    Returns per hour (mu, sigma) in log space, or None if the target
    lacks the needed features or the library is too small.
    """
    if feats.get("lnp1") is None or feats.get("lnpre48") is None:
        return None
    out = {}
    for h in HOURS:
        X, y = [], []
        for _, f, logcurve in train:
            if h in logcurve and f.get("lnp1") is not None and f.get("lnpre48") is not None:
                X.append([1.0, f["lnp1"], f["lnpre48"]])
                y.append(logcurve[h])
        if len(X) < 8:
            return None
        beta = _ols(X, y)
        resid = [yy - sum(b * x for b, x in zip(beta, xx)) for xx, yy in zip(X, y)]
        sd = math.sqrt(sum(r * r for r in resid) / max(1, len(X) - 3))
        mu = sum(b * x for b, x in zip(beta, [1.0, feats["lnp1"], feats["lnpre48"]]))
        out[h] = (mu, max(SPREAD_FLOOR_REG, sd))
    return out


def median_curve_fallback(train, feats):
    """Fallback: whole-library median curve, ratio-early / level-late."""
    out = {}
    for h in HOURS:
        lv = [lc[h] for _, _, lc in train if h in lc]
        rv = [lc[h] - lc[1] for _, _, lc in train if h in lc and 1 in lc]
        if not lv or not rv:
            return None
        wr = max(0.0, min(1.0, (18 - h) / 16.0))
        mu = wr * (feats["lnp1"] + st.median(rv)) + (1.0 - wr) * st.median(lv)
        out[h] = (mu, max(SPREAD_FLOOR_NN, _iqr_sigma(lv)))
    return out


def blend_forecast(train, feats):
    """Blend neighbour ensembles + regression for a given library/target.

    Returns {hour: [q05, q25, q50, q75, q95]} for hours 2..24.
    """
    scales = _feature_scales(train)
    nn_a = neighbour_mix(train, feats, "level", N_LEVEL, scales)
    nn_b = neighbour_mix(train, feats, "profile", N_PROFILE, scales)
    reg = regression_forecast(train, feats)
    glob = median_curve_fallback(train, feats)
    out = {}
    for h in HOURS:
        parts_mu, parts_sd = [], []
        for nn in (nn_a, nn_b):
            if nn is not None and h in nn:
                parts_mu.append(nn[h][0])
                parts_sd.append(nn[h][1])
        mu_nn = sum(parts_mu) / len(parts_mu) if parts_mu else None
        sd_nn = sum(parts_sd) / len(parts_sd) if parts_sd else None
        w_reg = REG_W_EARLY if h <= 4 else REG_W
        if mu_nn is not None and reg is not None and h in reg:
            mu = (1.0 - w_reg) * mu_nn + w_reg * reg[h][0]
            sd = SPREAD_MULT * 0.5 * (sd_nn + reg[h][1])
        elif mu_nn is not None:
            mu, sd = mu_nn, SPREAD_MULT * sd_nn
        elif reg is not None and h in reg:
            mu, sd = reg[h][0], SPREAD_MULT * reg[h][1]
        elif glob is not None and h in glob:
            mu, sd = glob[h][0], SPREAD_MULT * glob[h][1]
        else:
            # Last resort: persistence at the hour-1 print, widening spread.
            mu = feats["lnp1"]
            sd = 0.15 * math.sqrt(h - 1)
        qs = []
        prev = 0.0
        for z in Z:
            q = math.exp(mu + sd * z)
            q = min(max(q, 0.01), 1e9)
            q = round(q, 2)
            if q <= prev:
                q = prev  # keep quantiles non-decreasing after rounding
            qs.append(q)
            prev = q
        out[h] = qs
    return out


def forecast_case(case):
    """Forecast one case: {hour: [q05, q25, q50, q75, q95]} for hours 2..24."""
    train = library_rows(case)
    feats = target_view(case)
    if not train or feats is None:
        raise ValueError("no usable library or no hour-1 print")
    return blend_forecast(train, feats)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    if not names:
        raise SystemExit(f"no .in case files found in {in_dir}")
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        fc = forecast_case(case)
        payload = {"hours": [{"hour": h, "q": fc[h]} for h in HOURS]}
        stem = name[:-3]  # strip ".in"
        with open(os.path.join(out_dir, stem + ".out"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=None, separators=(",", ":"))
            fh.write("\n")


if __name__ == "__main__":
    main()
