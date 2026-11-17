"""Direct pinball-loss quantile regression for gta6-twitch-ccu-forecast-01.

Forecasts Twitch viewers in hours 2..24 of a launch day as five quantiles,
from the hour-1 print. Python 3.13, standard library only.

THE MODEL
---------
For each forecast hour h, look at every comparable launch i and form the
log-ratio r_i = ln(v_h,i / v_1,i) of that hour's viewers to its own hour-1
print, against its log size z_i = ln(v_1,i).  For each level tau, fit the
conditional quantile of the log-ratio directly, by minimising pinball loss:

    q_tau(r | z) = alpha_tau + beta1_tau * u + beta2_tau * v

where u is log hour-1 size standardised, and v is the standardised square
of u: the size effect bends -- small launches explode through the day while
the biggest stay near flat -- and a quadratic catches that bend where a
straight line would keep falling past the biggest launches seen.  The fit
is plain gradient descent on the pinball loss (subgradient -tau when the
point is above the line, 1-tau when below), with the intercept re-fit
exactly at every step as the tau-quantile of the residuals, a decaying step
size, and a tiny ridge on the slopes for stability.  No kernel weighting,
no empirical quantiles, no assumed distribution, no hierarchical shrinkage:
the quantiles come straight out of the loss the contest scores.  The
target's standardised size is clamped to the training range, so the curve
never extrapolates the line past the biggest launch seen.

The forecast for hour h is then p1 * exp(q_tau), p1 the case's hour-1
print.  The five quantiles of each hour are sorted ascending (monotone
rearrangement, which can only lower pinball loss), rounded to 2 decimals
half up, and clamped to (0, 1e9].

FALLBACKS
---------
A comparable is used for hour h only if it has a positive hour-1 print and
a positive hour-h print (a print below 1 counts as 1, as the scorer does).
If fewer than 8 comparables qualify for an hour, that hour falls back to the
house persistence forecast (q_tau = p1 * exp(z_tau * 0.15 * sqrt(h - 1))).

PROGRAM CONTRACT (same as the kit starter): read every <id>.in in the input
directory (the grader runs one case at a time), write <id>.out to the output
directory: {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]},
exactly 23 rows, hours 2..24 each once, each q five finite numbers above 0
and at most 1e9, non-decreasing, at most 65,536 bytes. Deterministic: fixed
iteration counts, fixed step schedule, no randomness, no wall-clock use.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path: put it back
# first, so a module shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]  # standard normal quantiles, house fallback
SIGMA = 0.15  # house fallback spread, log units per square-root hour

# Pinball-regression fitting constants (fixed => deterministic).
ITERATIONS = 200          # gradient steps per (hour, level)
STEP0 = 0.8               # initial step size; decays as STEP0 / (1 + t / 40)
RIDGE = 0.005             # tiny L2 pull on the slope, stability only
MIN_POINTS = 8            # comparables needed for an hour, else house fallback


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _quantile(sorted_vals: list[float], tau: float) -> float:
    """tau-quantile of an already-sorted list, linear interpolation."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _fit_level(
    rs: list[float], feats: list[list[float]], tau: float
) -> tuple[float, list[float]]:
    """Pinball regression of r on the feature rows for one level.

    q = alpha + sum_j beta_j * feat_j.  Alternating exact-intercept /
    gradient-slope steps.  Deterministic.
    """
    n = len(rs)
    nfeat = len(feats[0]) if feats else 0
    betas = [0.0] * nfeat
    alpha = _quantile(sorted(rs), tau)
    for t in range(ITERATIONS):
        # Exact intercept: tau-quantile of the residuals at current slopes.
        resid = sorted(
            r - sum(b * f for b, f in zip(betas, fs)) for r, fs in zip(rs, feats)
        )
        alpha = _quantile(resid, tau)
        # Subgradients of the mean pinball loss w.r.t. each slope (+ ridge).
        grads = [0.0] * nfeat
        for r, fs in zip(rs, feats):
            pred = alpha + sum(b * f for b, f in zip(betas, fs))
            g = -tau if r >= pred else (1.0 - tau)
            for j in range(nfeat):
                grads[j] += g * fs[j]
        step = STEP0 / (1.0 + t / 40.0)
        for j in range(nfeat):
            betas[j] -= step * (grads[j] / n + 2.0 * RIDGE * betas[j])
    resid = sorted(
        r - sum(b * f for b, f in zip(betas, fs)) for r, fs in zip(rs, feats)
    )
    alpha = _quantile(resid, tau)
    return alpha, betas


def _comparable_points(comp: dict) -> dict[str, dict]:
    """A comparable's hourly points keyed by their `from` timestamp."""
    return {p["from"]: p for p in comp.get("hourly", [])}


def _iso_hour(day_start: str, h: int) -> str:
    """The `from` timestamp of hour h (1-based) of a day starting day_start."""
    from datetime import datetime, timedelta, timezone

    dt = datetime.strptime(day_start, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    return (dt + timedelta(hours=h - 1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _house_row(p1: float, h: int) -> list[float | int]:
    """The house persistence forecast for one hour (fallback)."""
    spread = SIGMA * math.sqrt(h - 1)
    return [round2(p1 * math.exp(z * spread)) for z in Z]


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)

    # Per comparable: its hour-1 print and its hour-h prints, keyed by hour.
    trains: list[tuple[float, dict[int, float]]] = []
    for comp in case.get("comparables", []):
        pts = _comparable_points(comp)
        h1 = pts.get(comp["day_start"])
        if h1 is None:
            continue
        v1 = h1.get("viewers")
        if v1 is None or v1 < 1:
            continue
        by_hour: dict[int, float] = {}
        for h in range(2, 25):
            p = pts.get(_iso_hour(comp["day_start"], h))
            if p is not None and p.get("viewers") is not None:
                by_hour[h] = max(1, p["viewers"])
        trains.append((float(v1), by_hour))

    hours_out: list[dict] = []
    for row in case["forecast"]:
        h = row["hour"]
        # Training pairs for this hour: (log-ratio, log-size).
        pairs = [
            (math.log(vh / v1), math.log(v1))
            for v1, by_hour in trains
            if (vh := by_hour.get(h)) is not None
        ]
        if len(pairs) < MIN_POINTS:
            hours_out.append({"hour": h, "q": _house_row(p1, h)})
            continue
        zs = [z for _, z in pairs]
        zbar = sum(zs) / len(zs)
        var = sum((z - zbar) ** 2 for z in zs) / len(zs)
        s = math.sqrt(var) if var > 0 else 1.0
        if s == 0:
            s = 1.0
        rs = [r for r, _ in pairs]
        xs = [(z - zbar) / s for z in zs]
        # Second feature: the standardised square of x (the bend).
        sq = [x * x for x in xs]
        sqbar = sum(sq) / len(sq)
        sqvar = sum((v - sqbar) ** 2 for v in sq) / len(sq)
        sqs = math.sqrt(sqvar) if sqvar > 0 else 1.0
        if sqs == 0:
            sqs = 1.0
        feats = [[x, (x * x - sqbar) / sqs] for x in xs]
        # Clamp the target's size to the training range: no extrapolation
        # past the biggest (or smallest) launch seen.
        x_new = (log_p1 - zbar) / s
        x_new = min(max(x_new, min(xs)), max(xs))
        feats_new = [x_new, (x_new * x_new - sqbar) / sqs]
        qs: list[float] = []
        for tau in LEVELS:
            alpha, betas = _fit_level(rs, feats, tau)
            qs.append(
                p1 * math.exp(alpha + sum(b * f for b, f in zip(betas, feats_new)))
            )
        # Monotone rearrangement, then round and clamp.
        qs.sort()
        hours_out.append(
            {"hour": h, "q": [round2(min(max(q, 0.01), 1e9)) for q in qs]}
        )
    return {"hours": hours_out}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
