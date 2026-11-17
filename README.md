# Direct pinball-loss quantile regression

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/koput04/gta6-twitch-forecast-3/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/koput04/gta6-twitch-forecast-3/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts Twitch viewers in hours 2–24 of Grand Theft Auto VI's launch day
(2026-11-19 UTC) as five quantiles, from the hour-1 print — an entry for
[make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour"
bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01), built by an AI
agent. Python 3.13, standard library only, deterministic.

## The model

For each forecast hour h, every comparable launch gives one observation: the
log-ratio `r = ln(v_h / v_1)` of that hour's viewers to its own hour-1 print,
against its log size `z = ln(v_1)`. For each quantile level tau, the
conditional quantile of the log-ratio is fit **directly, by minimising
pinball loss** — the same loss the contest scores:

```
q_tau(r | z) = alpha_tau + beta1_tau * u + beta2_tau * v
```

where `u` is standardised log hour-1 size and `v` is the standardised
square of `u`. The size effect bends: small launches explode through the
day while the biggest stay near flat, and the quadratic catches that bend
where a straight line would keep falling past the biggest launches seen.
The fit is gradient descent on the pinball loss with the intercept re-fit
exactly at every step (the tau-quantile of the residuals), a decaying step
size, and a tiny ridge on the slopes. No kernel weighting, no empirical
quantiles, no assumed distribution, no hierarchical shrinkage. The target's
standardised size is clamped to the training range, so the curve never
extrapolates past the biggest launch seen. Hours with fewer than 8 usable
comparables fall back to the house persistence forecast.

## Validation (leave-one-out over the kit's 37 usable comparables)

Mean pinball loss on the log scale, same protocol for every row:

| model | overall | big hour-1 (≥100k) | huge hour-1 (≥250k) |
| --- | --- | --- | --- |
| this entry | **0.3776** | **0.2052** | **0.1184** |
| house baseline | 0.9297 | 0.2611 | 0.2255 |

The house numbers reproduce the kit's own baseline exactly, so the
evaluation protocol is the same one the house is scored on. The biggest
wins are on the biggest launches — ELDEN RING 0.0621 (house 0.2108),
Cyberpunk 2077 0.1067 (0.2299), Elden Ring Nightreign 0.1125 (0.2516) —
which is what the final case (GTA VI, a huge launch) rewards. One honest
miss: Diablo IV 0.1799 vs house 0.1254 — its day is nearly perfectly flat,
and persistence is nearly unbeatable there.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out`:
`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`, 23 rows,
hours 2–24. Runs in ~1.4 s per case; identical output bytes across runs.

## Files

- `main.py` — the whole entry (model + I/O, standard library only)
- `LICENSE` — MIT

## License

MIT — see `LICENSE`.
