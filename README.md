# How Efficient Is the NFL Betting Market?

**A statistical study of 7,276 NFL games (1999–2025) testing whether real Vegas
closing lines can be beaten** — pure Python, built from scratch: vig-removal
math, a walk-forward Elo rating model, weak-form efficiency tests, a
backtesting engine with fractional-Kelly staking, and bootstrap confidence
intervals on every result.

---

## Headline findings

| Question | Answer |
|---|---|
| Is the closing line calibrated? | Almost perfectly. De-vigged probabilities track observed win rates within binomial error across all 10 buckets (Brier **0.210** vs 0.250 for a coin flip). |
| Is the point spread an unbiased forecast? | Yes: regressing actual margin on the spread gives **α = −0.003 (p = 0.99)** and **β = 1.043 (p vs 1 = 0.10)** — statistically indistinguishable from the efficient-market prediction (0, 1). |
| Do famous "angles" still pay? | No. Home dogs, divisional dogs, rest advantages, big favorites ATS — every angle's ROI confidence interval straddles zero at real listed prices. |
| Does the favorite–longshot bias exist? | Directionally yes: blindly backing longshots lost **−12.0%** out of sample while heavy favorites lost only **−1.5%** — but neither side is a profit. |
| Can a from-scratch Elo model beat the market? | No. The closing line beats Elo on Brier and log loss, and the Elo-edge strategy returned **−6.2% [−16.5%, +4.4%]** on untouched holdout seasons. |
| What does random betting return? | **−5.1%**, with a CI excluding zero — almost exactly the average overround (2.8% per side). The house edge, measured empirically. |

**The intellectually honest conclusion:** a public dataset plus a sensible
model is not enough to beat the NFL closing line. The negative result *is*
the result — and it precisely replicates the efficient-market hypothesis.

## What makes this rigorous

- Real data - Every completed NFL game since 1999 with actual closing
  spreads, totals, and moneylines (nflverse `games.csv`), auto-downloaded and
  cached.
- Proper vig removal - Two de-vigging methods implemented from the math up:
  proportional normalization and the **power method** (solve Σpᵢᵏ = 1 by
  bisection), which correctly shades longshot prices harder.
- No lookahead, anywhere - The Elo model walks forward chronologically
  (K = 20, home-field = 52 rating points, log margin-of-victory damping with
  autocorrelation correction, ⅓ offseason mean reversion, franchise-relocation
  handling).
- Train/test discipline - The one tunable parameter (Elo-vs-market edge
  threshold) is grid-searched on 2007–2017 **only**, then frozen and evaluated
  on 2018–2025.
- Uncertainty on everything - 10,000-resample bootstrap CIs on every ROI,
  Wilson intervals on every proportion, t-tests and exact binomial tests,
  plus an explicit multiple-comparisons warning.
- A control group - A seeded random-betting strategy measures the house
  edge empirically — the benchmark every "system" must be judged against.
- Engineering hygiene - Modular package, type hints, dataclasses, logging,
  CLI with argparse, deterministic seeds, and a **20-test pytest suite**
  covering odds math, Elo updates, settlement arithmetic, Kelly sizing, and
  drawdown logic.

## Project structure

```
nfl-market-efficiency/
├── run_analysis.py            # CLI: full pipeline in ~40s → output/REPORT.md
├── src/
│   ├── config.py              # all constants & hyperparameters in one place
│   ├── data/loader.py         # download, cache, clean, derive market columns
│   ├── markets/odds.py        # conversions, de-vig (proportional + power), Kelly
│   ├── models/elo.py          # walk-forward Elo with MOV damping & reversion
│   ├── analysis/
│   │   ├── calibration.py     # Brier, log loss, reliability tables
│   │   ├── efficiency.py      # spread OLS (α=0, β=1 test), cover rates
│   │   └── biases.py          # FLB buckets + ATS angle battery at real prices
│   ├── backtest/
│   │   ├── engine.py          # bet settlement, flat & fractional-Kelly bankrolls
│   │   ├── strategies.py      # Elo-edge, home dogs, favorites, longshots, control
│   │   └── stats.py           # bootstrap CIs, Wilson, drawdown, hypothesis tests
│   ├── viz/plots.py           # six publication-style figures
│   └── report.py              # auto-generates the Markdown research report
├── tests/                     # 20 unit tests (pytest)
├── output/                    # REPORT.md + figures/ (generated)
└── data/games.csv             # cached dataset
```

## Quickstart

```bash
pip install -r requirements.txt
python run_analysis.py            # full study (downloads data on first run)
python -m pytest tests/ -v        # run the test suite
python run_analysis.py --refresh --dev-end 2016 --n-boot 5000   # options
```

Outputs: `output/REPORT.md` (the full write-up with tables) and six figures in
`output/figures/` — calibration curve, favorite–longshot chart, spread
efficiency scatter, Elo history, holdout bankroll curves, and ROI confidence
intervals.

## The math

- **Implied probability:** American odds A → p = 100/(A+100) if A>0 else |A|/(|A|+100)
- **Power de-vig:** find k ≥ 1 with Σ pᵢᵏ = 1 (bisection; unique root since f is monotone)
- **Elo win prob:** P(home) = 1 / (1 + 10^(−(Rₕ + HFA − Rₐ)/400))
- **MOV multiplier:** ln(|margin|+1) · 2.2/(0.001·ΔElo_winner + 2.2)
- **Kelly fraction:** f* = (pb − q)/b with b = decimal − 1; staked at ¼-Kelly, capped at 5% of bankroll
- **Bootstrap ROI CI:** resample (profit, stake) pairs 10,000×; percentile interval on Σprofit/Σstake

## Limitations & roadmap

- Closing lines only — no openers or line-shopping across books, where real
  edges are likelier to live (closing-line-value analysis is the natural sequel).
- Elo ignores injuries, QB status, and weather; a gradient-boosted model with
  richer features is the obvious next model class.
- One league. Replicating on thinner markets (college totals) would map
  *where* efficiency breaks down.

## Data source & credit

Game and line data from [nflverse/nfldata](https://github.com/nflverse/nfldata)
(Lee Sharpe's `games.csv`). Elo design follows the well-known
FiveThirtyEight NFL Elo methodology, reimplemented from scratch.
