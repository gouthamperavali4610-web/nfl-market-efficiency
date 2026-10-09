#!/usr/bin/env python3
"""End-to-end pipeline: data -> Elo -> efficiency tests -> backtests -> report.

Usage:
    python run_analysis.py               # full study with cached data
    python run_analysis.py --refresh     # re-download latest data first
    python run_analysis.py --dev-end 2016 --n-boot 5000
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import pandas as pd

from src import config
from src.analysis.biases import ats_angles_table, favorite_longshot_table
from src.analysis.calibration import reliability_table, scoring_table
from src.analysis.efficiency import home_cover_summary, spread_regression
from src.backtest.engine import Backtester
from src.backtest.strategies import (
    all_static_strategies,
    make_elo_edge_strategy,
    split_dev_holdout,
    tune_elo_threshold,
)
from src.data.loader import load_games, moneyline_era
from src.models.elo import run_elo
from src.report import write_report
from src.viz import plots

log = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refresh", action="store_true", help="re-download the dataset")
    p.add_argument("--dev-end", type=int, default=config.DEV_END_SEASON,
                   help="last season of the development (tuning) period")
    p.add_argument("--n-boot", type=int, default=config.BACKTEST.n_bootstrap,
                   help="bootstrap resamples for confidence intervals")
    return p.parse_args()


def main() -> int:
    t0 = time.time()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()

    # 1. Data ---------------------------------------------------------------
    games = load_games(refresh=args.refresh)
    games = run_elo(games)
    ml = moneyline_era(games)
    log.info("Moneyline-era games: %d (%d+)", len(ml), config.MONEYLINE_ERA_START)

    # 2. Market quality -----------------------------------------------------
    scoring = scoring_table(ml)
    reliability = reliability_table(ml)
    reg = spread_regression(games)
    cover = home_cover_summary(games)
    flb = favorite_longshot_table(ml)
    ats = ats_angles_table(games)

    # 3. Strategies: dev/holdout protocol ------------------------------------
    backtester = Backtester()
    dev, holdout = split_dev_holdout(ml, args.dev_end)
    grid = tune_elo_threshold(dev, backtester)
    chosen = float(grid.loc[grid["dev_ROI"].idxmax(), "threshold"])
    log.info("Chosen Elo edge threshold (dev period): %.2f", chosen)

    holdout_results = {}
    elo_strategy = make_elo_edge_strategy(chosen)
    for staking in ("flat", "kelly"):
        res = backtester.run(f"elo_edge>={chosen:.2f}", elo_strategy(holdout), staking)
        holdout_results[f"elo_edge ({staking})"] = res
    for name, fn in all_static_strategies().items():
        holdout_results[name] = backtester.run(name, fn(holdout), "flat")

    holdout_table = pd.DataFrame([r.headline() for r in holdout_results.values()])
    print("\n=== HOLDOUT RESULTS (out of sample) ===")
    print(holdout_table.to_string(index=False))

    # 4. Figures -------------------------------------------------------------
    plots.plot_calibration(reliability)
    plots.plot_favorite_longshot(flb)
    plots.plot_spread_efficiency(games, reg)
    plots.plot_elo_history(games.attrs["elo_history"], ["KC", "NE", "BUF", "DAL"])
    flat_curves = {k: v for k, v in holdout_results.items() if v.staking == "flat"}
    plots.plot_bankrolls(flat_curves)
    plots.plot_roi_intervals(holdout_table)

    # 5. Report --------------------------------------------------------------
    report_path = write_report({
        "n_games": len(games), "season_min": int(games["season"].min()),
        "season_max": int(games["season"].max()),
        "avg_overround": float(ml["overround"].mean()),
        "scoring": scoring, "reg": reg, "cover": cover,
        "flb": flb, "ats": ats, "grid": grid,
        "chosen_threshold": chosen, "holdout_table": holdout_table,
    })
    log.info("Done in %.1fs — report: %s", time.time() - t0, report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
