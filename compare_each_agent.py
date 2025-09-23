#!/usr/bin/env python3
"""
Compare each agent strategy vs buy-and-hold over the last 12 months.

- Runs each agent individually (respecting horizons via Backtester/run_hedge_fund)
- Compares final portfolio value vs equal-weight buy-and-hold baseline
- Exports CSV and prints a concise summary table
"""

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd
from colorama import Fore, Style, init

# Local imports
sys.path.append(str(Path(__file__).parent))

from src.backtester import Backtester
from src.main import run_hedge_fund
from src.tools.api import get_prices
from src.utils.analysts import ANALYST_CONFIG

init(autoreset=True)


class BuyAndHoldSimulator:
    def __init__(self, tickers: List[str], initial_capital: float):
        self.tickers = tickers
        self.initial_capital = initial_capital

    def run(self, start_date: str, end_date: str) -> List[Dict]:
        dates = pd.date_range(start_date, end_date, freq="B")
        num_tickers = max(len(self.tickers), 1)
        allocation = self.initial_capital / num_tickers

        price_data = {}
        for t in self.tickers:
            prices = get_prices(t, start_date, end_date)
            df = pd.DataFrame([
                {"date": p.time, "close": p.close}
            for p in prices])
            df["date"] = pd.to_datetime(df["date"]) 
            df = df.set_index("date")
            price_data[t] = df

        # initial shares
        shares = {}
        for t in self.tickers:
            first_date = price_data[t].index[0]
            first_price = float(price_data[t].loc[first_date, "close"]) 
            shares[t] = allocation / first_price if first_price > 0 else 0

        values: List[Dict] = []
        for d in dates:
            total = 0.0
            has = False
            for t in self.tickers:
                if d in price_data[t].index:
                    total += shares[t] * float(price_data[t].loc[d, "close"]) 
                    has = True
            if has:
                values.append({"Date": d, "Portfolio Value": total})
        return values


def calc_final_value(values: List[Dict], initial_capital: float) -> Dict:
    if not values:
        return {"final": initial_capital, "total_return_pct": 0.0}
    final_val = float(values[-1]["Portfolio Value"]) 
    ret_pct = (final_val / float(initial_capital) - 1.0) * 100.0
    return {"final": final_val, "total_return_pct": ret_pct}


def main():
    parser = argparse.ArgumentParser(description="Compare each agent vs buy-and-hold over last 12 months")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated tickers")
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--output-dir", type=str, default="outputs/agent_vs_bh")
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini")
    parser.add_argument("--model-provider", type=str, default="OpenAI")
    parser.add_argument("--exclude-damodaran", action="store_true", 
                       help="Exclude Aswath Damodaran analysis (cost optimization)")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    end_date = args.end_date
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=12)).strftime("%Y-%m-%d")

    os.makedirs(args.output_dir, exist_ok=True)

    # Baseline once
    bh = BuyAndHoldSimulator(tickers, args.initial_cash)
    bh_values = bh.run(start_date, end_date)
    bh_metrics = calc_final_value(bh_values, args.initial_cash)

    # Run each agent individually
    rows = []
    agent_keys = list(ANALYST_CONFIG.keys())
    
    # Exclude Damodaran if requested (cost optimization)
    if args.exclude_damodaran:
        agent_keys = [key for key in agent_keys if key != "aswath_damodaran"]
        print(f"{Fore.YELLOW}⚠️  Excluding Aswath Damodaran analysis (cost optimization){Style.RESET_ALL}")
        print(f"   💰 Damodaran agent is 6-12x more expensive than other agents")
        print(f"   📊 Running {len(agent_keys)} agents instead of {len(ANALYST_CONFIG)}")
        print()
    
    for agent_key in agent_keys:
        display = ANALYST_CONFIG[agent_key]["display_name"]
        print(f"\n{Fore.CYAN}Running agent{Style.RESET_ALL}: {display}")

        backtester = Backtester(
            agent=lambda **kwargs: run_hedge_fund(**kwargs),
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.initial_cash,
            model_name=args.model_name,
            model_provider=args.model_provider,
            selected_analysts=[agent_key],
        )
        backtester.run_backtest()
        agent_values = backtester.portfolio_values

        agent_metrics = calc_final_value(agent_values, args.initial_cash)

        rows.append({
            "Agent": display,
            "BuyHold_Final": bh_metrics["final"],
            "Agent_Final": agent_metrics["final"],
            "Agent_vs_BH_pp": agent_metrics["total_return_pct"] - ((bh_metrics["final"] / args.initial_cash - 1) * 100),
            "Agent_Return_%": agent_metrics["total_return_pct"],
            "BH_Return_%": (bh_metrics["final"] / args.initial_cash - 1) * 100,
        })

        # Save per-agent series
        agent_df = pd.DataFrame(agent_values)
        agent_df.to_csv(os.path.join(args.output_dir, f"{agent_key}_equity.csv"), index=False)

    # Export summary
    df = pd.DataFrame(rows).sort_values(by="Agent_vs_BH_pp", ascending=False)
    out_csv = os.path.join(args.output_dir, "each_agent_vs_buyhold.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n{Fore.GREEN}Comparison complete.{Style.RESET_ALL} Saved: {out_csv}")
    print(df.to_string(index=False, formatters={
        "BuyHold_Final": lambda v: f"${v:,.0f}",
        "Agent_Final": lambda v: f"${v:,.0f}",
        "Agent_vs_BH_pp": lambda v: f"{v:+.2f}",
        "Agent_Return_%": lambda v: f"{v:.2f}",
        "BH_Return_%": lambda v: f"{v:.2f}",
    }))


if __name__ == "__main__":
    main()


