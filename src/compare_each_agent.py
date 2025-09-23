#!/usr/bin/env python3
"""
Compare each agent strategy vs buy-and-hold over the last 12 months.

- Runs each agent individually (respecting horizons via Backtester/run_hedge_fund)
- Compares final portfolio value vs equal-weight buy-and-hold baseline
- Exports CSV and prints a comprehensive comparison table with performance metrics
- Logs any agents that were skipped due to errors
"""

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional
import traceback
import numpy as np

import pandas as pd
from colorama import Fore, Style, init

# Local imports
sys.path.append(str(Path(__file__).parent.parent))

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


def calculate_comprehensive_metrics(values: List[Dict], initial_capital: float) -> Dict:
    """Calculate comprehensive performance metrics from portfolio values."""
    if not values:
        return {
            "cash_balance": initial_capital,
            "total_position_value": 0.0,
            "total_value": initial_capital,
            "return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "error": "No portfolio values"
        }
    
    df = pd.DataFrame(values)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Basic metrics
    final_value = float(df['Portfolio Value'].iloc[-1])
    total_return_pct = (final_value / initial_capital - 1.0) * 100.0
    
    # Calculate cash balance and position value (approximate from portfolio value)
    # For buy-and-hold, this is the total value
    cash_balance = final_value  # Simplified for buy-and-hold
    total_position_value = final_value
    
    # Calculate returns for risk metrics
    df['Returns'] = df['Portfolio Value'].pct_change()
    returns = df['Returns'].dropna()
    
    if len(returns) > 1:
        # Risk-free rate (assume 4.34% annually)
        risk_free_rate = 0.0434 / 252  # Daily risk-free rate
        excess_returns = returns - risk_free_rate
        
        # Sharpe ratio
        if returns.std() > 1e-12:
            sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / returns.std())
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 1e-12:
            sortino_ratio = np.sqrt(252) * (excess_returns.mean() / negative_returns.std())
        else:
            sortino_ratio = float('inf') if excess_returns.mean() > 0 else 0.0
        
        # Maximum drawdown
        rolling_max = df['Portfolio Value'].cummax()
        drawdown = (df['Portfolio Value'] - rolling_max) / rolling_max
        max_drawdown_pct = drawdown.min() * 100.0
    else:
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
        max_drawdown_pct = 0.0
    
    return {
        "cash_balance": cash_balance,
        "total_position_value": total_position_value,
        "total_value": final_value,
        "return_pct": total_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "error": None
    }


def calc_final_value(values: List[Dict], initial_capital: float) -> Dict:
    """Legacy function for backward compatibility."""
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
    print(f"{Fore.GREEN}📊 Running Buy-and-Hold baseline...{Style.RESET_ALL}")
    bh = BuyAndHoldSimulator(tickers, args.initial_cash)
    bh_values = bh.run(start_date, end_date)
    bh_metrics = calculate_comprehensive_metrics(bh_values, args.initial_cash)

    # Run each agent individually
    rows = []
    skipped_agents = []
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

        try:
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

            agent_metrics = calculate_comprehensive_metrics(agent_values, args.initial_cash)

            # Add comprehensive metrics to the row
            rows.append({
                "Agent": display,
                "Cash_Balance": agent_metrics["cash_balance"],
                "Total_Position_Value": agent_metrics["total_position_value"],
                "Total_Value": agent_metrics["total_value"],
                "Return_%": agent_metrics["return_pct"],
                "Sharpe_Ratio": agent_metrics["sharpe_ratio"],
                "Sortino_Ratio": agent_metrics["sortino_ratio"],
                "Max_Drawdown_%": agent_metrics["max_drawdown_pct"],
                "BuyHold_Total_Value": bh_metrics["total_value"],
                "BuyHold_Return_%": bh_metrics["return_pct"],
                "BuyHold_Sharpe_Ratio": bh_metrics["sharpe_ratio"],
                "BuyHold_Sortino_Ratio": bh_metrics["sortino_ratio"],
                "BuyHold_Max_Drawdown_%": bh_metrics["max_drawdown_pct"],
                "Outperformance_%": agent_metrics["return_pct"] - bh_metrics["return_pct"],
                "Status": "Success"
            })

            # Save per-agent series
            agent_df = pd.DataFrame(agent_values)
            agent_df.to_csv(os.path.join(args.output_dir, f"{agent_key}_equity.csv"), index=False)
            
            print(f"   ✅ {display}: {agent_metrics['return_pct']:.2f}% return, Sharpe: {agent_metrics['sharpe_ratio']:.2f}")

        except Exception as e:
            error_msg = f"Error running {display}: {str(e)}"
            print(f"   {Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
            
            # Log the full traceback for debugging
            print(f"   {Fore.RED}Full error:{Style.RESET_ALL}")
            traceback.print_exc()
            
            skipped_agents.append({
                "Agent": display,
                "Error": str(e),
                "Status": "Failed"
            })
            
            # Add failed row with error info
            rows.append({
                "Agent": display,
                "Cash_Balance": 0.0,
                "Total_Position_Value": 0.0,
                "Total_Value": args.initial_cash,
                "Return_%": 0.0,
                "Sharpe_Ratio": 0.0,
                "Sortino_Ratio": 0.0,
                "Max_Drawdown_%": 0.0,
                "BuyHold_Total_Value": bh_metrics["total_value"],
                "BuyHold_Return_%": bh_metrics["return_pct"],
                "BuyHold_Sharpe_Ratio": bh_metrics["sharpe_ratio"],
                "BuyHold_Sortino_Ratio": bh_metrics["sortino_ratio"],
                "BuyHold_Max_Drawdown_%": bh_metrics["max_drawdown_pct"],
                "Outperformance_%": -bh_metrics["return_pct"],
                "Status": f"Failed: {str(e)[:50]}..."
            })

    # Export summary
    df = pd.DataFrame(rows).sort_values(by="Outperformance_%", ascending=False)
    out_csv = os.path.join(args.output_dir, "each_agent_vs_buyhold.csv")
    df.to_csv(out_csv, index=False)

    # Export skipped agents log
    if skipped_agents:
        skipped_df = pd.DataFrame(skipped_agents)
        skipped_csv = os.path.join(args.output_dir, "skipped_agents.csv")
        skipped_df.to_csv(skipped_csv, index=False)
        print(f"\n{Fore.RED}⚠️  Skipped {len(skipped_agents)} agents due to errors. Log saved: {skipped_csv}{Style.RESET_ALL}")
        for agent in skipped_agents:
            print(f"   ❌ {agent['Agent']}: {agent['Error']}")

    print(f"\n{Fore.GREEN}📊 Comparison complete.{Style.RESET_ALL} Saved: {out_csv}")
    
    # Display comprehensive comparison table
    print(f"\n{Fore.CYAN}📈 AGENT vs BUY-AND-HOLD COMPARISON{Style.RESET_ALL}")
    print("=" * 120)
    
    # Format the table with proper alignment
    display_df = df[["Agent", "Return_%", "Sharpe_Ratio", "Sortino_Ratio", "Max_Drawdown_%", 
                     "BuyHold_Return_%", "BuyHold_Sharpe_Ratio", "Outperformance_%", "Status"]].copy()
    
    # Format numbers for display
    for col in ["Return_%", "BuyHold_Return_%", "Outperformance_%"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:+.2f}%")
    
    for col in ["Sharpe_Ratio", "BuyHold_Sharpe_Ratio"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
    
    for col in ["Sortino_Ratio"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if x != float('inf') else "∞")
    
    for col in ["Max_Drawdown_%"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
    
    # Rename columns for display
    display_df.columns = ["Agent", "Agent Return", "Agent Sharpe", "Agent Sortino", "Agent Max DD", 
                         "BH Return", "BH Sharpe", "Outperformance", "Status"]
    
    print(display_df.to_string(index=False))
    
    # Summary statistics
    successful_agents = df[df["Status"] == "Success"]
    if len(successful_agents) > 0:
        print(f"\n{Fore.GREEN}📊 SUMMARY STATISTICS{Style.RESET_ALL}")
        print(f"   ✅ Successful agents: {len(successful_agents)}")
        print(f"   ❌ Failed agents: {len(skipped_agents)}")
        print(f"   🏆 Best performer: {successful_agents.iloc[0]['Agent']} ({successful_agents.iloc[0]['Outperformance_%']:+.2f}%)")
        print(f"   📉 Worst performer: {successful_agents.iloc[-1]['Agent']} ({successful_agents.iloc[-1]['Outperformance_%']:+.2f}%)")
        print(f"   📈 Average outperformance: {successful_agents['Outperformance_%'].mean():+.2f}%")
        print(f"   🎯 Agents beating buy-and-hold: {len(successful_agents[successful_agents['Outperformance_%'] > 0])}")
    
    print(f"\n{Fore.BLUE}💡 Buy-and-Hold Baseline:{Style.RESET_ALL}")
    print(f"   💰 Total Value: ${bh_metrics['total_value']:,.0f}")
    print(f"   📈 Return: {bh_metrics['return_pct']:+.2f}%")
    print(f"   📊 Sharpe Ratio: {bh_metrics['sharpe_ratio']:.2f}")
    print(f"   📉 Max Drawdown: {bh_metrics['max_drawdown_pct']:.2f}%")


if __name__ == "__main__":
    main()


