#!/usr/bin/env python3
"""
Enhanced Agent Comparison Script

Compare each agent strategy vs buy-and-hold with comprehensive metrics:
- Cash Balance, Total Position Value, Total Value
- Return %, Sharpe Ratio, Sortino Ratio, Max Drawdown
- Error logging for failed agents
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

init(autoreset=True)


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


def create_sample_data():
    """Create sample portfolio data for testing."""
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')
    np.random.seed(42)
    
    # Simulate buy-and-hold with some volatility
    returns = np.random.normal(0.0008, 0.02, len(dates))  # ~20% annual return, 2% daily vol
    prices = 100000 * (1 + returns).cumprod()
    
    return [{"Date": d, "Portfolio Value": p} for d, p in zip(dates, prices)]


def main():
    parser = argparse.ArgumentParser(description="Enhanced agent comparison with comprehensive metrics")
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,NVDA", help="Comma-separated tickers")
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--output-dir", type=str, default="outputs/enhanced_agent_comparison")
    parser.add_argument("--test-mode", action="store_true", help="Run with sample data for testing")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    end_date = args.end_date
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=12)).strftime("%Y-%m-%d")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{Fore.GREEN}📊 Enhanced Agent Comparison{Style.RESET_ALL}")
    print(f"   📅 Period: {start_date} to {end_date}")
    print(f"   💰 Initial Capital: ${args.initial_cash:,.0f}")
    print(f"   📈 Tickers: {', '.join(tickers)}")
    print()

    # Create sample data for testing
    if args.test_mode:
        print(f"{Fore.YELLOW}🧪 Running in test mode with sample data{Style.RESET_ALL}")
        
        # Buy-and-hold baseline
        bh_values = create_sample_data()
        bh_metrics = calculate_comprehensive_metrics(bh_values, args.initial_cash)
        
        # Simulate different agent performances
        agents = [
            {"name": "Ben Graham", "outperformance": 2.5, "volatility": 0.015},
            {"name": "Warren Buffett", "outperformance": 1.8, "volatility": 0.012},
            {"name": "Peter Lynch", "outperformance": 3.2, "volatility": 0.018},
            {"name": "Technical Analyst", "outperformance": -1.5, "volatility": 0.025},
            {"name": "Failed Agent", "outperformance": 0, "volatility": 0, "error": True}
        ]
        
        rows = []
        skipped_agents = []
        
        for agent in agents:
            if agent.get("error"):
                # Simulate failed agent
                skipped_agents.append({
                    "Agent": agent["name"],
                    "Error": "Simulated API timeout error",
                    "Status": "Failed"
                })
                
                rows.append({
                    "Agent": agent["name"],
                    "Cash_Balance": args.initial_cash,
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
                    "Status": "Failed: Simulated API timeout error"
                })
            else:
                # Simulate successful agent
                agent_return = bh_metrics["return_pct"] + agent["outperformance"]
                agent_sharpe = bh_metrics["sharpe_ratio"] + (agent["outperformance"] / 10)
                
                rows.append({
                    "Agent": agent["name"],
                    "Cash_Balance": args.initial_cash * (1 + agent_return/100),
                    "Total_Position_Value": args.initial_cash * (1 + agent_return/100),
                    "Total_Value": args.initial_cash * (1 + agent_return/100),
                    "Return_%": agent_return,
                    "Sharpe_Ratio": agent_sharpe,
                    "Sortino_Ratio": agent_sharpe * 1.1,
                    "Max_Drawdown_%": bh_metrics["max_drawdown_pct"] + agent["volatility"] * 100,
                    "BuyHold_Total_Value": bh_metrics["total_value"],
                    "BuyHold_Return_%": bh_metrics["return_pct"],
                    "BuyHold_Sharpe_Ratio": bh_metrics["sharpe_ratio"],
                    "BuyHold_Sortino_Ratio": bh_metrics["sortino_ratio"],
                    "BuyHold_Max_Drawdown_%": bh_metrics["max_drawdown_pct"],
                    "Outperformance_%": agent["outperformance"],
                    "Status": "Success"
                })
        
        # Export results
        df = pd.DataFrame(rows).sort_values(by="Outperformance_%", ascending=False)
        out_csv = os.path.join(args.output_dir, "enhanced_agent_comparison.csv")
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
        
        print(f"\n{Fore.CYAN}📋 COMPREHENSIVE METRICS TABLE{Style.RESET_ALL}")
        print("=" * 120)
        
        # Display full metrics table
        full_display_df = df[["Agent", "Cash_Balance", "Total_Position_Value", "Total_Value", 
                              "Return_%", "Sharpe_Ratio", "Sortino_Ratio", "Max_Drawdown_%", "Status"]].copy()
        
        # Format currency columns
        for col in ["Cash_Balance", "Total_Position_Value", "Total_Value"]:
            full_display_df[col] = full_display_df[col].apply(lambda x: f"${x:,.0f}")
        
        # Format percentage columns
        for col in ["Return_%", "Max_Drawdown_%"]:
            full_display_df[col] = full_display_df[col].apply(lambda x: f"{x:+.2f}%")
        
        # Format ratio columns
        for col in ["Sharpe_Ratio", "Sortino_Ratio"]:
            full_display_df[col] = full_display_df[col].apply(lambda x: f"{x:.2f}" if x != float('inf') else "∞")
        
        # Rename columns for display
        full_display_df.columns = ["Agent", "Cash Balance", "Position Value", "Total Value", 
                                   "Return %", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown %", "Status"]
        
        print(full_display_df.to_string(index=False))
        
    else:
        print(f"{Fore.RED}❌ This script requires the full backtesting system.{Style.RESET_ALL}")
        print(f"   Use --test-mode to run with sample data")
        print(f"   Or run the original compare_each_agent.py with proper dependencies")


if __name__ == "__main__":
    main()
