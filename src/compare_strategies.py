#!/usr/bin/env python3
"""
Strategy Comparison Tool
Compares agent-driven trading strategies against buy-and-hold baseline.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore, Style, init
import questionary

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from backtester import Backtester
from main import run_hedge_fund
from tools.api import get_prices
from utils.analysts import ANALYST_ORDER, ANALYST_CONFIG

init(autoreset=True)


class BuyAndHoldSimulator:
    """Simulates buy-and-hold strategy for comparison."""
    
    def __init__(self, tickers: List[str], initial_capital: float):
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.portfolio_values = []
    
    def run_simulation(self, start_date: str, end_date: str) -> List[Dict]:
        """Run buy-and-hold simulation and return portfolio values over time."""
        print(f"\n{Fore.CYAN}📈 Running Buy-and-Hold Simulation{Style.RESET_ALL}")
        print(f"Tickers: {', '.join(self.tickers)}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        
        # Get date range
        dates = pd.date_range(start_date, end_date, freq="B")
        
        # Calculate equal weight allocation
        num_tickers = len(self.tickers)
        allocation_per_ticker = self.initial_capital / num_tickers
        
        # Fetch price data for all tickers
        price_data = {}
        for ticker in self.tickers:
            print(f"Fetching price data for {ticker}...")
            prices = get_prices(ticker, start_date, end_date)
            if not prices:
                raise ValueError(f"No price data found for {ticker}")
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'date': p.time,
                'close': p.close
            } for p in prices])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            price_data[ticker] = df
        
        # Calculate portfolio value for each date
        portfolio_values = []
        initial_prices = {}
        shares_held = {}
        
        for ticker in self.tickers:
            # Get first available price
            first_date = price_data[ticker].index[0]
            initial_price = price_data[ticker].loc[first_date, 'close']
            initial_prices[ticker] = initial_price
            shares_held[ticker] = allocation_per_ticker / initial_price
        
        print(f"Initial allocation: {allocation_per_ticker:,.2f} per ticker")
        for ticker in self.tickers:
            print(f"  {ticker}: {shares_held[ticker]:.2f} shares @ ${initial_prices[ticker]:.2f}")
        
        # Calculate portfolio value for each trading day
        for date in dates:
            # Check if any ticker has data for this date
            has_data = False
            total_value = 0
            
            for ticker in self.tickers:
                if date in price_data[ticker].index:
                    current_price = price_data[ticker].loc[date, 'close']
                    total_value += shares_held[ticker] * current_price
                    has_data = True
            
            if has_data:
                portfolio_values.append({
                    'Date': date,
                    'Portfolio Value': total_value
                })
        
        self.portfolio_values = portfolio_values
        print(f"✅ Buy-and-hold simulation complete. {len(portfolio_values)} trading days.")
        return portfolio_values


def calculate_metrics(portfolio_values: List[Dict], initial_capital: float) -> Dict:
    """Calculate performance metrics from portfolio values."""
    if not portfolio_values:
        return {}
    
    df = pd.DataFrame(portfolio_values)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Calculate returns
    df['Returns'] = df['Portfolio Value'].pct_change()
    
    # Basic metrics
    final_value = df['Portfolio Value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    
    # Annualized metrics
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25
    cagr = (final_value / initial_capital) ** (1 / years) - 1
    
    # Risk metrics
    returns = df['Returns'].dropna()
    if len(returns) > 1:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        sortino_ratio = returns.mean() / returns[returns < 0].std() * np.sqrt(252) if returns[returns < 0].std() > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
    else:
        sharpe_ratio = sortino_ratio = max_drawdown = 0
    
    return {
        'Initial Capital': initial_capital,
        'Final Value': final_value,
        'Total Return (%)': total_return,
        'CAGR (%)': cagr * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Trading Days': len(portfolio_values)
    }


def create_comparison_plot(buy_hold_values: List[Dict], agent_values: List[Dict], 
                          output_dir: str, tickers: List[str], agents: List[str]):
    """Create comparison plot of both strategies."""
    print(f"\n{Fore.CYAN}📊 Creating comparison plot...{Style.RESET_ALL}")
    
    # Check if we have data
    if not buy_hold_values or not agent_values:
        print(f"{Fore.YELLOW}⚠️  No data to plot. Skipping plot creation.{Style.RESET_ALL}")
        return
    
    # Convert to DataFrames
    bh_df = pd.DataFrame(buy_hold_values)
    agent_df = pd.DataFrame(agent_values)
    
    if bh_df.empty or agent_df.empty:
        print(f"{Fore.YELLOW}⚠️  Empty dataframes. Skipping plot creation.{Style.RESET_ALL}")
        return
    
    bh_df['Date'] = pd.to_datetime(bh_df['Date'])
    agent_df['Date'] = pd.to_datetime(agent_df['Date'])
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    plt.plot(bh_df['Date'], bh_df['Portfolio Value'], 
             label='Buy & Hold', linewidth=2, color='blue')
    plt.plot(agent_df['Date'], agent_df['Portfolio Value'], 
             label=f'Agent Strategy ({", ".join(agents)})', linewidth=2, color='red')
    
    plt.title(f'Strategy Comparison: {", ".join(tickers)}', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'strategy_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plot saved to: {plot_path}")


def export_results(buy_hold_values: List[Dict], agent_values: List[Dict],
                  buy_hold_metrics: Dict, agent_metrics: Dict,
                  output_dir: str, tickers: List[str], agents: List[str]):
    """Export results to CSV files."""
    print(f"\n{Fore.CYAN}💾 Exporting results...{Style.RESET_ALL}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have data
    if not buy_hold_values or not agent_values:
        print(f"{Fore.YELLOW}⚠️  No data to export. Skipping CSV creation.{Style.RESET_ALL}")
        return
    
    # Export portfolio values
    bh_df = pd.DataFrame(buy_hold_values)
    agent_df = pd.DataFrame(agent_values)
    
    if bh_df.empty or agent_df.empty:
        print(f"{Fore.YELLOW}⚠️  Empty dataframes. Skipping CSV creation.{Style.RESET_ALL}")
        return
    
    # Combine for comparison
    comparison_df = pd.DataFrame({
        'Date': bh_df['Date'],
        'Buy_Hold_Value': bh_df['Portfolio Value'],
        'Agent_Value': agent_df['Portfolio Value']
    })
    
    # Calculate relative performance
    comparison_df['Agent_vs_BH'] = (comparison_df['Agent_Value'] / comparison_df['Buy_Hold_Value'] - 1) * 100
    
    csv_path = os.path.join(output_dir, 'portfolio_comparison.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"✅ Portfolio values saved to: {csv_path}")
    
    # Export metrics
    metrics_df = pd.DataFrame({
        'Metric': list(buy_hold_metrics.keys()),
        'Buy_and_Hold': list(buy_hold_metrics.values()),
        'Agent_Strategy': list(agent_metrics.values())
    })
    
    metrics_path = os.path.join(output_dir, 'performance_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ Performance metrics saved to: {metrics_path}")
    
    # Print summary
    print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}STRATEGY COMPARISON SUMMARY{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Agents: {', '.join(agents)}")
    print(f"Period: {buy_hold_values[0]['Date']} to {buy_hold_values[-1]['Date']}")
    print()
    
    print(f"{Fore.BLUE}Buy & Hold:{Style.RESET_ALL}")
    print(f"  Final Value: ${buy_hold_metrics['Final Value']:,.2f}")
    print(f"  Total Return: {buy_hold_metrics['Total Return (%)']:.2f}%")
    print(f"  CAGR: {buy_hold_metrics['CAGR (%)']:.2f}%")
    print(f"  Sharpe Ratio: {buy_hold_metrics['Sharpe Ratio']:.3f}")
    print(f"  Max Drawdown: {buy_hold_metrics['Max Drawdown (%)']:.2f}%")
    print()
    
    print(f"{Fore.RED}Agent Strategy:{Style.RESET_ALL}")
    print(f"  Final Value: ${agent_metrics['Final Value']:,.2f}")
    print(f"  Total Return: {agent_metrics['Total Return (%)']:.2f}%")
    print(f"  CAGR: {agent_metrics['CAGR (%)']:.2f}%")
    print(f"  Sharpe Ratio: {agent_metrics['Sharpe Ratio']:.3f}")
    print(f"  Max Drawdown: {agent_metrics['Max Drawdown (%)']:.2f}%")
    print()
    
    # Calculate outperformance
    outperformance = agent_metrics['Total Return (%)'] - buy_hold_metrics['Total Return (%)']
    print(f"{Fore.YELLOW}Outperformance: {outperformance:+.2f} percentage points{Style.RESET_ALL}")


def main():
    parser = argparse.ArgumentParser(description="Compare agent trading strategies vs buy-and-hold")
    parser.add_argument("--tickers", type=str, required=True, 
                       help="Comma-separated list of stock tickers")
    parser.add_argument("--start-date", type=str, required=True,
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True,
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial-cash", type=float, default=100000.0,
                       help="Initial cash position (default: 100000)")
    parser.add_argument("--agents", type=str, required=False,
                       help="Comma-separated list of agent names (if not provided, will prompt for selection)")
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini",
                       help="LLM model name (default: gpt-4o-mini)")
    parser.add_argument("--model-provider", type=str, default="OpenAI",
                       help="LLM provider (default: OpenAI)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory (default: outputs)")
    parser.add_argument("--ollama", action="store_true",
                       help="Use Ollama for local LLM inference")
    
    args = parser.parse_args()
    
    # Parse inputs
    tickers = [t.strip() for t in args.tickers.split(',')]
    
    # Handle agent selection
    if args.agents:
        # Use provided agents
        agents = [a.strip() for a in args.agents.split(',')]
        
        # Convert display names to internal keys
        agent_keys = []
        for agent_name in agents:
            # Find the internal key for this display name
            found = False
            for key, config in ANALYST_CONFIG.items():
                if config["display_name"] == agent_name:
                    agent_keys.append(key)
                    found = True
                    break
            if not found:
                print(f"{Fore.RED}❌ Invalid agent: {agent_name}{Style.RESET_ALL}")
                print(f"Available agents: {', '.join([config['display_name'] for config in ANALYST_CONFIG.values()])}")
                sys.exit(1)
        
        agents = agent_keys  # Use internal keys for the backtester
    else:
        # Interactive agent selection
        print(f"{Fore.CYAN}🤖 Select AI Analysts for Strategy Comparison{Style.RESET_ALL}")
        print("Choose which analysts to include in your strategy comparison.")
        print()
        
        choices = questionary.checkbox(
            "Select your AI analysts (use Space to select/unselect):",
            choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
            instruction="\n\nInstructions: \n1. Press Space to select/unselect analysts.\n2. Press 'a' to select/unselect all.\n3. Press Enter when done to run the comparison.\n",
            validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
            style=questionary.Style(
                [
                    ("checkbox-selected", "fg:green"),
                    ("selected", "fg:green noinherit"),
                    ("highlighted", "noinherit"),
                    ("pointer", "noinherit"),
                ]
            ),
        ).ask()
        
        if not choices:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)
        else:
            agents = choices
            print(f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}\n")
    
    # Get display names for output
    agent_display_names = [ANALYST_CONFIG[key]["display_name"] for key in agents]
    
    print(f"{Fore.GREEN}🚀 Starting Strategy Comparison{Style.RESET_ALL}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Agents: {', '.join(agent_display_names)}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Initial Cash: ${args.initial_cash:,.2f}")
    print(f"Model: {args.model_name} ({args.model_provider})")
    
    # Run buy-and-hold simulation
    bh_simulator = BuyAndHoldSimulator(tickers, args.initial_cash)
    buy_hold_values = bh_simulator.run_simulation(args.start_date, args.end_date)
    
    # Run agent-driven backtest
    print(f"\n{Fore.CYAN}🤖 Running Agent-Driven Backtest{Style.RESET_ALL}")
    backtester = Backtester(
        agent=lambda **kwargs: run_hedge_fund(**kwargs),
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_cash,
        model_name=args.model_name,
        model_provider=args.model_provider,
        selected_analysts=agents
    )
    
    # Run the backtest (this will take time)
    backtester.run_backtest()
    agent_values = backtester.portfolio_values
    
    # Calculate metrics
    buy_hold_metrics = calculate_metrics(buy_hold_values, args.initial_cash)
    agent_metrics = calculate_metrics(agent_values, args.initial_cash)
    
    # Create outputs
    create_comparison_plot(buy_hold_values, agent_values, args.output_dir, tickers, agent_display_names)
    export_results(buy_hold_values, agent_values, buy_hold_metrics, agent_metrics,
                  args.output_dir, tickers, agent_display_names)
    
    print(f"\n{Fore.GREEN}✅ Strategy comparison complete!{Style.RESET_ALL}")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
