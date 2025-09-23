from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
import pandas as pd
import numpy as np
import json
from src.utils.api_key import get_api_key_from_state
from src.tools.api import get_insider_trades, get_company_news


##### Sentiment Agent #####
def sentiment_analyst_agent(state: AgentState, agent_id: str = "sentiment_analyst_agent"):
    """Analyzes market sentiment and generates trading signals for multiple tickers."""
    import time
    from datetime import datetime
    
    start_time = time.time()
    max_execution_time = 15  # seconds per ticker
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    # Diagnostic logging
    print(f"🔍 {agent_id} DIAGNOSTICS:")
    print(f"   ⏱️  Timeout limit: {max_execution_time}s per ticker")
    print(f"   📊 Total tickers: {len(tickers)}")
    print(f"   🎯 Expected total time: {max_execution_time * len(tickers)}s")
    print(f"   🤖 LLM provider: {state.get('model_provider', 'Unknown')}")
    print(f"   🧠 LLM model: {state.get('model_name', 'Unknown')}")
    print()
    # Initialize sentiment analysis for each ticker
    sentiment_analysis = {}

    for ticker in tickers:
        ticker_start = time.time()
        
        # Check execution time limit
        elapsed_time = time.time() - start_time
        if elapsed_time > max_execution_time:
            print(f"🚨 CRITICAL TIMEOUT: {agent_id} exceeded {max_execution_time}s limit")
            print(f"   ⏱️  Elapsed time: {elapsed_time:.2f}s")
            print(f"   📊 Processed tickers: {len(sentiment_analysis)}")
            print(f"   ⚠️  Skipping remaining tickers - SENTIMENT ANALYSIS INCOMPLETE!")
            print(f"   📰 Missing sentiment analysis for: {', '.join(tickers[len(sentiment_analysis):])}")
            print(f"   🔧 Consider: Increasing timeout, reducing tickers, or optimizing data fetching")
            break
            
        progress.update_status(agent_id, ticker, "Fetching insider trades")

        # Step 1: Insider trades
        insider_start = time.time()
        try:
            insider_trades = get_insider_trades(
                ticker=ticker,
                end_date=end_date,
                limit=1000,
                api_key=api_key,
            )
            insider_time = time.time() - insider_start
            print(f"   👥 Insider trades: {insider_time:.2f}s")
        except Exception as e:
            print(f"⚠️  Error fetching insider trades for {ticker}: {e}")
            progress.update_status(agent_id, ticker, "Error fetching insider trades")
            insider_trades = []
            insider_time = time.time() - insider_start

        progress.update_status(agent_id, ticker, "Analyzing trading patterns")

        # Get the signals from the insider trades
        transaction_shares = pd.Series([t.transaction_shares for t in insider_trades]).dropna()
        insider_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()

        progress.update_status(agent_id, ticker, "Fetching company news")

        # Step 2: Company news
        news_start = time.time()
        try:
            company_news = get_company_news(ticker, end_date, limit=100, api_key=api_key)
            news_time = time.time() - news_start
            print(f"   📰 Company news: {news_time:.2f}s")
        except Exception as e:
            print(f"⚠️  Error fetching company news for {ticker}: {e}")
            progress.update_status(agent_id, ticker, "Error fetching news")
            company_news = []
            news_time = time.time() - news_start

        # Get the sentiment from the company news
        sentiment = pd.Series([n.sentiment for n in company_news]).dropna()
        news_signals = np.where(sentiment == "negative", "bearish", 
                              np.where(sentiment == "positive", "bullish", "neutral")).tolist()
        
        progress.update_status(agent_id, ticker, "Combining signals")
        # Combine signals from both sources with weights
        insider_weight = 0.3
        news_weight = 0.7
        
        # Calculate weighted signal counts
        bullish_signals = (
            insider_signals.count("bullish") * insider_weight +
            news_signals.count("bullish") * news_weight
        )
        bearish_signals = (
            insider_signals.count("bearish") * insider_weight +
            news_signals.count("bearish") * news_weight
        )

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level based on the weighted proportion
        total_weighted_signals = len(insider_signals) * insider_weight + len(news_signals) * news_weight
        confidence = 0  # Default confidence when there are no signals
        if total_weighted_signals > 0:
            confidence = round((max(bullish_signals, bearish_signals) / total_weighted_signals) * 100, 2)
        
        # Create structured reasoning similar to technical analysis
        reasoning = {
            "insider_trading": {
                "signal": "bullish" if insider_signals.count("bullish") > insider_signals.count("bearish") else 
                         "bearish" if insider_signals.count("bearish") > insider_signals.count("bullish") else "neutral",
                "confidence": round((max(insider_signals.count("bullish"), insider_signals.count("bearish")) / max(len(insider_signals), 1)) * 100),
                "metrics": {
                    "total_trades": len(insider_signals),
                    "bullish_trades": insider_signals.count("bullish"),
                    "bearish_trades": insider_signals.count("bearish"),
                    "weight": insider_weight,
                    "weighted_bullish": round(insider_signals.count("bullish") * insider_weight, 1),
                    "weighted_bearish": round(insider_signals.count("bearish") * insider_weight, 1),
                }
            },
            "news_sentiment": {
                "signal": "bullish" if news_signals.count("bullish") > news_signals.count("bearish") else 
                         "bearish" if news_signals.count("bearish") > news_signals.count("bullish") else "neutral",
                "confidence": round((max(news_signals.count("bullish"), news_signals.count("bearish")) / max(len(news_signals), 1)) * 100),
                "metrics": {
                    "total_articles": len(news_signals),
                    "bullish_articles": news_signals.count("bullish"),
                    "bearish_articles": news_signals.count("bearish"),
                    "neutral_articles": news_signals.count("neutral"),
                    "weight": news_weight,
                    "weighted_bullish": round(news_signals.count("bullish") * news_weight, 1),
                    "weighted_bearish": round(news_signals.count("bearish") * news_weight, 1),
                }
            },
            "combined_analysis": {
                "total_weighted_bullish": round(bullish_signals, 1),
                "total_weighted_bearish": round(bearish_signals, 1),
                "signal_determination": f"{'Bullish' if bullish_signals > bearish_signals else 'Bearish' if bearish_signals > bullish_signals else 'Neutral'} based on weighted signal comparison"
            }
        }

        sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))
        
        # Ticker completion summary
        ticker_time = time.time() - ticker_start
        print(f"   ✅ {ticker} completed in {ticker_time:.2f}s")

    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name=agent_id,
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = sentiment_analysis

    progress.update_status(agent_id, None, "Done")
    
    # Final completion summary
    total_time = time.time() - start_time
    processed_tickers = len(sentiment_analysis)
    print(f"\n📰 {agent_id} COMPLETION SUMMARY:")
    print(f"   ✅ Processed tickers: {processed_tickers}/{len(tickers)}")
    print(f"   ⏱️  Total time: {total_time:.2f}s")
    if processed_tickers < len(tickers):
        print(f"   ⚠️  WARNING: Analysis incomplete due to timeout")
        print(f"   📊 Skipped tickers: {len(tickers) - processed_tickers}")

    return {
        "messages": [message],
        "data": data,
    }
