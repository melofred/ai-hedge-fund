python3 src/compare_each_agent.py \
  --tickers AAPL,MSFT,NVDA \
  --initial-cash 100000 \
  --end-date $(date +%F) \
  --model-provider ollama
