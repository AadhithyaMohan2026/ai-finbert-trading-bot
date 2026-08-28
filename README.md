# NLP-Driven Algorithmic Trading Bot using FinBERT & Alpaca API

An event-driven algorithmic trading bot developed in Python that leverages Natural Language Processing (NLP) to make automated equity trading decisions on the S&P 500 (`SPY`) based on financial news sentiment.

---

## 📌 Project Overview
Traditional technical indicators (RSI, Moving Averages, MACD) often act as lagging signals during major macroeconomic shifts. This project implements a sentiment-driven pipeline where textual data (financial news headlines) serves as the primary leading indicator for trade execution.

The bot ingests real-time news data, runs inference via a fine-tuned Transformer model (**FinBERT**), dynamically determines position sizing based on risk parameters, and executes simulated bracket orders via the Alpaca Paper Trading API.

---

## 🏗️ Architecture & Pipeline
1. **Data Ingestion**: Ingests real-time financial news headlines via Alpaca's Market Data REST API (`/v1beta1/news`).
2. **NLP Sentiment Classification**: Leverages Hugging Face's `ProsusAI/finbert` model (BERT architecture fine-tuned on financial phrasebanks) to output probabilities across `Positive`, `Negative`, and `Neutral` classes.
3. **Algorithmic Signal Generation**: Executes orders only when the model's confidence exceeds a high-conviction threshold (`confidence > 0.85`).
4. **Risk Management & Execution**:
   - **Dynamic Position Sizing**: Allocates capital proportionally based on available cash and risk threshold (`cash_at_risk = 0.5`).
   - **Bracket Orders**: Automatically places Take-Profit (+20%) and Stop-Loss (-5%) triggers alongside market orders to cap downside risk.
5. **Backtesting Engine**: Simulates end-to-end execution against historical Yahoo Finance market data using the `lumibot` framework.

---

## 📁 Repository Structure
```text
├── trading_bot.py      # Core strategy logic, risk management, and backtest harness
├── sentiment_test.py   # Standalone FinBERT inference & tokenization script
├── alpaca_test.py      # Broker REST API connection & news fetching script
├── requirements.txt    # Project dependencies
├── .gitignore          # Excludes virtual environment and temporary artifacts
└── README.md           # Project documentation