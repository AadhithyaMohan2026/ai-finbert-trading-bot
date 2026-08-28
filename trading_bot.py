from datetime import datetime, timedelta
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.strategies.strategy import Strategy
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==========================================
# 1. Configuration & API Credentials
# ==========================================
API_KEY = "YOUR_ALPACA_API_KEY"
API_SECRET = "YOUR_ALPACA_SECRET_KEY"
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True,
}

# ==========================================
# 2. NLP Sentiment Analysis Engine (FinBERT)
# ==========================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Initializing FinBERT model on {device}...")

MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(
    device
)
LABELS = ["Positive", "Negative", "Neutral"]


def analyze_sentiment(news_headlines):
  """Tokenizes headlines and returns mean probability and highest-scoring sentiment label."""
  if not news_headlines:
    return 0.0, "Neutral"

  tokens = tokenizer(
      news_headlines, padding=True, truncation=True, return_tensors="pt"
  ).to(device)

  with torch.no_grad():
    outputs = model(**tokens)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

  mean_probabilities = torch.mean(probabilities, dim=0)
  best_idx = torch.argmax(mean_probabilities).item()
  confidence = mean_probabilities[best_idx].item()
  sentiment = LABELS[best_idx]

  return confidence, sentiment


# ==========================================
# 3. Algorithmic Strategy Logic
# ==========================================
class FinBERTSentimentStrategy(Strategy):

  def initialize(self, symbol: str = "SPY", cash_at_risk: float = 0.5):
    self.symbol = symbol
    self.sleeptime = "24H"
    self.last_trade = None
    self.cash_at_risk = cash_at_risk

  def get_news_headlines(self):
    """Fetches news headlines from Alpaca Market Data endpoint for the past 3 days."""
    today = self.get_datetime()
    three_days_ago = today - timedelta(days=3)

    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
    }
    params = {
        "symbols": self.symbol,
        "start": three_days_ago.strftime("%Y-%m-%d"),
        "end": today.strftime("%Y-%m-%d"),
        "limit": 10,
    }

    try:
      response = requests.get(
          "https://data.alpaca.markets/v1beta1/news",
          headers=headers,
          params=params,
          timeout=10,
      )
      data = response.json()
      news_items = data.get("news", [])
      return [
          item.get("headline", "")
          for item in news_items
          if "headline" in item
      ]
    except Exception as e:
      print(f"[WARN] Failed to fetch news: {e}")
      return []

  def calculate_position_size(self):
    """Calculates dynamic share quantity based on account cash and risk allocation."""
    cash = self.get_cash()
    last_price = self.get_last_price(self.symbol)
    if not last_price or last_price <= 0:
      return cash, 0, 0
    quantity = int((cash * self.cash_at_risk) / last_price)
    return cash, last_price, quantity

  def on_trading_iteration(self):
    """Main event-driven execution loop called every trading cycle."""
    cash, last_price, quantity = self.calculate_position_size()
    if quantity <= 0 or cash < last_price:
      return

    headlines = self.get_news_headlines()
    confidence, sentiment = analyze_sentiment(headlines)

    print(
        f"\n[{self.get_datetime().strftime('%Y-%m-%d')}] Symbol: {self.symbol} |"
        f" Price: ${last_price:.2f}"
    )
    print(
        f"Sentiment: {sentiment} ({confidence:.2%} confidence) | Headcount:"
        f" {len(headlines)} headlines"
    )

    # Execute trades on high confidence signals
    if confidence > 0.85:
      if sentiment == "Positive":
        if self.last_trade == "sell":
          self.sell_all()

        # Positional arguments: (asset, quantity, side)
        buy_order = self.create_order(
            self.symbol,
            quantity,
            "buy",
            take_profit_price=last_price * 1.20,
            stop_loss_price=last_price * 0.95,
        )
        self.submit_order(buy_order)
        self.last_trade = "buy"
        print(
            f"[EXECUTED] BUY {quantity} shares of {self.symbol} @"
            f" ${last_price:.2f}"
        )

      elif sentiment == "Negative":
        if self.last_trade == "buy":
          self.sell_all()

        # Positional arguments: (asset, quantity, side)
        sell_order = self.create_order(
            self.symbol,
            quantity,
            "sell",
            take_profit_price=last_price * 0.80,
            stop_loss_price=last_price * 1.05,
        )
        self.submit_order(sell_order)
        self.last_trade = "sell"
        print(
            f"[EXECUTED] SELL/SHORT {quantity} shares of {self.symbol} @"
            f" ${last_price:.2f}"
        )


# ==========================================
# 4. Strategy Execution / Backtest Harness
# ==========================================
if __name__ == "__main__":
  start_date = datetime(2023, 1, 1)
  end_date = datetime(2023, 12, 31)

  print("[INFO] Initializing Alpaca Paper Broker & Backtester...")
  broker = Alpaca(ALPACA_CREDS)
  strategy = FinBERTSentimentStrategy(
      name="FinBERT_SPY_Strategy", broker=broker
  )

  print(
      f"[INFO] Starting historical backtest from {start_date.date()} to"
      f" {end_date.date()}...\n"
  )
  strategy.backtest(
      YahooDataBacktesting, start_date, end_date, benchmark_asset="SPY"
  )