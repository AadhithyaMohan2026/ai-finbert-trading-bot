import requests
from alpaca_trade_api import REST
from datetime import datetime, timedelta

# 1. API Credentials
API_KEY = "YOUR_ALPACA_API_KEY"
API_SECRET = "YOUR_ALPACA_SECRET_KEY"
BASE_URL = "https://paper-api.alpaca.markets"

# 2. Test Account Connection
api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET, api_version='v2')

try:
    account = api.get_account()
    print("--- Alpaca Connection Successful ---")
    print(f"Account Status : {account.status}")
    print(f"Currency       : {account.currency}")
    print(f"Cash Balance   : ${float(account.cash):,.2f}")
    print(f"Buying Power   : ${float(account.buying_power):,.2f}\n")
except Exception as e:
    print(f"[ERROR] Account fetch failed: {e}")
    exit()

# 3. Test News Fetching via Alpaca Market Data Endpoint
symbol = "SPY"
today = datetime.today()
three_days_prior = today - timedelta(days=3)

print(f"--- Fetching News for {symbol} ({three_days_prior.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}) ---")

headers = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}

news_url = "https://data.alpaca.markets/v1beta1/news"
params = {
    "symbols": symbol,
    "start": three_days_prior.strftime("%Y-%m-%d"),
    "end": today.strftime("%Y-%m-%d"),
    "limit": 5
}

try:
    response = requests.get(news_url, headers=headers, params=params)
    data = response.json()
    news_items = data.get("news", [])

    if not news_items:
        print("No recent news found for this symbol within the selected date range.")
    else:
        for idx, item in enumerate(news_items, start=1):
            headline = item.get("headline", "")
            created_at = item.get("created_at", "")[:10]
            print(f"{idx}. [{created_at}] {headline}")
except Exception as e:
    print(f"[ERROR] News fetch failed: {e}")