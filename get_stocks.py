import yfinance as yf
import pandas as pd

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

for ticker in MAG7:
    mentions = pd.read_csv(f"./csvs/{ticker}_mentions.csv")
    start_date = mentions["date"].min()
    end_date = mentions["date"].max()
    print(f"Fetching {ticker} data from {start_date} to {end_date}")
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)  # flatten multi-level header
    df.index.name = "Date"
    df.to_csv(f"./stocks/{ticker}.csv")
    print(f"Saved {ticker}")