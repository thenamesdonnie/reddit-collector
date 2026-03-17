import yfinance as yf
import pandas as pd

MAG7 = {"AAPL":"AAPL",
        "MSFT":"MSFT",
        "GOOGL":"GOOGL",
        "AMZN":"AMZN",
        "NVDA":"NVDA",
        "META":"META",
        "TSLA":"TSLA",
        "MARKET":"^GSPC"}

for name, ticker in MAG7.items():
    mentions = pd.read_csv(f"./csvs/{name}_mentions.csv")
    start_date = mentions["date"].min()
    end_date = mentions["date"].max()
    print(f"Fetching {name} data from {start_date} to {end_date}")
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)  # flatten multi-level header
    df.index.name = "Date"
    df.to_csv(f"./stocks/{name}.csv")
    print(f"Saved {name}")