import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
matplotlib.use("Agg")

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]

for ticker in MAG7:
    sentiment = pd.read_csv(f"./csvs/{ticker}_daily_sentiment.csv")
    stock     = pd.read_csv(f"./stocks/{ticker}.csv")

    sentiment["date"] = pd.to_datetime(sentiment["date"])
    stock["date"]     = pd.to_datetime(stock["Date"])

    # sentiment - z-score
    sentiment["avg_compound_smooth"] = (
        sentiment["avg_compound"].rolling(7, min_periods=1).mean()
    )
    sentiment["avg_compound_norm"] = (
        (sentiment["avg_compound_smooth"] - sentiment["avg_compound_smooth"].mean())
        / sentiment["avg_compound_smooth"].std()
    ) * 10
    lower = sentiment["avg_compound_norm"].quantile(0.05)
    upper = sentiment["avg_compound_norm"].quantile(0.95)
    sentiment["avg_compound_norm"] = sentiment["avg_compound_norm"].clip(lower, upper)

    # stock price - normalised
    stock["close_norm"] = (stock["Close"] - stock["Close"].mean()) / stock["Close"].std() * 100

    # % change - z-score
    stock["pct_change"] = stock["Close"].pct_change() * 100
    stock.fillna({"pct_change": 0}, inplace=True)
    stock["pct_change_smooth"] = stock["pct_change"].rolling(7, min_periods=1).mean()
    stock["pct_change_norm"] = (
        (stock["pct_change_smooth"] - stock["pct_change_smooth"].mean())
        / stock["pct_change_smooth"].std()
    )

    # two subplots stacked vertically, shared x axis
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # ── top plot: price + sentiment ───────────────────────────────────────
    ax_top.plot(stock["date"], stock["close_norm"],
                color="orange", linestyle="--", label="Normalised Price")
    ax_top.set_ylabel("Normalised Price")

    ax_top_r = ax_top.twinx()
    ax_top_r.plot(sentiment["date"], sentiment["avg_compound_norm"],
                  color="blue", linestyle="-", label="Normalised Sentiment")
    ax_top_r.set_ylabel("Normalised Sentiment")

    lines1, labels1 = ax_top.get_legend_handles_labels()
    lines2, labels2 = ax_top_r.get_legend_handles_labels()
    ax_top.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax_top.set_title(f"{ticker} Normalised Price vs Reddit Sentiment")
    ax_top.grid(True)

    # ── bottom plot: % change + sentiment ────────────────────────────────
    ax_bot.plot(stock["date"], stock["pct_change_norm"],
                color="green", linestyle=":", label="% Change")
    ax_bot.set_ylabel("% Change (normalised)")

    ax_bot_r = ax_bot.twinx()
    ax_bot_r.plot(sentiment["date"], sentiment["avg_compound_norm"],
                  color="blue", linestyle="-", label="Normalised Sentiment")
    ax_bot_r.set_ylabel("Normalised Sentiment")

    lines3, labels3 = ax_bot.get_legend_handles_labels()
    lines4, labels4 = ax_bot_r.get_legend_handles_labels()
    ax_bot.legend(lines3 + lines4, labels3 + labels4, loc="upper left")
    ax_bot.set_title(f"{ticker} % Change vs Reddit Sentiment")
    ax_bot.grid(True)

    # ── shared x axis formatting ──────────────────────────────────────────
    ax_bot.xaxis.set_major_locator(mdates.MonthLocator())
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax_bot.xaxis.get_majorticklabels(), rotation=45)

    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(f"./graphs/{ticker}_sentiment.png", dpi=200)
    plt.close()