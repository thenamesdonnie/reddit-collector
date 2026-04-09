import pandas as pd
import os
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.dates as mdates
from scipy.stats import spearmanr

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "MARKET"]

SENTIMENT_COLS = ["avg_compound", "ratio", "avg_positive", "avg_negative",
                  "ratio_lag1", "ratio_lag2", "compound_lag1", "compound_lag2"]



def load_aligned(ticker):
    sentiment = pd.read_csv(f"./csvs/{ticker}_daily_sentiment.csv")
    stock     = pd.read_csv(f"./stocks/{ticker}.csv")

    sentiment["date"] = pd.to_datetime(sentiment["date"])
    stock["date"]     = pd.to_datetime(stock["Date"])

    # step 2 - derived columns
    stock["return"] = stock["Close"].pct_change()
    stock["direction"] = (stock["return"] > 0).astype(int)  # 1 = up, 0 = down

    sentiment["ratio"] = (
        (sentiment["avg_positive"] - sentiment["avg_negative"].abs())
        / (sentiment["avg_positive"] + sentiment["avg_negative"].abs())
    )

    # merge
    df = pd.merge(sentiment, stock[["date","Close","return","direction"]], 
                  on="date", how="inner")

    # lagged sentiment
    df["ratio_lag1"] = df["ratio"].shift(1)
    df["ratio_lag2"] = df["ratio"].shift(2)
    df["compound_lag1"] = df["avg_compound"].shift(1)
    df["compound_lag2"] = df["avg_compound"].shift(2)

    # lagged returns for ML later
    df["return_lag1"] = df["return"].shift(1)
    df["return_lag2"] = df["return"].shift(2)

    df = df.dropna()
    df["ticker"] = ticker

    return df

def correlate(df, sentiment_col, return_col):
    clean = df[[sentiment_col, return_col]].dropna()
    if len(clean) < 10:
        return None, None
    r, p = stats.pearsonr(clean[sentiment_col], clean[return_col])
    return round(r, 4), round(p, 4)

def build_correlation_table():
    results = []
    for ticker in MAG7:
        df = pd.read_csv(f"./analysis/{ticker}_aligned.csv")
        df["date"] = pd.to_datetime(df["date"])

        for sentiment_col in SENTIMENT_COLS:
            for lag, return_col in [(0, "return"), (1, "return_lag1"), (2, "return_lag2")]:
                r, p = correlate(df, sentiment_col, return_col)
                if r is not None:
                    results.append({
                        "ticker":        ticker,
                        "sentiment":     sentiment_col,
                        "return_lag":    lag,
                        "pearson_r":     r,
                        "p_value":       p,
                        "significant":   p < 0.05 if p else False,
                    })

    corr_df = pd.DataFrame(results)
    corr_df.to_csv("./analysis/correlation_table.csv", index=False)
    print(corr_df[corr_df["significant"]].to_string())
    print(f"\n{corr_df['significant'].sum()} significant correlations found")
    return corr_df

def plot_heatmap(corr_df, return_lag, title, filename):
    pivot = corr_df[corr_df["return_lag"] == return_lag].pivot(
        index="ticker", columns="sentiment", values="pearson_r"
    )
    sig = corr_df[corr_df["return_lag"] == return_lag].pivot(
        index="ticker", columns="sentiment", values="significant"
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.5)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)

    # annotate each cell with r value and * if significant
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            is_sig = sig.values[i, j]
            if not np.isnan(val):
                label = f"{val:.2f}{'*' if is_sig else ''}"
                ax.text(j, i, label, ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, label="Pearson r")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"./graphs/{filename}", dpi=200)
    plt.close()
    print(f"Saved {filename}")

def plot_scatter(ticker, sentiment_col, return_col, lag_label):
    df = pd.read_csv(f"./analysis/{ticker}_aligned.csv")
    
    clean = df[[sentiment_col, return_col]].dropna()
    r, p = stats.pearsonr(clean[sentiment_col], clean[return_col])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(clean[sentiment_col], clean[return_col], 
               alpha=0.6, color="steelblue", edgecolors="white", linewidth=0.5)
    
    # trend line
    m, b = np.polyfit(clean[sentiment_col], clean[return_col], 1)
    x_line = np.linspace(clean[sentiment_col].min(), clean[sentiment_col].max(), 100)
    ax.plot(x_line, m * x_line + b, color="red", linewidth=2, label=f"r={r:.3f}, p={p:.3f}")
    
    # zero lines for reference
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    
    ax.set_xlabel(sentiment_col)
    ax.set_ylabel(f"Return ({lag_label})")
    ax.set_title(f"{ticker} — {sentiment_col} vs {lag_label} Return")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"./graphs/scatter_{ticker}_{sentiment_col}_{lag_label}.png", dpi=200)
    plt.close()
    print(f"Saved scatter_{ticker}_{sentiment_col}_{lag_label}.png")

def plot_rolling_correlation(ticker, sentiment_col, return_col, window=14):
    df = pd.read_csv(f"./analysis/{ticker}_aligned.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[sentiment_col, return_col])

    df["rolling_corr"] = df[sentiment_col].rolling(window).corr(df[return_col])

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(df["date"], df["rolling_corr"],
            color="steelblue", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.fill_between(df["date"], df["rolling_corr"], 0,
                    where=df["rolling_corr"] > 0, alpha=0.2, color="green")
    ax.fill_between(df["date"], df["rolling_corr"], 0,
                    where=df["rolling_corr"] <= 0, alpha=0.2, color="red")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-1, 1)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.title(f"{ticker} — Rolling {window}-day Sentiment-Return Correlation")
    plt.tight_layout()
    plt.savefig(f"./graphs/rolling_{ticker}_{sentiment_col}.png", dpi=200)
    plt.close()

def build_spearman_table():
    results = []
    for ticker in MAG7:
        df = pd.read_csv(f"./analysis/{ticker}_aligned.csv")
        df["date"] = pd.to_datetime(df["date"])

        for sentiment_col in SENTIMENT_COLS:
            for lag, return_col in [(0, "return"), (1, "return_lag1"), (2, "return_lag2")]:
                clean = df[[sentiment_col, return_col]].dropna()
                if len(clean) < 10:
                    continue
                r, p = spearmanr(clean[sentiment_col], clean[return_col])
                results.append({
                    "ticker":      ticker,
                    "sentiment":   sentiment_col,
                    "return_lag":  lag,
                    "spearman_r":  round(r, 4),
                    "p_value":     round(p, 4),
                    "significant": p < 0.05,
                })

    spear_df = pd.DataFrame(results)
    spear_df.to_csv("./analysis/spearman_table.csv", index=False)
    sig = spear_df[spear_df["significant"]]
    print(f"\nSpearman — {len(sig)} significant correlations:")
    print(sig.to_string(index=False))
    return spear_df

if __name__ == "__main__":
    os.makedirs("./analysis", exist_ok=True)

    all_dfs = []
    for ticker in MAG7:
        df = load_aligned(ticker)
        df.to_csv(f"./analysis/{ticker}_aligned.csv", index=False)
        print(f"{ticker}: {len(df)} trading days aligned")
        all_dfs.append(df)

    combined = pd.concat(all_dfs)
    combined.to_csv("./analysis/all_aligned.csv", index=False)
    print(f"\nCombined: {len(combined)} rows across {len(MAG7)} tickers")

    print("\nBuilding correlation table...")

    corr_df = build_correlation_table()

    plot_heatmap(corr_df, 0, "Sentiment vs Same-Day Return",    "heatmap_lag0.png")
    plot_heatmap(corr_df, 1, "Sentiment vs Next-Day Return",    "heatmap_lag1.png")
    plot_heatmap(corr_df, 2, "Sentiment vs 2-Day Ahead Return", "heatmap_lag2.png")

    #strongest results
    plot_scatter("TSLA", "avg_compound",   "return_lag1", "t+1")
    plot_scatter("TSLA", "compound_lag1",  "return_lag2", "t+2")
    plot_scatter("NVDA", "avg_compound",   "return_lag1", "t+1")
    plot_scatter("GOOGL", "avg_compound",  "return_lag1", "t+1")
    plot_scatter("GOOGL", "avg_compound",  "return_lag2", "t+2")

    plot_rolling_correlation("TSLA", "avg_compound", "return_lag1")
    plot_rolling_correlation("NVDA", "avg_compound", "return_lag1")
    plot_rolling_correlation("GOOGL", "avg_compound", "return_lag1")
    plot_rolling_correlation("MARKET", "avg_compound", "return_lag1")

    print("\nBuilding Spearman correlation table...")
    build_spearman_table()
