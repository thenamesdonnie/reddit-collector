import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")
import os
from sklearn.model_selection import TimeSeriesSplit

os.makedirs("./graphs", exist_ok=True)
os.makedirs("./analysis", exist_ok=True)

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "MARKET"]

def load_features(ticker):
    df = pd.read_csv(f"./analysis/{ticker}_aligned.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # baseline features — price only
    baseline_features = ["return_lag1", "return_lag2"]

    # sentiment enhanced features
    sentiment_features = [
        "return_lag1", "return_lag2",
        "avg_compound", "ratio",
        "compound_lag1", "ratio_lag1"
    ]

    target = "direction"  # 1 = up, 0 = down

    df = df.dropna(subset=baseline_features + sentiment_features + [target])

    return df, baseline_features, sentiment_features, target

def train_test_split_temporal(df, target, features, split=0.7):
    """Time-ordered split — no shuffling to avoid look-ahead bias"""
    n = int(len(df) * split)
    train = df.iloc[:n]
    test  = df.iloc[n:]

    X_train = train[features]
    y_train = train[target]
    X_test  = test[features]
    y_test  = test[target]

    # scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy":  round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        "f1":        round(f1_score(y_test, preds, zero_division=0), 4),
        "preds":     preds,
        "actual":    y_test.values,
    }

def run_models(ticker, df, baseline_features, sentiment_features, target):
    results = []

    for label, features in [("baseline", baseline_features), 
                              ("sentiment", sentiment_features)]:
        X_train, X_test, y_train, y_test = train_test_split_temporal(
            df, target, features)

        for model_name, model in [
            ("LogisticRegression", LogisticRegression(max_iter=1000)),
            ("RandomForest",       RandomForestClassifier(n_estimators=100, random_state=42)),
        ]:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            results.append({
                "ticker":    ticker,
                "model":     model_name,
                "features":  label,
                "accuracy":  metrics["accuracy"],
                "precision": metrics["precision"],
                "recall":    metrics["recall"],
                "f1":        metrics["f1"],
            })
            print(f"  {ticker} | {model_name} | {label}: accuracy={metrics['accuracy']}, f1={metrics['f1']}")

            # confusion matrix
            cm = confusion_matrix(metrics["actual"], metrics["preds"])
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Down","Up"], yticklabels=["Down","Up"])
            plt.title(f"{ticker} — {model_name} ({label})")
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plt.tight_layout()
            plt.savefig(f"./graphs/cm_{ticker}_{model_name}_{label}.png", dpi=200)
            plt.close()

    return results

def run_models_kfold(ticker, df, baseline_features, sentiment_features, target, n_splits=5):
    results = []
    tscv = TimeSeriesSplit(n_splits=n_splits)

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=1000)),
        ("RandomForest",       RandomForestClassifier(n_estimators=100, random_state=42))
    ]

    for label, features in [("baseline", baseline_features),
                              ("sentiment", sentiment_features)]:
        X = df[features].values
        y = df[target].values

        for model_name, model in models:
            fold_metrics = {
                "accuracy": [], "precision": [], "recall": [], "f1": []
            }

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test  = scaler.transform(X_test)

                # skip fold if only one class in training set
                if len(np.unique(y_train)) < 2:
                    continue

                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                fold_metrics["accuracy"].append(accuracy_score(y_test, preds))
                fold_metrics["precision"].append(precision_score(y_test, preds, zero_division=0))
                fold_metrics["recall"].append(recall_score(y_test, preds, zero_division=0))
                fold_metrics["f1"].append(f1_score(y_test, preds, zero_division=0))

            if not fold_metrics["accuracy"]:
                continue

            results.append({
                "ticker":    ticker,
                "model":     model_name,
                "features":  label,
                "accuracy":  round(np.mean(fold_metrics["accuracy"]), 4),
                "precision": round(np.mean(fold_metrics["precision"]), 4),
                "recall":    round(np.mean(fold_metrics["recall"]), 4),
                "f1":        round(np.mean(fold_metrics["f1"]), 4),
                "accuracy_std": round(np.std(fold_metrics["accuracy"]), 4),
            })
            print(f"  {ticker} | {model_name} | {label}: "
                  f"accuracy={results[-1]['accuracy']:.3f} "
                  f"(±{results[-1]['accuracy_std']:.3f}), "
                  f"f1={results[-1]['f1']:.3f}")

    return results

def plot_ml_comparison(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, model_name in zip(axes, ["LogisticRegression", "RandomForest"]):
        df = results_df[results_df["model"] == model_name]

        baseline  = df[df["features"] == "baseline"].set_index("ticker")["accuracy"]
        sentiment = df[df["features"] == "sentiment"].set_index("ticker")["accuracy"]

        tickers = baseline.index.tolist()
        x = np.arange(len(tickers))
        width = 0.35

        bars1 = ax.bar(x - width/2, baseline.values,  width, label="Baseline",  color="steelblue")
        bars2 = ax.bar(x + width/2, sentiment.values, width, label="Sentiment", color="coral")

        # add value labels on bars
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

        ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", 
                   alpha=0.5, label="Random chance (0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45)
        ax.set_ylim(0, 0.75)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{model_name}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Baseline vs Sentiment-Enhanced Model Accuracy", fontsize=14)
    plt.tight_layout()
    plt.savefig("./graphs/ml_comparison.png", dpi=200)
    plt.close()
    print("Saved ml_comparison.png")

def plot_f1_comparison(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, model_name in zip(axes, ["LogisticRegression", "RandomForest"]):
        df = results_df[results_df["model"] == model_name]

        baseline  = df[df["features"] == "baseline"].set_index("ticker")["f1"]
        sentiment = df[df["features"] == "sentiment"].set_index("ticker")["f1"]

        tickers = baseline.index.tolist()
        x = np.arange(len(tickers))
        width = 0.35

        bars1 = ax.bar(x - width/2, baseline.values,  width, label="Baseline",  color="steelblue")
        bars2 = ax.bar(x + width/2, sentiment.values, width, label="Sentiment", color="coral")

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45)
        ax.set_ylim(0, 0.75)
        ax.set_ylabel("F1 Score")
        ax.set_title(f"{model_name}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Baseline vs Sentiment-Enhanced Model F1 Score", fontsize=14)
    plt.tight_layout()
    plt.savefig("./graphs/ml_f1_comparison.png", dpi=200)
    plt.close()
    print("Saved ml_f1_comparison.png")

def plot_summary_table(results_df):
    # get best model per ticker and feature set based on accuracy
    best = (
        results_df
        .sort_values("accuracy", ascending=False)
        .groupby(["ticker", "features"])
        .first()
        .reset_index()
    )

    baseline  = best[best["features"] == "baseline"].set_index("ticker")
    sentiment = best[best["features"] == "sentiment"].set_index("ticker")

    tickers = sorted(set(baseline.index) & set(sentiment.index))

    rows = []
    for ticker in tickers:
        b = baseline.loc[ticker]
        s = sentiment.loc[ticker]
        improvement = round(s["accuracy"] - b["accuracy"], 4)
        rows.append({
            "Ticker":           ticker,
            "Best Baseline Model":    b["model"],
            "Baseline Acc":     b["accuracy"],
            "Baseline Acc ±":   b["accuracy_std"],
            "Best Sentiment Model":   s["model"],
            "Sentiment Acc":    s["accuracy"],
            "Sentiment Acc ±":  s["accuracy_std"],
            "Improvement":      improvement,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv("./analysis/ml_summary.csv", index=False)

    # plot as a table
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")

    col_labels = [
        "Ticker", "Best Baseline", "Baseline Acc (±std)",
        "Best Sentiment", "Sentiment Acc (±std)", "Improvement"
    ]

    table_data = []
    for _, row in summary.iterrows():
        table_data.append([
            row["Ticker"],
            row["Best Baseline Model"],
            f"{row['Baseline Acc']:.3f} (±{row['Baseline Acc ±']:.3f})",
            row["Best Sentiment Model"],
            f"{row['Sentiment Acc']:.3f} (±{row['Sentiment Acc ±']:.3f})",
            f"{row['Improvement']:+.3f}",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # colour improvement column green/red
    for i, row in enumerate(table_data):
        improvement = float(row[5])
        color = "#d4edda" if improvement > 0 else "#f8d7da" if improvement < 0 else "#ffffff"
        table[i + 1, 5].set_facecolor(color)
        # shade header
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#343a40")
            table[0, j].set_text_props(color="white", fontweight="bold")

    plt.title("Best Model Summary — Baseline vs Sentiment Enhanced", 
              fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig("./graphs/ml_summary_table.png", dpi=200)
    plt.close()
    print("Saved ml_summary_table.png")
    print("\nSummary:")
    print(summary.to_string(index=False))

def plot_feature_importance(ticker, df, sentiment_features, target):
    X = df[sentiment_features].values
    y = df[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # sort descending

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(sentiment_features)),
                  importances[indices],
                  color="steelblue", edgecolor="white")

    ax.set_xticks(range(len(sentiment_features)))
    ax.set_xticklabels([sentiment_features[i] for i in indices], 
                        rotation=45, ha="right")
    ax.set_ylabel("Feature Importance")
    ax.set_title(f"{ticker} — Random Forest Feature Importance")

    # add value labels on bars
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./graphs/feature_importance_{ticker}.png", dpi=200)
    plt.close()
    print(f"Saved feature_importance_{ticker}.png")

if __name__ == "__main__":
    all_results = []

    for ticker in MAG7 + ["MARKET"]:
        print(f"\n{ticker}")
        try:
            df, baseline_features, sentiment_features, target = load_features(ticker)
            print(f"  {len(df)} samples, {5}-fold CV")
            results = run_models_kfold(ticker, df, baseline_features, 
                                       sentiment_features, target)
            all_results.extend(results)
        except Exception as e:
            print(f"  ERROR: {e}")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.drop_duplicates(subset=["ticker","model","features"])
    results_df.to_csv("./analysis/ml_results_kfold.csv", index=False)
    print("\n")
    print(results_df.to_string(index=False))

    plot_ml_comparison(results_df)
    plot_f1_comparison(results_df)

    plot_summary_table(results_df)

    print("\nGenerating feature importance plots...")
    for ticker in MAG7:
        try:
            df, baseline_features, sentiment_features, target = load_features(ticker)
            plot_feature_importance(ticker, df, sentiment_features, target)
        except Exception as e:
            print(f"  {ticker} ERROR: {e}")