"""
preprocess_reddit.py
--------------------
Pipeline:
  1. Clean post/comment text (URLs, boilerplate, markdown noise)
  2. Detect Mag7 ticker mentions via tickers, company names & aliases
  3. Build per-ticker  <TICKER>_mentions  tables
  4. Score with SentiStrength → <TICKER>_daily_sentiment tables

Requirements:  pip install pandas tqdm
Also needs:    Java + SentiStrength.jar + SentiStrength_Data/
               Download from http://sentistrength.wlv.ac.uk/
"""

import sqlite3, re, os, logging, subprocess, tempfile
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import shutil

# ── CONFIG ────────────────────────────────────────────────────────────────────
DB_PATH            = "reddit.db"
SENTISTRENGTH_JAR  = "./sentistrength/SentiStrength.jar"
SENTISTRENGTH_DATA = "./sentistrength/SentiStrength_Data/"
INPUT_DIR         = "./input/"
OUTPUT_DIR        = "./output/"
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Magnificent 7 aliases ─────────────────────────────────────────────────────
MAG7 = {
    "AAPL":  ["apple", "aapl", r"\$aapl"],
    "MSFT":  ["microsoft", "msft", r"\$msft", "azure"],
    "GOOGL": ["google", "googl", "goog", r"\$googl", r"\$goog",
               "alphabet", "deepmind", "waymo"],
    "AMZN":  ["amazon", "amzn", r"\$amzn", "aws"],
    "NVDA":  ["nvidia", "nvda", r"\$nvda", "nvdia"],
    "META":  ["meta", r"\$meta", "facebook", "instagram",
               "whatsapp", "threads", "zuckerberg", "zuck"],
    "TSLA":  ["tesla", "tsla", r"\$tsla", "elon musk"],
}

TICKER_PATTERNS = {}
for ticker, aliases in MAG7.items():
    parts = []
    for a in aliases:
        if any(c in a for c in r"\.^$*+?{}[]|()\$"):
            parts.append(a)
        else:
            parts.append(r"\b" + re.escape(a) + r"\b")
    TICKER_PATTERNS[ticker] = re.compile("|".join(parts), re.IGNORECASE)

# ── Text cleaning ─────────────────────────────────────────────────────────────
_URL      = re.compile(r"https?://\S+|www\.\S+")
_MENTION  = re.compile(r"u/\w+|r/\w+")
_HTML     = re.compile(r"&[a-z]+;|&#\d+;")
_WS       = re.compile(r"\s+")
_BOILER   = re.compile(r"^\s*(\[deleted\]|\[removed\])?\s*$", re.I)

def clean_text(text):
    if not text:
        return None
    t = _URL.sub(" ", text)
    t = _MENTION.sub(" ", t)
    t = _HTML.sub(" ", t)
    t = re.sub(r"[*_~`>#]", " ", t)
    t = _WS.sub(" ", t).strip()
    return None if (_BOILER.match(t) or len(t) < 3) else t

# ── DB helpers ────────────────────────────────────────────────────────────────
def get_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

# ── Load & clean ──────────────────────────────────────────────────────────────
def load_and_clean(conn):
    log.info("Loading posts...")
    posts = pd.read_sql_query(
        "SELECT id, subreddit, title, selftext, score, created_utc FROM posts", conn)
    log.info("Loading comments...")
    comments = pd.read_sql_query(
        "SELECT id, post_id, body, score, created_utc FROM comments", conn)

    posts["body"] = (
        (posts["title"].fillna("") + " " + posts["selftext"].fillna(""))
        .str.strip().apply(clean_text))
    posts["source"] = "post"

    comments["body"]      = comments["body"].apply(clean_text)
    comments["source"]    = "comment"
    comments["subreddit"] = None

    posts    = posts.dropna(subset=["body"])
    comments = comments.dropna(subset=["body"])
    log.info(f"Clean posts: {len(posts):,}  |  Clean comments: {len(comments):,}")
    return posts, comments

# ── Mention detection ─────────────────────────────────────────────────────────
def detect_tickers(text):
    return [t for t, pat in TICKER_PATTERNS.items() if pat.search(text)]

def utc_to_date(ts):
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return None

def populate_mentions(conn, posts, comments):
    buckets = {t: [] for t in MAG7}

    def scan(df, src):
        for row in tqdm(df.to_dict("records"), total=len(df), desc=f"Scanning {src}s"):
            hits = detect_tickers(row["body"])
            if not hits:
                continue
            date    = utc_to_date(row.get("created_utc"))
            post_id = row["id"] if src == "post" else row.get("post_id")
            for t in hits:
                buckets[t].append((
                    row["id"], src, post_id, row.get("subreddit"),
                    row["body"], row.get("score"), row.get("created_utc"), date,
                ))

    scan(posts, "post")
    scan(comments, "comment")

    for ticker in MAG7:
        filename = f"./csvs/{ticker}_mentions.csv"
        if not buckets[ticker]:
            log.info(f"  {ticker}: 0 mentions")
            continue
        df = pd.DataFrame(buckets[ticker],
                        columns=["id","source","post_id","subreddit",
                        "body","score","created_utc","date"])
        df.to_csv(filename, index=False)
        
        with open(f"./input/{ticker}.txt", "w", encoding="utf-8") as f:
            for row in buckets[ticker]:
                id   = row[0]  # comment id
                body = row[4].replace("\n", " ").replace("\t", " ")  # flatten
                f.write(f"{id}\t{body}\n")
        
        log.info(f"  {ticker}: {len(buckets[ticker]):,} mentions inserted")
        

# ── SentiStrength ─────────────────────────────────────────────────────────────

def roll_to_monday(date):
    if date.weekday() == 5:   # Saturday
        return date + pd.Timedelta(days=2)
    elif date.weekday() == 6:  # Sunday
        return date + pd.Timedelta(days=1)
    return date

def run_sentistrength():
    for ticker in MAG7:
        subprocess.run(
            ["java", "-jar", SENTISTRENGTH_JAR, "sentidata", SENTISTRENGTH_DATA, "input",
             f"./input/{ticker}.txt", "textCol", "2", "idCol", "1", "overwrite", "resultsExtension", "_out.csv"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        shutil.move(f"./input/{ticker}_classID.txt", f"./output/{ticker}_out.csv")
        log.info(f"  {ticker}: SentiStrength scoring completed")

def append_sentiment_scores():
    for ticker in MAG7:
        mentions_file = f"./csvs/{ticker}_mentions.csv"
        scores_file   = f"./output/{ticker}_out.csv"

        mentions = pd.read_csv(mentions_file)
        mentions = mentions.drop(columns=["positive","negative","compound"], errors="ignore")

        scores = pd.read_csv(scores_file, sep="\t", header=None, 
                            names=["id","positive","negative"])

        mentions["id"] = mentions["id"].astype(str)
        scores["id"]   = scores["id"].astype(str)

        mentions = mentions.merge(scores[["id","positive","negative"]],
                                on="id", how="left")
        mentions["compound"] = mentions["positive"] + mentions["negative"]

        mentions.to_csv(mentions_file, index=False)
        log.info(f"  {ticker}: scores appended to mentions csv")

        daily = (
            mentions.groupby("date")
            .agg(
                mention_count=("id", "count"),
                avg_positive=("positive", "mean"),
                avg_negative=("negative", "mean"),
                avg_compound=("compound", "mean"),
                median_positive=("positive", "median"),
                median_negative=("negative", "median"),
            )
            .reset_index()
        )

        # roll weekend sentiment into Monday
        daily["date"] = pd.to_datetime(daily["date"])

        daily["date"] = daily["date"].apply(roll_to_monday)

        # re-aggregate in case multiple days now share the same Monday
        daily = (
            daily.groupby("date")
            .agg(
                mention_count=("mention_count", "sum"),
                avg_positive=("avg_positive", "mean"),
                avg_negative=("avg_negative", "mean"),
                avg_compound=("avg_compound", "mean"),
                median_positive=("median_positive", "mean"),
                median_negative=("median_negative", "mean"),
            )
            .reset_index()
        )
        daily.to_csv(f"./csvs/{ticker}_daily_sentiment.csv", index=False)
        log.info(f"  {ticker}: daily sentiment written ({len(daily)} days)")


# ── Entire Market ────────────────────────────────────────────────────

def aggregate_market_sentiment(conn):
    log.info("Loading all comments for market sentiment...")
    comments = pd.read_sql_query(
        "SELECT id, body, score, created_utc FROM comments", conn)
    posts = pd.read_sql_query(
        "SELECT id, title, selftext, score, created_utc FROM posts", conn)

    comments["body"] = comments["body"].apply(clean_text)
    posts["body"] = (
        (posts["title"].fillna("") + " " + posts["selftext"].fillna(""))
        .str.strip().apply(clean_text))

    comments = comments.dropna(subset=["body"])
    posts     = posts.dropna(subset=["body"])

    # combine into one
    all_text = pd.concat([
        comments[["id","body","score","created_utc"]],
        posts[["id","body","score","created_utc"]]
    ]).reset_index(drop=True)

    all_text["date"] = all_text["created_utc"].apply(utc_to_date)
    all_text["date"] = pd.to_datetime(all_text["date"])

    # roll weekends into Monday
    all_text["date"] = all_text["date"].apply(roll_to_monday)

    all_text.to_csv("./input/MARKET.txt", index=False, header=False)

    # write input file with id and body
    with open("./input/MARKET.txt", "w", encoding="utf-8") as f:
        for _, row in tqdm(all_text.iterrows(), total=len(all_text), desc="Writing market input"):
            body = row["body"].replace("\n", " ").replace("\t", " ")
            f.write(f"{row['id']}\t{body}\n")

    log.info(f"Market input written: {len(all_text):,} texts")

    # run sentistrength
    subprocess.run([
        "java", "-jar", SENTISTRENGTH_JAR,
        "sentidata", SENTISTRENGTH_DATA,
        "input", "./input/MARKET.txt",
        "textCol", "2",
        "idCol", "1",
        "overwrite",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    shutil.move("./input/MARKET_classID.txt", "./output/MARKET_out.csv")

    # read scores back
    scores = pd.read_csv("./output/MARKET_out.csv", sep="\t", header=None,
                         names=["id","positive","negative"])

    all_text["id"]  = all_text["id"].astype(str)
    scores["id"]    = scores["id"].astype(str)

    merged = all_text.merge(scores[["id","positive","negative"]], on="id", how="left")
    merged["compound"] = merged["positive"] + merged["negative"]
    merged["ratio"]    = (merged["positive"] + merged["negative"]) / (merged["positive"].abs() + merged["negative"].abs())

    merged.to_csv("./csvs/MARKET_mentions.csv", index=False)

    # daily aggregation
    daily = (
        merged.groupby("date")
        .agg(
            mention_count=("id", "count"),
            avg_positive=("positive", "mean"),
            avg_negative=("negative", "mean"),
            avg_compound=("compound", "mean"),
            median_positive=("positive", "median"),
            median_negative=("negative", "median"),
        )
        .reset_index()
    )

    daily.to_csv("./csvs/MARKET_daily_sentiment.csv", index=False)
    log.info(f"Market daily sentiment written ({len(daily)} days)")

# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ns", "--no-scan", action="store_true", 
                        help="Skip scanning comments and use existing mentions tables")
    parser.add_argument("-nm", "--no-market", action="store_true",
                    help="Skip aggregate market sentiment analysis")
    parser.add_argument("-ss", "--skip-sentiment", action="store_true",
                        help="Skip running SentiStrength and use existing sentiment scores")
    args = parser.parse_args()

    conn = get_db()

    if not args.no_market:
        aggregate_market_sentiment(conn)
    if not args.no_scan:
        posts, comments = load_and_clean(conn)
        populate_mentions(conn, posts, comments)
    else:
        log.info("Skipping scan, using existing mentions tables")
    if not args.skip_sentiment:
        run_sentistrength()
    else:
        log.info("Skipping SentiStrength, using existing sentiment scores")
    append_sentiment_scores()
    conn.close()
    log.info("Done!")

if __name__ == "__main__":
     main()