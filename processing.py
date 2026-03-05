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
import csv
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

def create_ticker_tables(conn):
    c = conn.cursor()
    for ticker in MAG7:
        c.execute(f"""
            CREATE TABLE IF NOT EXISTS {ticker}_mentions (
                id TEXT PRIMARY KEY, source TEXT NOT NULL,
                post_id TEXT, subreddit TEXT, body TEXT,
                score INTEGER, created_utc INTEGER, date TEXT
            )""")
        c.execute(f"""
            CREATE TABLE IF NOT EXISTS {ticker}_daily_sentiment (
                date TEXT PRIMARY KEY, mention_count INTEGER,
                avg_positive REAL, avg_negative REAL, avg_compound REAL,
                median_positive REAL, median_negative REAL,
                weighted_avg_positive REAL, weighted_avg_negative REAL
            )""")
    conn.commit()
    log.info("Ticker tables ready.")

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
    c = conn.cursor()
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

def run_sentistrength():
    for ticker in MAG7:
        subprocess.run(
            ["java", "-jar", SENTISTRENGTH_JAR, "sentidata", SENTISTRENGTH_DATA, "input",
             f"./input/{ticker}.txt", "textCol", "2", "idCol", "1", "overwrite", "resultsExtension", "_out.csv"]
        )
        shutil.move(f"./input/{ticker}_classID.txt", f"./output/{ticker}_out.csv")

def append_sentiment_scores():
    for ticker in MAG7:
        mentions_file = f"./csvs/{ticker}_mentions.csv"
        scores_file   = f"./output/{ticker}_out.csv"

        mentions = pd.read_csv(mentions_file)
        print(mentions.head())

        scores = pd.read_csv(scores_file, sep="\t", header=None, names=["id","positive","negative"])
        print(scores.head())

        # merge on id so mismatched lengths don't matter
        mentions = mentions.merge(scores[["id","positive","negative"]],
                                  on="id", how="left")
        mentions["compound"] = mentions["positive"] + mentions["negative"]

        mentions.to_csv(mentions_file, index=False, header=False)
        log.info(f"  {ticker}: scores appended to mentions csv")

# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ns", "--no-scan", action="store_true", 
                        help="Skip scanning comments and use existing mentions tables")
    parser.add_argument("-ss", "--skip-sentiment", action="store_true",
                        help="Skip running SentiStrength and use existing sentiment scores")
    args = parser.parse_args()

    conn = get_db()
    create_ticker_tables(conn)

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