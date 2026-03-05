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

# ── CONFIG ────────────────────────────────────────────────────────────────────
DB_PATH            = "reddit.db"
SENTISTRENGTH_JAR  = "./sentistrength/SentiStrength.jar"
SENTISTRENGTH_DATA = "./sentistrength/SentiStrength_Data/"
BATCH_SIZE         = 500          # texts per SentiStrength call
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
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Scanning {src}s"):
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
    texts = [row[4] for row in buckets["AAPL"]]
    with open("input.txt", "w") as f:
        for text in texts:
            f.write(text.replace("\n", " ") + "\n")  # flatten to single line
        

# ── SentiStrength ─────────────────────────────────────────────────────────────

def run_sentistrength():
    subprocess.run(
        ["java", "-jar", SENTISTRENGTH_JAR, "sentidata", SENTISTRENGTH_DATA, "input", "input.csv"]
    )

def score():
    for ticker in MAG7:
        filename = f"./csvs/{ticker}_daily_sentiment.csv"
        text = ("bullish today")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "positive", "negative"])
            positive, negative = run_sentistrength(text)
            writer.writerow([text, positive, negative])
            print(f"text: {text} | positive: {positive} | negative: {negative}")

# ── Main ─────────────────────────────────────────────────────────────

def main():
    conn = get_db()
    create_ticker_tables(conn)
    posts, comments = load_and_clean(conn)
    populate_mentions(conn, posts, comments)
    log.info("Running SentiStrength scoring...")
    # for ticker in MAG7:
    #     score_ticker(conn, ticker)
    conn.close()
    log.info("Done!")

if __name__ == "__main__":
    #  main()
    run_sentistrength()