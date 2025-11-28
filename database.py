import sqlite3

DB_PATH = "reddit.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        id TEXT PRIMARY KEY,
        subreddit TEXT,
        title TEXT,
        selftext TEXT,
        score INTEGER,
        upvote_ratio REAL,
        num_comments INTEGER,
        created_utc INTEGER,
        url TEXT,
        permalink TEXT,
        flair TEXT,
        fetched_at_utc INTEGER
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        id TEXT PRIMARY KEY,
        post_id TEXT,
        parent_id TEXT,
        author TEXT,
        body TEXT,
        score INTEGER,
        created_utc INTEGER,
        fetched_at_utc INTEGER,
        FOREIGN KEY(post_id) REFERENCES posts(id)
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS active_posts (
        post_id TEXT PRIMARY KEY,
        subreddit TEXT,
        created_utc INTEGER,
        last_checked_utc INTEGER,
        no_new_comments_runs INTEGER DEFAULT 0
    );
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()