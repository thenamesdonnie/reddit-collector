import os
import json
import argparse
from datetime import datetime, timezone
import praw
from dotenv import load_dotenv
import re
import sqlite3, time, logging

DB_PATH = "reddit.db"
LOG_DIR = "logs"

MAX_ACTIVE_AGE_HOURS = 24
MAX_RUNS_WITHOUT_NEW = 4

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "reddit_collector.log")),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def save_posts_and_comments(conn, subreddit, posts):
    now_ts = int(datetime.utcnow().timestamp())
    cur = conn.cursor()
    post_count = 0
    comment_count = 0

    for post in posts:
        post_id = post["id"]
        created_utc = post.get("created_utc")

        cur.execute("""
        INSERT OR IGNORE INTO posts
        (id, subreddit, title, selftext, score, upvote_ratio, num_comments,
         created_utc, url, permalink, flair, fetched_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            post_id,
            subreddit,
            post.get("title"),
            post.get("selftext"),
            post.get("score"),
            post.get("upvote_ratio"),
            post.get("num_comments"),
            created_utc,
            post.get("url"),
            post.get("permalink"),
            post.get("link_flair_text"),
            now_ts,
        ))
        if cur.rowcount:
            post_count += 1

        if created_utc is not None:
            cur.execute("""
            INSERT OR IGNORE INTO active_posts
            (post_id, subreddit, created_utc, last_checked_utc, no_new_comments_runs)
            VALUES (?, ?, ?, ?, 0)
            """, (post_id, subreddit, created_utc, now_ts))

        for c in post.get("comments", []):
            if not c.get("id"):
                continue
            cur.execute("""
            INSERT OR IGNORE INTO comments
            (id, post_id, parent_id, body, score, created_utc, fetched_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                c.get("id"),
                post_id,
                c.get("parent_id"),
                c.get("body"),
                c.get("score"),
                c.get("created_utc"),
                now_ts,
            ))
            if cur.rowcount:
                comment_count += 1

    try:
        conn.commit()
    except sqlite3.Error as e:
        logger.error("Database commit failed: %s", e)
        raise

    logger.info(
        "Saved %d new posts and %d new comments for /r/%s",
        post_count, comment_count, subreddit
    )
    return post_count, comment_count

def get_active_posts(conn, max_age_hours=MAX_ACTIVE_AGE_HOURS, max_runs_without_new=MAX_RUNS_WITHOUT_NEW):
    now_ts = int(time.time())
    min_created = now_ts - max_age_hours * 3600
    cur = conn.cursor()
    cur.execute("""
    SELECT post_id, subreddit
    FROM active_posts
    WHERE created_utc >= ?
      AND no_new_comments_runs < ?
    """, (min_created, max_runs_without_new))
    return cur.fetchall()

def refresh_comments_for_post(reddit, conn, post_id, comment_limit_per_post=100):
    now_ts = int(time.time())
    cur = conn.cursor()

    # Count comments before refresh
    before = cur.execute(
        "SELECT COUNT(*) FROM comments WHERE post_id = ?",
        (post_id,)
    ).fetchone()[0]

    # Fetch submission + comments from Reddit
    try:
        submission = reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)
        all_comments = submission.comments.list()
    except Exception as e:
        logger.warning("Failed to refresh comments for post %s: %s", post_id, e)
        return 0

    for c in all_comments[:comment_limit_per_post]:
        cid = getattr(c, "id", None)
        if not cid:
            continue

        created_utc = int(getattr(c, "created_utc", 0)) if getattr(c, "created_utc", None) else None

        cur.execute("""
        INSERT OR IGNORE INTO comments
        (id, post_id, parent_id, body, score, created_utc, fetched_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            cid,
            post_id,
            getattr(c, "parent_id", None),
            getattr(c, "body", None),
            getattr(c, "score", None),
            created_utc,
            now_ts,
        ))

    # Count comments after refresh
    after = cur.execute(
        "SELECT COUNT(*) FROM comments WHERE post_id = ?",
        (post_id,)
    ).fetchone()[0]

    new_comments = after - before

    # Update active_posts record
    if new_comments > 0:
        cur.execute("""
        UPDATE active_posts
        SET last_checked_utc = ?, no_new_comments_runs = 0
        WHERE post_id = ?
        """, (now_ts, post_id))
    else:
        cur.execute("""
        UPDATE active_posts
        SET last_checked_utc = ?, no_new_comments_runs = no_new_comments_runs + 1
        WHERE post_id = ?
        """, (now_ts, post_id))

    conn.commit()
    return new_comments

def cleanup_active_posts(conn, max_age_hours=MAX_ACTIVE_AGE_HOURS, max_runs_without_new=MAX_RUNS_WITHOUT_NEW):
    now_ts = int(time.time())
    min_created = now_ts - max_age_hours * 3600
    cur = conn.cursor()
    cur.execute("""
    DELETE FROM active_posts
    WHERE created_utc < ?
       OR no_new_comments_runs >= ?
    """, (min_created, max_runs_without_new))
    deleted = cur.rowcount
    conn.commit()
    return deleted


#!/usr/bin/env python3
"""
Fetch posts and comments from subreddits (r/stocks and r/investing by default)
Requires environment variables:
    - REDDIT_CLIENT_ID
    - REDDIT_CLIENT_SECRET
    - REDDIT_USER_AGENT
"""
#Build Reddit API client
def build_reddit():
    load_dotenv("reddit.env")
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = None
    user_agent = os.environ.get(
        "REDDIT_USER_AGENT",
        "Dissertation: Reddit Stock Sentiment by u/Donnie_Sucklong"
    )

    if not client_id:
        logger.error("Missing REDDIT_CLIENT_ID in reddit.env / environment")
        raise SystemExit("Reddit API credentials not configured")

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )
    logger.info("Reddit client initialised")
    return reddit


#Fetch posts and comments from subreddits 
def fetch_subreddit(reddit, name, post_limit=50, comment_limit_per_post=100, sort="new"):
        subreddit = reddit.subreddit(name)
        fetcher = {
                "new": subreddit.new,
                "hot": subreddit.hot,
                "top": subreddit.top,
                "rising": subreddit.rising,
        }.get(sort, subreddit.hot)

        results = []
        try:
                submissions = fetcher(limit=post_limit)
        except Exception as e:
                logger.error("Error fetching listing for /r/%s: %s", name, e)
                return results
        for submission in submissions:
                try:
                        submission_data = {
                                #Collect main post data
                                "id": submission.id,
                                "title": submission.title,
                                "selftext": submission.selftext,
                                "score": submission.score,
                                "upvote_ratio": getattr(submission, "upvote_ratio", None),
                                "num_comments": submission.num_comments,
                                "created_utc": int(submission.created_utc),
                                "url": submission.url,
                                "permalink": submission.permalink,
                                "link_flair_text": submission.link_flair_text,
                        }

                        # Expand comments (limit=0 expands top-level "MoreComments" into real comments)
                        submission.comments.replace_more(limit=0)
                        all_comments = submission.comments.list()

                        #Extract selected comments
                        comments_out = []
                        for c in all_comments[:comment_limit_per_post]:
                                comments_out.append({
                                        "id": getattr(c, "id", None),
                                        "parent_id": getattr(c, "parent_id", None),
                                        "body": getattr(c, "body", None),
                                        "score": getattr(c, "score", None),
                                        "created_utc": int(getattr(c, "created_utc", 0)) if getattr(c, "created_utc", None) else None,
                                })

                        submission_data["comments"] = comments_out
                        results.append(submission_data)
                except Exception as e:
                        logger.warning(
                                "Skipping submission %s in /r/%s due to error: %s",
                                getattr(submission, "id", "unknown"),
                                name,
                                e,
                        )
                continue
        logger.info("Fetched %d posts from /r/%s", len(results), name)
        return results

def main():
    parser = argparse.ArgumentParser(description="Fetch posts and comments from subreddits")
    parser.add_argument("-sub","--subreddits", type=str, default="stocks,investing",
                        help="Comma-separated subreddit names (default: stocks,investing)")
    parser.add_argument("-p","--posts", type=int, default=50,
                        help="Posts per subreddit (default: 50)")
    parser.add_argument("-c","--comments", type=int, default=100,
                        help="Comments per post (default: 100)")
    parser.add_argument("-s","--sort", type=str, default="new",
                        choices=["new", "hot", "top", "rising"],
                        help="Which listing to use (default: new)")
    args = parser.parse_args()

    start_time = time.time()
    total_posts = 0
    total_comments = 0

    reddit = build_reddit()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")

    try:
        # 1) Fetch new posts
        for sub in [s.strip() for s in args.subreddits.split(",") if s.strip()]:
            logger.info(
                "Fetching /r/%s (%d posts, up to %d comments/post, sort=%s)",
                sub, args.posts, args.comments, args.sort
            )
            items = fetch_subreddit(
                reddit,
                sub,
                post_limit=args.posts,
                comment_limit_per_post=args.comments,
                sort=args.sort
            )
            p_count, c_count = save_posts_and_comments(conn, sub, items)
            total_posts += p_count
            total_comments += c_count

        # 2) Refresh active posts
        active = get_active_posts(conn)
        logger.info("Refreshing comments for %d active posts", len(active))
        for post_id, subreddit in active:
            try:
                refresh_comments_for_post(reddit, conn, post_id, comment_limit_per_post=args.comments)
            except Exception as e:
                logger.warning("Error refreshing post %s: %s", post_id, e)

        # 3) Cleanup stale active posts
        deleted = cleanup_active_posts(conn)
        if deleted:
            logger.info("Removed %d stale active posts", deleted)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user, shutting down gracefully...")
    finally:
        conn.close()
        duration = time.time() - start_time
        logger.info(
            "Run finished: %d new posts, %d new comments in %.2f seconds",
            total_posts, total_comments, duration
        )



if __name__ == "__main__":
        main()

