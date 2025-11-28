import os
import json
import argparse
from datetime import datetime, timezone
import praw
from dotenv import load_dotenv
import re
import sqlite3

DB_PATH = "reddit.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def save_posts_and_comments(conn, subreddit, posts):
    now_ts = int(datetime.now(timezone.utc).timestamp())
    cur = conn.cursor()

    for post in posts:
        # enrich post with subreddit + fetched timestamp
        post_id = post["id"]
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
            post.get("created_utc"),
            post.get("url"),
            post.get("permalink"),
            post.get("link_flair_text"),
            now_ts,
        ))

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

    conn.commit()


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
        user_agent = os.environ.get("REDDIT_USER_AGENT", "Dissertation: Reddit Stock Sentiment by u/Donnie_Sucklong")

        if not client_id:
                raise SystemExit("Set REDDIT_CLIENT_ID in reddit.env or environment")

        return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent, check_for_async=False)

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
        for submission in fetcher(limit=post_limit):
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
                except Exception:
                        # skip a post if something goes wrong
                        continue

        return results

def main():
        parser = argparse.ArgumentParser(description="Fetch posts and comments from subreddits")
        parser.add_argument("-sub","--subreddits", type=str, default="stocks,investing",
                                                help="Comma-separated subreddit names (default: stocks,investing)")
        parser.add_argument("-p","--posts", type=int, default=4, help="Posts per subreddit (default: 50)")
        parser.add_argument("-c","--comments", type=int, default=100, help="Comments per post (default: 100)")
        parser.add_argument("-s","--sort", type=str, default="new", choices=["new", "hot", "top", "rising"],
                                                help="Which listing to use (default: new)")
        args = parser.parse_args()

        reddit = build_reddit()

        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")

        for sub in [s.strip() for s in args.subreddits.split(",") if s.strip()]:
                print(f"Fetching /r/{sub} ({args.posts} posts, up to {args.comments} comments/post) ...")
                items = fetch_subreddit(
                reddit,
                sub,
                post_limit=args.posts,
                comment_limit_per_post=args.comments,
                sort=args.sort
                )
                print(f"Fetched {len(items)} posts from /r/{sub}, saving to database...")
                save_posts_and_comments(conn, sub, items)
                print(f"Saved /r/{sub} batch to {DB_PATH}")

        conn.close()


if __name__ == "__main__":
        main()

