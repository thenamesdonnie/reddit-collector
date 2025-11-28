import os
import json
import argparse
from datetime import datetime
import praw
from dotenv import load_dotenv
import re

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
                                        "author": c.author.name if getattr(c, "author", None) else None,
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
        parser.add_argument("--outdir", type=str, default="data", help="Output directory (default: data)")
        args = parser.parse_args()

        reddit = build_reddit()
        # Create output directory if it doesn't exist
        os.makedirs(args.outdir, exist_ok=True)

        for sub in [s.strip() for s in args.subreddits.split(",") if s.strip()]:
                print(f"Fetching /r/{sub} ({args.posts} posts, up to {args.comments} comments/post) ...")
                items = fetch_subreddit(reddit, sub, post_limit=args.posts, comment_limit_per_post=args.comments, sort=args.sort)
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                out_path = os.path.join(args.outdir, f"{sub}_{ts}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                        json.dump({"subreddit": sub, "fetched_at_utc": ts, "posts": items}, f, ensure_ascii=False, indent=2)
                print(f"Wrote {len(items)} posts to {out_path}")


if __name__ == "__main__":
        main()

