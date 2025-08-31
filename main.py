import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime, timedelta

def website_scraper():
    """ Scraper using snscrape """
    hashtags = ['nifty50', 'sensex', 'intraday', 'banknifty']
    all_data = []

    # Calculate the date 24 hours ago
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    for hashtag in hashtags:
        query = f"#{hashtag} since:{yesterday}"
        tweets = []
        
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i > 500:  # limit to avoid huge data
                break
            tweets.append([tweet.date, tweet.user.username, tweet.content])
        
        df = pd.DataFrame(tweets, columns=["date", "user", "content"])
        all_data.append(df)

    # Combine all data
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv('backup_tweets.csv', index=False)
        return final_df
    else:
        print("No tweets found in the last 24 hours.")
        return pd.DataFrame(columns=["date", "user", "content"])


if __name__ == "__main__":
    df = website_scraper()
    print("Tweets saved to backup_tweets.csv")
    print(df.head())
