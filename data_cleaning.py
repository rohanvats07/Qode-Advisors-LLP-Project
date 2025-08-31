import pandas as pd
import numpy as np
import re
from datetime import datetime
import unicodedata

def clean_and_process_tweets(input_file, output_format='parquet'):
    """
    Clean and normalize Twitter data with deduplication and Unicode handling
    """
    print("Loading data...")
    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
    
    initial_count = len(df)
    print(f"Initial data count: {initial_count}")
    
    # Step 1: Clean and normalize data
    # Clean usernames - strip whitespace and convert to lowercase
    df['username'] = df['username'].str.strip().str.lower()
    
    # Clean content - strip whitespace and handle Unicode
    df['content'] = df['content'].str.strip()
    df['content'] = df['content'].apply(normalize_unicode)
    
    # Clean mentions and hashtags
    df['mentions'] = df['mentions'].fillna('').str.strip()
    df['hashtags'] = df['hashtags'].fillna('').str.strip()

    # Convert timestamp to pandas datetime with UTC timezone
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    
    # Drop rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])
    cleaned_count = len(df)
    
    # Step 2: Data deduplication
    print("Removing duplicates...")
    df = df.drop_duplicates(subset=['content', 'timestamp'])
    deduplicated_count = len(df)
    
    # Step 3: Add data quality metrics
    df['content_length'] = df['content'].str.len()
    df['total_engagement'] = df['retweets'] + df['likes'] + df['replies']
    
    # Step 4: Sort by timestamp (newest first)
    df = df.sort_values('timestamp', ascending=False)
    
    # Step 5: Save processed data
    print("Saving processed data...")
    if output_format == 'parquet':
        try:
            output_file = 'cleaned_stock_market_tweets.parquet'
            df.to_parquet(output_file, index=False, engine='pyarrow')
        except ImportError:
            print("Parquet engine not available, saving as CSV instead...")
            output_file = 'cleaned_stock_market_tweets.csv'
            df.to_csv(output_file, index=False)
    else:
        output_file = 'cleaned_stock_market_tweets.csv'
        df.to_csv(output_file, index=False)
    
    stats = {
        'initial_count': initial_count,
        'after_cleaning': cleaned_count,
        'final_count': deduplicated_count,
        'duplicates_removed': cleaned_count - deduplicated_count,
        'output_file': output_file,
        'data_quality': {
            'avg_content_length': df['content_length'].mean(),
            'avg_engagement': df['total_engagement'].mean(),
            'unique_users': df['username'].nunique(),
            'date_range': {
                'earliest': df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'latest': df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    }
    
    print(f"Processing complete! Final count: {deduplicated_count}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    return stats, df

def normalize_unicode(text):
    """
        Normalize Unicode characters for consistent storage
        Handles Indian language content and special characters
    """
    if pd.isna(text):
        return text
    
    text = unicodedata.normalize('NFC', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Handle common Unicode issues
    text = text.replace('\u200d', '')  # Zero-width joiner
    text = text.replace('\u200c', '')  # Zero-width non-joiner
    text = text.replace('\ufeff', '')  # Byte order mark
    
    return text.strip()

def create_storage_schema():
    """
    Define optimal storage schema for parquet format
    """
    schema = {
        'username': 'string',
        'timestamp': 'datetime64[ns, UTC]',
        'content': 'string',
        'retweets': 'int32',
        'likes': 'int32',
        'replies': 'int32',
        'mentions': 'string',
        'hashtags': 'string',
        'content_length': 'int32',
        'total_engagement': 'int32'
    }
    return schema

if __name__ == "__main__":
    input_file = 'stock_market_tweets.xlsx'
    
    stats, processed_df = clean_and_process_tweets(input_file, output_format='csv')
    
    print("\n<---------------- Data Processing Statistics ----------------->")
    print(f"Initial tweets: {stats['initial_count']}")
    print(f"After cleaning: {stats['after_cleaning']}")
    print(f"Final tweets: {stats['final_count']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"Unique users: {stats['data_quality']['unique_users']}")
    print(f"Average content length: {stats['data_quality']['avg_content_length']:.1f} characters")
    print(f"Average engagement: {stats['data_quality']['avg_engagement']:.1f}")
    print(f"Date range: {stats['data_quality']['date_range']['earliest']} to {stats['data_quality']['date_range']['latest']}")
    print(f"Output saved to: {stats['output_file']}")

    print("\n<------------------ Sample Processed Data ------------------>")
    print(processed_df[['username', 'timestamp', 'content', 'total_engagement']].head())
