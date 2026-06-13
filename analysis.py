import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import Blobber
from textblob_nl import PatternTagger, PatternAnalyzer
from langdetect import detect

# Set paths dynamically relative to script location
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(current_dir, 'data', 'dutch_tweets_chunk7.json')
OUTPUT_DIR = os.path.join(current_dir, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

def run_analysis():
    print("Checking dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}.")
        print("Please place 'dutch_tweets_chunk7.json' in the 'data' directory.")
        return

    print("Loading data...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.json_normalize(data)[['full_text', 'created_at']]
    
    print("Preprocessing text & detecting language...")
    df['clean_text'] = df['full_text'].astype(str).apply(clean_text)
    df = df[df['clean_text'].str.len() > 20]  # Filter out very short tweets
    df['language'] = df['clean_text'].apply(lambda x: detect(x) if x else '')
    
    # Filter for Dutch tweets ('nl')
    df = df[df['language'] == 'nl'].dropna(subset=['clean_text'])
    
    if df.empty:
        print("No Dutch tweets found in dataset matching criteria.")
        return

    print("Performing sentiment analysis...")
    tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
    df['sentiment'] = df['clean_text'].apply(lambda x: tb(x).sentiment[0])
    df['sentiment_label'] = df['sentiment'].apply(
        lambda s: 'Positive' if s > 0.1 else 'Negative' if s < -0.1 else 'Neutral'
    )
    
    print("Generating distribution plot...")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='sentiment_label', data=df, palette='pastel')
    plt.title("Sentiment Distribution")
    plot_path = os.path.join(OUTPUT_DIR, 'sentiment_distribution.png')
    plt.savefig(plot_path)
    print(f"Saved distribution plot to {plot_path}")
    plt.close()
    
    print("Generating word cloud...")
    wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['clean_text']))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud")
    wc_path = os.path.join(OUTPUT_DIR, 'word_cloud.png')
    plt.savefig(wc_path)
    print(f"Saved word cloud to {wc_path}")
    plt.close()
    
    print("Generating sentiment over time trend...")
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df.set_index('created_at', inplace=True)
    df.resample('D')['sentiment'].mean().rolling(3).mean().plot(title='Sentiment Over Time', figsize=(10, 4))
    plt.grid()
    trend_path = os.path.join(OUTPUT_DIR, 'sentiment_over_time.png')
    plt.savefig(trend_path)
    print(f"Saved trend plot to {trend_path}")
    plt.close()
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    run_analysis()
