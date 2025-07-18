# review-sentiment-analysis
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

# Paths
DATA_PATH = os.path.join('..', 'data', 'dutch_tweets_chunk7.json')
OUTPUT_DIR = os.path.join('..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.json_normalize(data)[['full_text', 'created_at']]

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

df['clean_text'] = df['full_text'].astype(str).apply(clean_text)
df = df[df['clean_text'].str.len() > 20]  # Filter out very short tweets
df['language'] = df['clean_text'].apply(lambda x: detect(x) if x else '')
df = df[df['language'] == 'nl'].dropna(subset=['clean_text'])

# Sentiment analysis
tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
df['sentiment'] = df['clean_text'].apply(lambda x: tb(x).sentiment[0])
df['sentiment_label'] = df['sentiment'].apply(
    lambda s: 'Positive' if s > 0.1 else 'Negative' if s < -0.1 else 'Neutral'
)

# Sentiment distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment_label', data=df, palette='pastel')
plt.title("Sentiment Distribution")
plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_distribution.png'))
plt.show()

# Word cloud
wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['clean_text']))
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud")
plt.savefig(os.path.join(OUTPUT_DIR, 'word_cloud.png'))
plt.show()

# Sentiment over time
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df.set_index('created_at', inplace=True)
df.resample('D')['sentiment'].mean().rolling(3).mean().plot(title='Sentiment Over Time', figsize=(10, 4))
plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_over_time.png'))
plt.show()
