# 📊 Dutch Tweets Sentiment Analysis

A professional Natural Language Processing (NLP) pipeline that filters Dutch language tweets and performs sentiment analysis using `TextBlob-NL`. The project automates cleaning, language classification, polarity scoring, and generates beautiful data visualizations.

---

## ✨ Features & Core Capabilities

*   **🧹 Text Preprocessing & Cleaning**: Automates removal of URLs, usernames, hashtags, special characters, and converts text to lowercase.
*   **🇳🇱 Language Detection**: Utilizes `langdetect` to isolate Dutch-language tweets (`nl`) and drop missing values.
*   **📈 Sentiment Classification**: Scores polarities using Dutch-trained NLP rules (`TextBlob-NL` PatternTagger and PatternAnalyzer) to label tweets as Positive, Neutral, or Negative.
*   **🎨 Dynamic Visualizations**: Saves 3 outputs in the `output/` folder on every run:
    1.  **Sentiment Distribution**: A bar plot showing the ratio of sentiments.
    2.  **Word Cloud**: Visual representation of the most frequent words in Dutch tweets.
    3.  **Sentiment Over Time**: A rolling daily average showing the overall sentiment trend line.

---

## 📁 Project Directory Structure

```
review-sentiment-analysis/
├── requirements.txt      ← Python dependency packages
├── analysis.py           ← Main execution script containing processing & plotting logic
├── data/                 ← Place raw datasets here (git-ignored)
│   └── dutch_tweets_chunk7.json
├── output/               ← Folder where charts and word clouds are saved (git-ignored)
│   ├── sentiment_distribution.png
│   ├── word_cloud.png
│   └── sentiment_over_time.png
└── README.md             ← This documentation
```

---

## ⚡ Setup & Run Instructions (5 minutes)

Ensure you have Python 3.10+ installed.

### Step 1 — Navigate to the directory
Open a terminal and change your directory to the repository folder:
```bash
cd review-sentiment-analysis
```

### Step 2 — Create a Virtual Environment
It is highly recommended to isolate your packages using a virtual environment:
```bash
python -m venv venv
# On Windows (Command Prompt)
venv\Scripts\activate
# On Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# On MacOS/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
Install all the required NLP and data science packages:
```bash
pip install -r requirements.txt
```

### Step 4 — Place the Dataset
Make a folder named `data/` in the project root and place your raw `dutch_tweets_chunk7.json` file inside it.

### Step 5 — Run the Analysis
Launch the processing script:
```bash
python analysis.py
```
*Visualizations will be automatically generated and exported to the `output/` directory.*
