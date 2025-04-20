from flask import Flask, render_template, request
import feedparser
from datetime import datetime
from nltk import download
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

# Download necessary NLTK resources
download('vader_lexicon')
download('punkt')

# Initialize Flask app
app = Flask(__name__)

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Initialize Punkt tokenizer manually to avoid punkt_tab issue
punkt_param = PunktParameters()
tokenizer = PunktSentenceTokenizer(punkt_param)

# RSS Feeds for various categories
RSS_FEEDS = {
    "world": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "politics": "http://feeds.bbci.co.uk/news/politics/rss.xml",
    "entertainment": "http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
    "sports": "http://feeds.bbci.co.uk/sport/rss.xml",
    "technology": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "fashion": "https://www.vogue.com/rss",
    "stories": "http://feeds.bbci.co.uk/news/stories/rss.xml",
}

# Helper function to fetch and parse RSS feeds with pagination
def fetch_news(category, page=1, per_page=50):
    feed = feedparser.parse(RSS_FEEDS[category])
    news_items = feed.entries
    news_items.sort(key=lambda x: x.get("published_parsed"), reverse=True)

    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    return news_items[start_index:end_index], len(news_items)

# Summarize content using custom Punkt tokenizer
def summarize_content(content, summary_length=3):
    sentences = tokenizer.tokenize(content)
    return ' '.join(sentences[:summary_length])

# Sentiment analysis helper
def analyze_sentiment(content):
    score = sentiment_analyzer.polarity_scores(content)
    if score["compound"] >= 0.05:
        return "Positive"
    elif score["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Homepage route with pagination and sentiment filter
@app.route("/", methods=["GET", "POST"])
def index():
    selected_category = request.form.get("category", "world")
    selected_sentiment = request.form.get("sentiment", "all")
    current_page = int(request.args.get("page", 1))

    news_items, total_items = fetch_news(selected_category, current_page)
    total_pages = (total_items // 50) + (1 if total_items % 50 != 0 else 0)

    formatted_news = []

    for item in news_items:
        summary = summarize_content(item.description)
        published_time = datetime(*item.published_parsed[:6]).strftime("%Y-%m-%d %H:%M:%S") if item.published_parsed else "Unknown"
        sentiment = analyze_sentiment(summary)

        formatted_news.append({
            "title": item.title,
            "link": item.link,
            "summary": summary,
            "published": published_time,
            "sentiment": sentiment
        })

    if selected_sentiment != "all":
        formatted_news = [item for item in formatted_news if item["sentiment"] == selected_sentiment]

    return render_template(
        "index.html",
        news=formatted_news,
        categories=RSS_FEEDS.keys(),
        category=selected_category,
        sentiment=selected_sentiment,
        current_page=current_page,
        total_pages=total_pages
    )

if __name__ == "__main__":
    app.run(debug=True)

