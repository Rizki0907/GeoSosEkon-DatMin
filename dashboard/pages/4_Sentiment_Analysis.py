import streamlit as st
import plotly.io as pio
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Public Sentiment", layout="wide")

st.markdown("<h2 style='text-align: center;'>💬 Public Sentiment Analysis (RoBERTa)</h2>", unsafe_allow_html=True)
st.markdown("---")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SENTIMENT_OUTPUT = BASE_DIR / "LAYER 5 - ROBERTA SENTIMENT" / "sentiment_output"

def load_plotly(filename):
    try:
        with open(SENTIMENT_OUTPUT / filename, 'r', encoding='utf-8') as f:
            json_str = f.read().replace('"heatmapgl"', '"heatmap"')
        return pio.from_json(json_str)
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None

fig_overall = load_plotly("plot_sentiment_overall.json")
if fig_overall:
    st.markdown("### 📊 Overall Sentiment Distribution")
    st.plotly_chart(fig_overall, use_container_width=True)

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    fig_month = load_plotly("plot_sentiment_per_month.json")
    if fig_month:
        st.markdown("### 📅 Trend of Sentiment by Month")
        st.plotly_chart(fig_month, use_container_width=True)

with col2:
    fig_word = load_plotly("plot_tweet_wordfreq.json")
    if fig_word:
        st.markdown("### 🔠 Top Frequent Words")
        st.plotly_chart(fig_word, use_container_width=True)

st.markdown("---")
fig_hm = load_plotly("plot_sentiment_heatmap.json")
if fig_hm:
    st.markdown("### 🔥 Sentiment Heatmap")
    st.plotly_chart(fig_hm, use_container_width=True)

st.markdown("---")
st.markdown("### ☁️ Sentiment Wordcloud")
wordcloud_path = SENTIMENT_OUTPUT / "plot_sentiment_wordcloud.png"
if wordcloud_path.exists():
    st.image(str(wordcloud_path), use_column_width=True)

st.markdown("---")
st.markdown("### 📝 Raw Sentiment Dataset")
try:
    df_tweets = pd.read_csv(SENTIMENT_OUTPUT / "output_sentiment_per_tweet.csv")
    st.dataframe(df_tweets, use_container_width=True)
except Exception as e:
    st.error(f"Could not load raw dataset: {e}")
