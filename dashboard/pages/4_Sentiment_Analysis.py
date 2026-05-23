import streamlit as st
import plotly.io as pio
from pathlib import Path

st.set_page_config(page_title="Public Sentiment", layout="wide")

st.markdown("<h2 style='text-align: center;'>💬 Public Sentiment Analysis (RoBERTa)</h2>", unsafe_allow_html=True)
st.markdown("---")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SENTIMENT_OUTPUT = BASE_DIR / "LAYER 5 - ROBERTA SENTIMENT" / "sentiment_output"

def load_plotly(filename):
    try:
        return pio.read_json(SENTIMENT_OUTPUT / filename)
    except Exception as e:
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
