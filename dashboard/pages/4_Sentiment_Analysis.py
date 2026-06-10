import streamlit as st
import plotly.io as pio
import pandas as pd
from pathlib import Path

def show():
    # Page Hero matching style.css variables
    st.markdown("""
    <div class="hero-section">
        <div class="hero-bg-grid"></div>
        <div class="hero-glow"></div>
        <div class="hero-eyebrow">Layer 5 Analysis</div>
        <h1 class="hero-title">
            Public <br><span class="grad">Sentiment Analysis</span>
        </h1>
        <p class="hero-sub">
            Qualitative NLP sentiment pipeline analyzing public discourse on poverty in Indonesia. 
            Using a fine-tuned IndoRoBERTa model on Twitter (X) datasets (2021-2026), we extract sentiment intensity trends and word frequencies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label="Dominant Sentiment", value="Neutral", delta="65.5% Average")
    with c2:
        st.metric(label="Negative Trend (2026)", value="45.0%", delta="Growing Negativity", delta_color="inverse")
    with c3:
        st.metric(label="NLP Engine", value="IndoRoBERTa", delta="Contextual Understanding")
    
    st.markdown("---")

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    SENTIMENT_OUTPUT = BASE_DIR / "LAYER 5 - ROBERTA SENTIMENT" / "sentiment_output"

    def load_plotly(filename):
        try:
            with open(SENTIMENT_OUTPUT / filename, 'r', encoding='utf-8') as f:
                json_str = f.read().replace('"heatmapgl"', '"heatmap"')
            fig = pio.from_json(json_str)
            
            # Apply premium dark re-theming
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0', family='Space Grotesk, sans-serif', size=11),
                title=dict(font=dict(color='#00D4FF', family='Plus Jakarta Sans', size=16)),
                legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', font=dict(color='#94A3B8')),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            
            # Axis styling
            for ax_name in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2']:
                if ax_name in fig.layout:
                    fig.layout[ax_name].gridcolor = 'rgba(30, 45, 74, 0.6)'
                    fig.layout[ax_name].linecolor = 'rgba(255, 255, 255, 0.1)'
                    fig.layout[ax_name].zerolinecolor = 'rgba(255, 255, 255, 0.1)'
                    fig.layout[ax_name].tickfont = dict(size=10, color='#94A3B8')
            
            # Colorbar / colorscale overrides
            if hasattr(fig, 'data'):
                for trace in fig.data:
                    if 'colorbar' in trace:
                        trace.colorbar.tickfont = dict(color='#E2E8F0', family='Space Grotesk')
                        trace.colorbar.title.font = dict(color='#00D4FF', family='Plus Jakarta Sans')
            
            return fig
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return None

    # Section 1: Overall Sentiment
    st.markdown("""
    <div class="section-header">
        <div class="section-title">Macro Sentiment Distribution</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    fig_overall = load_plotly("plot_sentiment_overall.json")
    if fig_overall:
        st.markdown('<div class="card"><div class="card-label">Overall Sentiment Profile</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_overall, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Section 2: Temporal & Textual Metrics
    st.markdown("""
    <div class="section-header">
        <div class="section-title">Temporal Trends & Top Words</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_month = load_plotly("plot_sentiment_per_month.json")
        fig_year = load_plotly("plot_sentiment_per_year.json")
        if fig_month:
            st.markdown('<div class="card"><div class="card-label">Sentiment Trends over Time (Monthly)</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_month, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        if fig_year:
            st.markdown('<div class="card"><div class="card-label">Sentiment Trends over Time (Yearly)</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_year, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        fig_word = load_plotly("plot_tweet_wordfreq.json")
        fig_eda = load_plotly("plot_tweet_eda.json")
        if fig_word:
            st.markdown('<div class="card"><div class="card-label">High-Frequency Discussion Words</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_word, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        if fig_eda:
            st.markdown('<div class="card"><div class="card-label">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_eda, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Section 3: Intensity & Wordcloud
    st.markdown("""
    <div class="section-header">
        <div class="section-title">Sentiment Intensity & Wordcloud</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    fig_hm = load_plotly("plot_sentiment_heatmap.json")
    if fig_hm:
        st.markdown('<div class="card"><div class="card-label">Sentiment Intensity Heatmap</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_hm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    wordcloud_path = SENTIMENT_OUTPUT / "plot_sentiment_wordcloud.png"
    if wordcloud_path.exists():
        st.markdown('<div class="card"><div class="card-label">Discussion Wordcloud Representation</div>', unsafe_allow_html=True)
        st.image(str(wordcloud_path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    engagement_path = SENTIMENT_OUTPUT / "plot_sentiment_engagement.png"
    if engagement_path.exists():
        st.markdown('<div class="card"><div class="card-label">Sentiment Engagement Distribution</div>', unsafe_allow_html=True)
        st.image(str(engagement_path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Section 4: Raw Dataset View
    st.markdown("""
    <div class="section-header">
        <div class="section-title">Social Media Datasets</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">Processed Twitter Sentiment Data</div>', unsafe_allow_html=True)
    try:
        df_tweets = pd.read_csv(SENTIMENT_OUTPUT / "output_sentiment_per_tweet.csv")
        st.dataframe(df_tweets, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load raw dataset: {e}")
    st.markdown("</div>", unsafe_allow_html=True)


    probdist_path = SENTIMENT_OUTPUT / "plot_sentiment_probdist.png"
    if probdist_path.exists():
        st.markdown('<div class="card"><div class="card-label">Sentiment Probability Distribution</div>', unsafe_allow_html=True)
        st.image(str(probdist_path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

