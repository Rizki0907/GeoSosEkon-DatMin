import streamlit as st
import plotly.io as pio
import pandas as pd
from pathlib import Path

def show():
    # Page Hero
    st.markdown("""
    <div class="hero-container">
        <div class="hero-subtitle">LAYER 5 ANALYSIS</div>
        <div class="hero-title">Public Sentiment Analysis</div>
        <div class="hero-desc">
            Qualitative NLP sentiment pipeline analyzing public discourse on poverty in Indonesia. 
            Using a fine-tuned IndoRoBERTa model on Twitter (X) datasets (2021-2026), we extract public sentiment distributions and trends.
        </div>
    </div>
    """, unsafe_allow_html=True)

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
                font=dict(color='#E2E8F0', family='Inter'),
                title=dict(font=dict(color='#06B6D4', family='Outfit', size=16)),
                legend=dict(font=dict(color='#94A3B8')),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            
            # Axis styling
            for ax_name in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2']:
                if ax_name in fig.layout:
                    fig.layout[ax_name].gridcolor = 'rgba(255, 255, 255, 0.05)'
                    fig.layout[ax_name].linecolor = 'rgba(255, 255, 255, 0.1)'
                    fig.layout[ax_name].zerolinecolor = 'rgba(255, 255, 255, 0.1)'
            
            # Colorbar / colorscale overrides
            if hasattr(fig, 'data'):
                for trace in fig.data:
                    if 'colorbar' in trace:
                        trace.colorbar.tickfont = dict(color='#E2E8F0', family='Inter')
                        trace.colorbar.title.font = dict(color='#06B6D4', family='Outfit')
            
            return fig
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return None

    fig_overall = load_plotly("plot_sentiment_overall.json")
    if fig_overall:
        st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>Overall Sentiment Distribution</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_overall, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_month = load_plotly("plot_sentiment_per_month.json")
        if fig_month:
            st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>Sentiment Trend by Month</h3>", unsafe_allow_html=True)
            st.plotly_chart(fig_month, use_container_width=True)

    with col2:
        fig_word = load_plotly("plot_tweet_wordfreq.json")
        if fig_word:
            st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>Top Frequent Words</h3>", unsafe_allow_html=True)
            st.plotly_chart(fig_word, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    
    fig_hm = load_plotly("plot_sentiment_heatmap.json")
    if fig_hm:
        st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>Sentiment Intensity Heatmap</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC;'>Sentiment Wordcloud</h3>", unsafe_allow_html=True)
    
    wordcloud_path = SENTIMENT_OUTPUT / "plot_sentiment_wordcloud.png"
    if wordcloud_path.exists():
        # Display the wordcloud inside a nice styled glass panel
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255,255,255,0.04); padding: 15px; border-radius: 12px; display: flex; justify-content: center; align-items: center; overflow: hidden; margin-bottom: 2rem;">
        """, unsafe_allow_html=True)
        st.image(str(wordcloud_path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-family: \"Outfit\"; color: #F8FAFC; margin-bottom: 1rem;'>Raw Sentiment Dataset</h3>", unsafe_allow_html=True)
    try:
        df_tweets = pd.read_csv(SENTIMENT_OUTPUT / "output_sentiment_per_tweet.csv")
        st.dataframe(df_tweets, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load raw dataset: {e}")
