#!/usr/bin/env python
# coding: utf-8

# # Indonesian Poverty Tweet Sentiment Analysis with RoBERTa (2021-2026)
# 
# This notebook analyzes sentiment in Indonesian tweets related to poverty using an Indonesian RoBERTa-based sentiment classifier. The results provide national-level qualitative context and are not attributed to provinces.

# ## Library Imports

# In[ ]:


get_ipython().system('pip install transformers torch datasets sentencepiece -q')
get_ipython().system('pip install pandas numpy matplotlib seaborn scikit-learn wordcloud -q')
get_ipython().system('pip install nltk Sastrawi -q')


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import re
import string
from pathlib import Path
import warnings
from collections import Counter
from datetime import datetime

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from sklearn.metrics import classification_report, confusion_matrix

import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path.cwd().parent
TWEET_DATA_PATH = PROJECT_ROOT / 'kemiskinan_tweets_2021_2026' / 'kemiskinan_tweets_2021_2026.csv'
OUTPUT_DIR = Path('sentiment_output')
OUTPUT_DIR.mkdir(exist_ok=True)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13

device = 0 if torch.cuda.is_available() else -1
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

print(f'Device: {"CUDA" if device == 0 else "CPU"} ({device_name})')
print(f'PyTorch version: {torch.__version__}')
import transformers
print(f'Transformers version: {transformers.__version__}')


# ## Tweet Data Loading and Exploration

# In[10]:


df_raw = pd.read_csv(TWEET_DATA_PATH)
df_raw = df_raw.rename(columns={
    'waktu': 'timestamp',
    'teks': 'text',
    'tanggal': 'date',
    'tahun': 'year',
    'bulan': 'month',
    'bahasa': 'language'
})

print('Tweet Dataset Information')
print(f'Number of tweets     : {len(df_raw)}')
print(f'Columns     : {df_raw.shape[1]}')
print(f'Columns            : {df_raw.columns.tolist()}')
print(f'Year range    : {df_raw["year"].min()} - {df_raw["year"].max()}')
print(f'Unique languages      : {df_raw["language"].unique()}')
print()
print('First 5 Rows')
df_raw[['timestamp', 'text', 'like', 'retweet', 'year', 'month']].head()


# In[11]:


print('Tweet Distribution by Year')
per_year = df_raw['year'].value_counts().sort_index()
print(per_year.to_string())

print()
print('Engagement Statistics')
print(df_raw[['like', 'retweet', 'reply', 'quote']].describe().round(2).to_string())

print()
print('Missing Values')
missing = df_raw.isnull().sum()
print(missing[missing > 0].to_string())


# In[12]:


fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
per_year.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black', alpha=0.8)
ax.set_title('Number of Tweets by Year')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Tweets')
ax.tick_params(axis='x', rotation=0)
ax.grid(axis='y', alpha=0.4)
for bar, val in zip(ax.patches, per_year.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha='center', va='bottom', fontsize=9)

ax = axes[0, 1]
per_month = df_raw['month'].value_counts().sort_index()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
per_month.index = [month_names[m-1] for m in per_month.index]
per_month.plot(kind='bar', ax=ax, color='#4472C4', edgecolor='black', alpha=0.8)
ax.set_title('Tweet Distribution by Month (All Years)')
ax.set_xlabel('Month')
ax.set_ylabel('Number of Tweets')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.4)

ax = axes[1, 0]
likes_nonzero = df_raw[df_raw['like'] > 0]['like']
ax.hist(np.log1p(likes_nonzero), bins=30, color='#70AD47', edgecolor='black', alpha=0.8)
ax.set_title(f'Log(likes + 1) Distribution (n={len(likes_nonzero)} tweets with likes > 0)')
ax.set_xlabel('log(likes + 1)')
ax.set_ylabel('Frequency')
ax.grid(alpha=0.4)

ax = axes[1, 1]
df_raw['length_text'] = df_raw['text'].astype(str).str.len()
ax.hist(df_raw['length_text'], bins=30, color='#FF7F00', edgecolor='black', alpha=0.8)
ax.set_title('Tweet Text Length Distribution')
ax.set_xlabel('Length (characters)')
ax.set_ylabel('Frequency')
ax.grid(alpha=0.4)
ax.axvline(df_raw['length_text'].median(), color='red', linestyle='--',
           label=f'Median = {df_raw["length_text"].median():.0f}')
ax.legend()

plt.suptitle('Indonesian Poverty Tweet Data Exploration 2021-2026',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plot_tweet_eda.png', bbox_inches='tight')
plt.show()


# ## Text Preprocessing
# 
# The preprocessing pipeline removes URLs, mentions, and noisy whitespace while preserving enough text structure for the transformer model.

# In[13]:


def preprocess_tweet(text):
    if pd.isna(text):
        return ''
    text = str(text)

    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower()
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[^\w\s.,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_for_model(text):
    if pd.isna(text):
        return ''
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:512]

df = df_raw.copy()
df['text_clean'] = df['text'].apply(preprocess_tweet)
df['text_model'] = df['text'].apply(preprocess_for_model)

df['length_clean'] = df['text_clean'].str.split().str.len()
df['length_model'] = df['text_model'].str.len()

df_valid = df[df['length_clean'] >= 3].copy().reset_index(drop=True)

print(f'Initial tweets       : {len(df)}')
print(f'Valid tweets (>=3 words): {len(df_valid)}')
print(f'Dropped tweets    : {len(df) - len(df_valid)}')
print()
print('Preprocessing examples')
for i in range(3):
    print(f'\nOriginal : {df_valid["text"].iloc[i][:120]}')
    print(f'Cleaned  : {df_valid["text_clean"].iloc[i][:120]}')
    print(f'For model: {df_valid["text_model"].iloc[i][:120]}')


# In[14]:


factory_sw = StopWordRemoverFactory()
stopwords_id = set(factory_sw.get_stop_words())

custom_stop = {'poverty', 'miskin', 'penduduk', 'persen', 'indonesia',
               'bps', 'year', 'juta', 'ribu', 'orang', 'juga', 'sehingga',
               'namun', 'dalam', 'yang', 'dengan', 'untuk', 'pada', 'dari',
               'dan', 'ini', 'itu', 'ke', 'di', 'tidak', 'ada'}
stopwords_id.update(custom_stop)

all_words = []
for text in df_valid['text_clean']:
    words = text.split()
    words = [w for w in words if w not in stopwords_id and len(w) > 2]
    all_words.extend(words)

word_freq = Counter(all_words)
top_words = pd.DataFrame(word_freq.most_common(30), columns=['word', 'frequency'])

print('Top 30 Most Frequent Words')
print(top_words.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(range(len(top_words)), top_words['frequency'],
        color='steelblue', edgecolor='black', alpha=0.8)
ax.set_yticks(range(len(top_words)))
ax.set_yticklabels(top_words['word'])
ax.invert_yaxis()
ax.set_xlabel('Frequency')
ax.set_title('Top 30 Most Frequent Words in Poverty Tweets 2021-2026')
ax.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plot_tweet_wordfreq.png', bbox_inches='tight')
plt.show()


# ## RoBERTa Model Initialization
# 
# The notebook attempts to load an Indonesian sentiment classifier and falls back to the next available model option when needed.

# In[15]:


MODEL_OPTIONS = [
    'w11wo/indonesian-roberta-base-sentiment-classifier',
    'ayameRushia51/text-classification-id-sentiment-analysis-smsa',
]

LABEL_MAP = {
    'positive': 'positive',
    'positif': 'positive',
    'POSITIVE': 'positive',
    'LABEL_2': 'positive',
    'neutral': 'neutral',
    'neutral': 'neutral',
    'NEUTRAL': 'neutral',
    'LABEL_1': 'neutral',
    'negative': 'negative',
    'negatif': 'negative',
    'NEGATIVE': 'negative',
    'LABEL_0': 'negative',
}

loaded_model_name = None
sentiment_pipeline = None

for model_name in MODEL_OPTIONS:
    try:
        print(f'Trying to load model: {model_name}')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        sentiment_pipeline = pipeline(
            'text-classification',
            model=model,
            tokenizer=tokenizer,
            device=device,
            return_all_scores=True,
            truncation=True,
            max_length=512
        )
        loaded_model_name = model_name
        print(f'Loaded model: {model_name}')
        print(f'Label model: {model.config.id2label}')
        break
    except Exception as e:
        print(f'Failed: {e}')
        continue

if sentiment_pipeline is None:
    raise RuntimeError('All models failed to load. Check the internet connection.')


# In[16]:


test_texts = [
    'Poverty in Indonesia has declined significantly this year, which is encouraging.',
    'The number of poor residents continues to increase, which is very concerning.',
    'BPS released the March 2023 poverty rate at 9.36 percent.',
    'Government policy has not been able to address poverty effectively.',
    'Social assistance programs help poor households access basic needs.'
]

print('Sentiment Model Test')

for text in test_texts:
    results = sentiment_pipeline(text)
    if isinstance(results[0], dict):
        scores = {results[0]['label']: results[0]['score']}
    else:
        scores = {r['label']: r['score'] for r in results[0]}

    best_label = max(scores, key=scores.get)
    best_label_mapped = LABEL_MAP.get(best_label, best_label)
    best_score = scores[best_label]

    print(f'\nText   : {text[:80]}')
    print(f'Label  : {best_label_mapped} (conf: {best_score:.4f})')
    print(f'Scores : { {LABEL_MAP.get(k, k): round(v, 4) for k, v in scores.items()} }')


# ## Sentiment Inference

# In[17]:


import time

def run_sentiment_batch(texts, pipeline_fn, batch_size=32, label_map=None):
    """Run sentiment inference in batches."""
    if label_map is None:
        label_map = {}

    results_all = []
    n = len(texts)

    start_time = time.time()

    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        batch_results = pipeline_fn(batch)

        for raw_result in batch_results:

            if isinstance(raw_result, dict):
                scores = {raw_result['label']: raw_result['score']}
            else:
                scores = {r['label']: r['score'] for r in raw_result}

            best_label = max(scores, key=scores.get)
            best_label_mapped = label_map.get(best_label, best_label)
            best_score = scores[best_label]

            score_pos = max([v for k, v in scores.items()
                             if label_map.get(k, k) == 'positive'], default=0)
            score_neg = max([v for k, v in scores.items()
                             if label_map.get(k, k) == 'negative'], default=0)
            score_neu = max([v for k, v in scores.items()
                             if label_map.get(k, k) == 'neutral'], default=0)

            results_all.append({
                'sentiment': best_label_mapped,
                'confidence': best_score,
                'score_positive': score_pos,
                'score_negative': score_neg,
                'score_neutral': score_neu,
            })

        if (i // batch_size + 1) % 5 == 0 or i + batch_size >= n:
            elapsed = time.time() - start_time
            done = min(i + batch_size, n)
            pct = done / n * 100
            print(f'Progress: {done}/{n} ({pct:.1f}%) | Waktu: {elapsed:.1f}s')

    return results_all

print(f'Running inference on {len(df_valid)} tweet...')
print(f'Batch size: 32')
print(f'Device: {"GPU" if device == 0 else "CPU"}')
print()

texts_for_model = df_valid['text_model'].tolist()

sentiment_results = run_sentiment_batch(
    texts_for_model,
    sentiment_pipeline,
    batch_size=32,
    label_map=LABEL_MAP
)

df_sent = pd.DataFrame(sentiment_results)
df_result = pd.concat([df_valid.reset_index(drop=True), df_sent], axis=1)

print(f'\nInference completed. Total results: {len(df_result)}')
print()

print('Sentiment Distribution')
sent_dist = df_result['sentiment'].value_counts()
print(sent_dist.to_string())

print()
print('Percentage')
print((sent_dist / len(df_result) * 100).round(2).astype(str).add('%').to_string())


# In[18]:


print('Confidence Score Statistics')
print(df_result['confidence'].describe().round(4).to_string())

threshold = 0.5
high_conf = df_result[df_result['confidence'] >= threshold]
low_conf = df_result[df_result['confidence'] < threshold]

print(f'\nConfidence >= {threshold}: {len(high_conf)} tweet ({len(high_conf)/len(df_result)*100:.1f}%)')
print(f'Confidence < {threshold}: {len(low_conf)} tweet ({len(low_conf)/len(df_result)*100:.1f}%)')

print('\nAverage Confidence by Sentiment')
print(df_result.groupby('sentiment')['confidence'].agg(['mean', 'std', 'min', 'max']).round(4).to_string())


# ## Sentiment Visualization

# In[19]:


SENT_COLORS = {
    'positive': '#70AD47',
    'neutral': '#4472C4',
    'negative': '#C00000'
}

sent_order = ['positive', 'neutral', 'negative']
sent_counts = df_result['sentiment'].value_counts().reindex(sent_order, fill_value=0)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

ax = axes[0]
wedge_colors = [SENT_COLORS.get(s, 'gray') for s in sent_order]
wedges, texts, autotexts = ax.pie(
    sent_counts.values,
    labels=sent_counts.index,
    colors=wedge_colors,
    autopct='%1.1f%%',
    startangle=140,
    wedgeprops=dict(edgecolor='white', linewidth=1.5)
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight('bold')
ax.set_title('Sentiment Distribution Overall\n(2021-2026)')

ax = axes[1]
bar_colors = [SENT_COLORS.get(s, 'gray') for s in sent_counts.index]
bars = ax.bar(sent_counts.index, sent_counts.values, color=bar_colors, edgecolor='black', alpha=0.85)
for bar, val in zip(bars, sent_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Number of Tweets by Sentiment')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Number of Tweets')
ax.grid(axis='y', alpha=0.4)

ax = axes[2]
for sent in sent_order:
    sub = df_result[df_result['sentiment'] == sent]['confidence']
    if len(sub) > 0:
        ax.hist(sub, bins=20, alpha=0.6, color=SENT_COLORS[sent],
                label=f'{sent} (n={len(sub)})', edgecolor='black', linewidth=0.3)
ax.set_xlabel('Confidence Score')
ax.set_ylabel('Frequency')
ax.set_title('Distribution Confidence Score by Sentiment')
ax.legend()
ax.grid(alpha=0.4)

plt.suptitle('Indonesian Poverty Tweet Sentiment Analysis Results',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plot_sentiment_overall.png', bbox_inches='tight')
plt.show()

print('Sentiment Distribution')
for s in sent_order:
    n = sent_counts.get(s, 0)
    pct = n / len(df_result) * 100
    avg_conf = df_result[df_result['sentiment'] == s]['confidence'].mean()
    print(f'{s:10s}: {n:4d} tweet ({pct:5.1f}%), average confidence = {avg_conf:.4f}')


# In[20]:


sent_by_year = df_result.groupby(['year', 'sentiment']).size().unstack(fill_value=0)
sent_by_year_pct = sent_by_year.div(sent_by_year.sum(axis=1), axis=0) * 100

print('Sentiment Distribution by Year (Number of)')
print(sent_by_year.to_string())
print()
print('Sentiment Distribution by Year (%)')
print(sent_by_year_pct.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
bottom = np.zeros(len(sent_by_year_pct))
for sent in sent_order:
    if sent in sent_by_year_pct.columns:
        vals = sent_by_year_pct[sent].values
        ax.bar(sent_by_year_pct.index, vals, bottom=bottom,
               label=sent, color=SENT_COLORS[sent], alpha=0.85, edgecolor='white')
        for i, (y, b, v) in enumerate(zip(sent_by_year_pct.index, bottom, vals)):
            if v > 5:
                ax.text(y, b + v/2, f'{v:.1f}%',
                        ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        bottom += vals
ax.set_xlabel('Year')
ax.set_ylabel('Percentage (%)')
ax.set_title('Sentiment Proportion by Year')
ax.legend(loc='upper right')
ax.set_xticks(sent_by_year_pct.index)
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3)
n
ax = axes[1]
for sent in sent_order:
    if sent in sent_by_year_pct.columns:
        ax.plot(sent_by_year_pct.index, sent_by_year_pct[sent],
                'o-', color=SENT_COLORS[sent], label=sent,
                linewidth=2, markersize=7)
        for x, y in zip(sent_by_year_pct.index, sent_by_year_pct[sent]):
            ax.annotate(f'{y:.1f}%', xy=(x, y),
                        xytext=(0, 8), textcoords='offset points',
                        fontsize=8, ha='center', color=SENT_COLORS[sent])
ax.set_xlabel('Year')
ax.set_ylabel('Percentage (%)')
ax.set_title('Trend of Percentage Sentiment by Year')
ax.legend()
ax.set_xticks(sent_by_year_pct.index)
ax.grid(alpha=0.4)

plt.suptitle('Poverty Tweet Sentiment Dynamics by Year', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plot_sentiment_per_year.png', bbox_inches='tight')
plt.show()


# In[21]:


sent_by_month = df_result.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
sent_by_month_pct = sent_by_month.div(sent_by_month.sum(axis=1), axis=0) * 100

month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, ax = plt.subplots(figsize=(14, 6))
for sent in sent_order:
    if sent in sent_by_month_pct.columns:
        sub = sent_by_month_pct[sent].reindex(range(1, 13), fill_value=0)
        ax.plot(range(1, 13), sub.values, 'o-',
                color=SENT_COLORS[sent], label=sent, linewidth=2, markersize=7)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels)
ax.set_xlabel('Month')
ax.set_ylabel('Percentage (%)')
ax.set_title('Trend of Sentiment by Month (Aggregate 2021-2026)')
ax.legend()
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plot_sentiment_per_month.png', bbox_inches='tight')
plt.show()

print('Sentiment Distribution by Month (%)')
print(sent_by_month_pct.round(2).to_string())


# In[22]:


fig, axes = plt.subplots(1, len(sent_order), figsize=(17, 6))

for idx, sent in enumerate(sent_order):
    pivot = df_result[df_result['sentiment'] == sent].groupby(['year', 'month']).size().unstack(fill_value=0)
    pivot.columns = [month_labels[m-1] for m in pivot.columns]

    ax = axes[idx]
    cmap = {
        'positive': 'Greens',
        'neutral': 'Blues',
        'negative': 'Reds'
    }[sent]
    sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt='d',
                linewidths=0.5, linecolor='gray',
                cbar_kws={'shrink': 0.8})
    ax.set_title(f'Tweet {sent.upper()}\nby Year-Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Year' if idx == 0 else '')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Heatmap Number of Tweets by Sentiment (Year x Month)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plot_sentiment_heatmap.png', bbox_inches='tight')
plt.show()


# In[23]:


print('Engagement by Sentiment')
engagement_stats = df_result.groupby('sentiment')[['like', 'retweet', 'reply', 'quote']].agg(['mean', 'median', 'sum'])
print(engagement_stats.round(2).to_string())

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
metrics_eng = ['like', 'retweet', 'reply', 'quote']
metric_labels = ['Likes', 'Retweet', 'Reply', 'Quote']

for i, (metric, label) in enumerate(zip(metrics_eng, metric_labels)):
    ax = axes[i // 2][i % 2]
    data_by_sent = [df_result[df_result['sentiment'] == s][metric].values for s in sent_order]
    bp = ax.boxplot(data_by_sent, labels=sent_order, patch_artist=True,
                    showfliers=False)
    for patch, sent in zip(bp['boxes'], sent_order):
        patch.set_facecolor(SENT_COLORS[sent])
        patch.set_alpha(0.7)
    ax.set_title(f'Distribution {label} by Sentiment')
    ax.set_ylabel(label)
    ax.grid(axis='y', alpha=0.4)

plt.suptitle('Engagement Tweet by Sentiment', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plot_sentiment_engagement.png', bbox_inches='tight')
plt.show()


# In[24]:


try:
    from wordcloud import WordCloud

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for idx, sent in enumerate(sent_order):
        ax = axes[idx]
        texts_sent = df_result[df_result['sentiment'] == sent]['text_clean'].tolist()
        combined_text = ' '.join(texts_sent)

        words_filtered = [w for w in combined_text.split() if w not in stopwords_id and len(w) > 2]
        combined_filtered = ' '.join(words_filtered)

        if len(combined_filtered.strip()) > 10:
            wc = WordCloud(
                width=500, height=300,
                background_color='white',
                colormap={'positive': 'Greens', 'neutral': 'Blues', 'negative': 'Reds'}[sent],
                max_words=80,
                collocations=False
            ).generate(combined_filtered)
            ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'WordCloud: {sent.upper()}\n(n={len(texts_sent)} tweet)',
                     color=SENT_COLORS[sent], fontweight='bold')

    plt.suptitle('WordCloud Tweet by Sentiment', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot_sentiment_wordcloud.png', bbox_inches='tight')
    plt.show()
except ImportError:
    print('WordCloud is not available. Skipping this step.')


# In[25]:


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, sent in enumerate(sent_order):
    score_col = f'score_{sent}'
    ax = axes[idx]
    ax.hist(df_result[score_col], bins=30, color=SENT_COLORS[sent],
            edgecolor='black', alpha=0.8, density=True)
    mean_val = df_result[score_col].mean()
    ax.axvline(mean_val, color='black', linestyle='--',
               label=f'Mean = {mean_val:.3f}')
    ax.set_xlabel(f'Probability {sent}')
    ax.set_ylabel('Density')
    ax.set_title(f'Probability Score Distribution\n{sent.upper()}')
    ax.legend()
    ax.grid(alpha=0.4)

plt.suptitle('Sentiment Probability Score Distribution', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plot_sentiment_probdist.png', bbox_inches='tight')
plt.show()

print('Probability Score Statistics')
print(df_result[['score_positive', 'score_neutral', 'score_negative']].describe().round(4).to_string())


# ## Save Output Files

# In[26]:


cols_output = [
    'tweet_id', 'timestamp', 'year', 'month', 'date',
    'username', 'text', 'text_clean',
    'like', 'retweet', 'reply', 'quote',
    'sentiment', 'confidence',
    'score_positive', 'score_neutral', 'score_negative'
]
cols_output_valid = [c for c in cols_output if c in df_result.columns]

df_out1 = df_result[cols_output_valid].copy()
df_out1 = df_out1.sort_values(['year', 'month']).reset_index(drop=True)
df_out1.to_csv(OUTPUT_DIR / 'output_sentiment_per_tweet.csv', index=False)
print(f'Saved: output_sentiment_per_tweet.csv ({len(df_out1)} rows)')
print(df_out1.head(5).to_string(index=False))
print()

agg_yearmonth = df_result.groupby(['year', 'month', 'sentiment']).agg(
    tweet_count=('sentiment', 'count'),
    avg_confidence=('confidence', 'mean'),
    avg_like=('like', 'mean'),
    avg_retweet=('retweet', 'mean')
).reset_index()

agg_yearmonth.to_csv(OUTPUT_DIR / 'output_sentiment_monthly_aggregate.csv', index=False)
print(f'Saved: output_sentiment_monthly_aggregate.csv ({len(agg_yearmonth)} rows)')
print(agg_yearmonth.head(12).round(4).to_string(index=False))
print()

agg_year = df_result.groupby(['year', 'sentiment']).agg(
    tweet_count=('sentiment', 'count'),
    avg_confidence=('confidence', 'mean'),
    total_like=('like', 'sum'),
    total_retweet=('retweet', 'sum')
).reset_index()

total_per_year = df_result.groupby('year').size().rename('total_tweets')
agg_year = agg_year.merge(total_per_year, on='year')
agg_year['sentiment_percent'] = (agg_year['tweet_count'] / agg_year['total_tweets'] * 100).round(2)

agg_year.to_csv(OUTPUT_DIR / 'output_sentiment_yearly_aggregate.csv', index=False)
print(f'Saved: output_sentiment_yearly_aggregate.csv ({len(agg_year)} rows)')
print(agg_year.round(4).to_string(index=False))


# In[27]:


print('ROBERTA SENTIMENT ANALYSIS SUMMARY')
print(f'Model            : {loaded_model_name}')
print(f'Device           : {"GPU" if device == 0 else "CPU"}')
print(f'Total tweets      : {len(df_result)}')
print(f'Time range    : {df_result["year"].min()} - {df_result["year"].max()}')
print()
print('Sentiment Distribution:')
for s in sent_order:
    n = len(df_result[df_result['sentiment'] == s])
    pct = n / len(df_result) * 100
    avg_conf = df_result[df_result['sentiment'] == s]['confidence'].mean()
    print(f'  {s:10s}: {n:4d} tweet ({pct:5.1f}%), avg confidence = {avg_conf:.4f}')
print()
print('CSV Outputs:')
print('  1. output_sentiment_per_tweet.csv         -> Sentiment by tweet')
print('  2. output_sentiment_monthly_aggregate.csv   -> Year-month aggregate')
print('  3. output_sentiment_yearly_aggregate.csv   -> Year aggregate')

