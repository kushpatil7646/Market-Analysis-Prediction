import os
import re
import zipfile
from datetime import datetime

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# =========================
# Config
# =========================
OUTPUT_DIR = "data_competitor_analysis"
CSV_PATH = "competitor_data.csv"
RANDOM_SEED = 42
PLOT_DPI = 300
np.random.seed(RANDOM_SEED)

# =========================
# Ensure NLTK data
# =========================
def download_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

print("Checking NLTK resources...")
download_nltk_resources()

# =========================
# Prepare output folders
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# =========================
# Dummy data (if none exists)
# =========================
if not os.path.exists(CSV_PATH):
    products = [
        "Organic Turmeric Powder - Rich color and earthy taste, sourced from Kerala. Best for immunity boosting.",
        "Premium Garam Masala Blend - Bold aroma, perfect for curries and meat dishes. Contains 12 spices.",
        "Pure Red Chilli Powder - Fiery heat with vibrant red color. Imported from Andhra Pradesh.",
        "Authentic Cumin Seeds - Warm, nutty flavor for Indian dishes. Helps digestion.",
        "Coriander Powder - Freshly ground, subtle citrus notes. Perfect for marinades.",
        "Black Pepper Whole - Pungent and sharp flavor. Kerala special.",
        "Cardamom Green Pods - Aromatic and sweet. Used in desserts and tea.",
        "Mustard Seeds Yellow - Pungent flavor for pickles and tempering.",
        "Fenugreek Leaves - Bitter-sweet taste. Used in curry and paratha.",
        "Asafoetida Powder - Strong aroma. Essential for lentil dishes."
    ]
    prices = [12.99, 8.99, 5.99, 4.99, 6.99, 9.99, 15.99, 3.99, 7.99, 6.49]
    brands = ["SpiceMaster", "SpiceCraft", "SpiceMaster", "PureSpice", "SpiceCraft",
              "PureSpice", "SpiceMaster", "PureSpice", "SpiceCraft", "PureSpice"]
    df_dummy = pd.DataFrame({"product_text": products, "price": prices, "brand": brands})
    df_dummy.to_csv(CSV_PATH, index=False)
    print(f"Dummy competitor data created: {CSV_PATH}")

# =========================
# Load data
# =========================
df = pd.read_csv(CSV_PATH)
df["product_text"] = df["product_text"].astype(str)
df["brand"] = df["brand"].astype(str)
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# =========================
# Preprocess text
# =========================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r"[^a-zA-Z ]", " ", text).lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t) > 2 and t not in stop_words]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return lemmas

df["tokens"] = df["product_text"].apply(preprocess)
df["tokens_str"] = df["tokens"].apply(lambda x: " ".join(x))

# =========================
# LDA Topic Modeling
# =========================
dictionary = corpora.Dictionary(df["tokens"])
corpus = [dictionary.doc2bow(tokens) for tokens in df["tokens"]]

# Use at most number of docs or 1 topic if data very small
NUM_TOPICS = min(4, max(1, len(df)))
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    passes=15,
    random_state=RANDOM_SEED,
    alpha="auto"
)

# Top words per topic
TOP_N_WORDS = 7
lda_topics_list = []
for i in range(NUM_TOPICS):
    words = lda_model.show_topic(i, TOP_N_WORDS)
    lda_topics_list.append({
        "topic": i,
        "top_words": ", ".join([w for w, _ in words])
    })
lda_topics_df = pd.DataFrame(lda_topics_list)

# Dominant topic per document
def get_dominant_topic(bow):
    topic_probs = lda_model.get_document_topics(bow, minimum_probability=0.0)
    topic_probs = sorted(topic_probs, key=lambda x: x[1], reverse=True)
    return topic_probs[0][0], topic_probs[0][1]

dominant_topics = [get_dominant_topic(bow) for bow in corpus]
df["dominant_topic"] = [t for t, p in dominant_topics]
df["topic_prob"] = [p for t, p in dominant_topics]

# =========================
# TF-IDF
# =========================
tfidf = TfidfVectorizer(max_features=50, stop_words=list(stop_words))
tfidf_matrix = tfidf.fit_transform(df["product_text"])
tfidf_terms = tfidf.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_terms)

# Average TF-IDF per term (for plotting)
tfidf_avg = tfidf_df.mean(axis=0).sort_values(ascending=False)
top_15 = tfidf_avg.head(15)

# =========================
# PLOTS (save all to PLOTS_DIR)
# =========================

# 1) Word Cloud
all_text = " ".join(df["product_text"])
wc = WordCloud(width=1200, height=600, background_color="white", colormap="viridis", max_words=100)
wc.generate(all_text)
plt.figure(figsize=(14, 7))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Competitor Product Keywords")
wordcloud_path = os.path.join(PLOTS_DIR, "wordcloud.png")
plt.savefig(wordcloud_path, dpi=PLOT_DPI, bbox_inches="tight")
plt.close()

# 2) TF-IDF Top Terms (bar)
plt.figure(figsize=(12, 6))
top_15[::-1].plot(kind="barh")  # horizontal bar for readability
plt.title("Top 15 TF-IDF Terms")
plt.xlabel("Average TF-IDF Score")
plt.tight_layout()
tfidf_bar_path = os.path.join(PLOTS_DIR, "tfidf_top_terms.png")
plt.savefig(tfidf_bar_path, dpi=PLOT_DPI, bbox_inches="tight")
plt.close()

# 3) LDA Topics – bar for each topic’s word weights
for i in range(NUM_TOPICS):
    words_weights = lda_model.show_topic(i, TOP_N_WORDS)
    words = [w for w, _ in words_weights]
    weights = [float(s) for _, s in words_weights]
    plt.figure(figsize=(10, 5))
    plt.bar(words, weights)
    plt.title(f"LDA Topic {i} – Top Words")
    plt.ylabel("Weight")
    plt.xticks(rotation=30)
    plt.tight_layout()
    path_i = os.path.join(PLOTS_DIR, f"lda_topic_{i}_top_words.png")
    plt.savefig(path_i, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

# 4) Price Distribution by Brand (box)
plt.figure(figsize=(10, 6))
df.boxplot(column="price", by="brand", grid=False)
plt.title("Price Distribution by Brand")
plt.suptitle("")
plt.xlabel("Brand")
plt.ylabel("Price ($)")
price_box_path = os.path.join(PLOTS_DIR, "price_distribution_by_brand.png")
plt.savefig(price_box_path, dpi=PLOT_DPI, bbox_inches="tight")
plt.close()

# 5) Overall Price Histogram
plt.figure(figsize=(10, 6))
plt.hist(df["price"].dropna(), bins=8, edgecolor="black")
plt.title("Overall Price Distribution")
plt.xlabel("Price ($)")
plt.ylabel("Count")
price_hist_path = os.path.join(PLOTS_DIR, "price_histogram.png")
plt.savefig(price_hist_path, dpi=PLOT_DPI, bbox_inches="tight")
plt.close()

# 6) Brand Means with Error Bars (std)
brand_stats = df.groupby("brand")["price"].agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
plt.figure(figsize=(10, 6))
plt.errorbar(brand_stats["brand"], brand_stats["mean"], yerr=brand_stats["std"], fmt="o", capsize=5)
plt.title("Average Price by Brand (±1 std)")
plt.xlabel("Brand")
plt.ylabel("Price ($)")
plt.grid(True, axis="y")
plt.tight_layout()
brand_mean_err_path = os.path.join(PLOTS_DIR, "brand_avg_price_errorbars.png")
plt.savefig(brand_mean_err_path, dpi=PLOT_DPI, bbox_inches="tight")
plt.close()

# 7) Feature Correlation (price vs brand_code)
df_num = df.copy()
df_num["brand_code"] = pd.factorize(df_num["brand"])[0]
corr = df_num[["price", "brand_code"]].corr()
plt.figure(figsize=(6, 5))
plt.matshow(corr, fignum=1, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr.columns)), ["price", "brand"], rotation=45)
plt.yticks(range(len(corr.columns)), ["price", "brand"])
plt.title("Feature Correlation Matrix")
corr_path = os.path.join(PLOTS_DIR, "feature_correlation.png")
plt.savefig(corr_path, dpi=PLOT_DPI, bbox_inches="tight")
plt.close()

# 8) Topic Mix by Brand (stacked bar)
topic_counts = df.pivot_table(index="brand", columns="dominant_topic", values="product_text", aggfunc="count", fill_value=0)
topic_share = topic_counts.div(topic_counts.sum(axis=1), axis=0)
topic_share.plot(kind="bar", stacked=True, figsize=(10, 6))
plt.title("Topic Mix by Brand (share)")
plt.xlabel("Brand"); plt.ylabel("Share")
plt.legend(title="Topic", bbox_to_anchor=(1.03, 1), loc="upper left")
plt.tight_layout()
topic_mix_path = os.path.join(PLOTS_DIR, "topic_mix_by_brand.png")
plt.savefig(topic_mix_path, dpi=PLOT_DPI, bbox_inches="tight")
plt.close()

# =========================
# Excel workbook
# =========================
excel_path = os.path.join(OUTPUT_DIR, "competitor_analysis.xlsx")

# Build a tidy topic assignment table
topic_assignments = df[["product_text", "brand", "price", "dominant_topic", "topic_prob"]].copy()
topic_assignments = topic_assignments.sort_values(["brand", "dominant_topic", "topic_prob"], ascending=[True, True, False])

# Summary sheet
summary = {
    "run_timestamp": [datetime.now().isoformat()],
    "n_products": [len(df)],
    "n_brands": [df["brand"].nunique()],
    "avg_price": [round(df["price"].mean(), 2)],
    "min_price": [round(df["price"].min(), 2)],
    "max_price": [round(df["price"].max(), 2)],
    "num_topics": [NUM_TOPICS]
}
summary_df = pd.DataFrame(summary)

with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    df.to_excel(writer, index=False, sheet_name="raw_data")
    df[["product_text", "brand", "price", "tokens_str"]].to_excel(writer, index=False, sheet_name="tokens")
    tfidf_df.to_excel(writer, index=False, sheet_name="tfidf_matrix")
    pd.DataFrame({"term": tfidf_terms, "avg_tfidf": tfidf_avg.reindex(tfidf_terms).values}).to_excel(writer, index=False, sheet_name="tfidf_terms")
    lda_topics_df.to_excel(writer, index=False, sheet_name="lda_topics")
    topic_assignments.to_excel(writer, index=False, sheet_name="topic_assignments")
    brand_stats.to_excel(writer, index=False, sheet_name="brand_stats")
    summary_df.to_excel(writer, index=False, sheet_name="summary")

print(f"Excel saved to: {os.path.abspath(excel_path)}")

# =========================
# Save text files (optional)
# =========================
with open(os.path.join(OUTPUT_DIR, "lda_topics.txt"), "w") as f:
    f.write("LDA Topics (top words):\n")
    for _, row in lda_topics_df.iterrows():
        f.write(f"Topic {row['topic']}: {row['top_words']}\n")

with open(os.path.join(OUTPUT_DIR, "tfidf_keywords.txt"), "w") as f:
    f.write("Top 15 TF-IDF Keywords:\n")
    for term, score in top_15.items():
        f.write(f"{term}: {score:.4f}\n")

# =========================
# Simple HTML index for “executive viewing”
# =========================
html_path = os.path.join(OUTPUT_DIR, "index.html")
plots_rel = [os.path.join("plots", os.path.basename(p)) for p in [
    wordcloud_path, tfidf_bar_path, price_box_path, price_hist_path, brand_mean_err_path, corr_path, topic_mix_path
]] + [os.path.join("plots", f"lda_topic_{i}_top_words.png") for i in range(NUM_TOPICS)]

html = [
    "<html><head><meta charset='utf-8'><title>Competitor Analysis</title>",
    "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;} img{max-width:100%;height:auto;} .plot{margin:24px 0;} h2{margin-top:36px;}</style>",
    "</head><body>",
    "<h1>Competitor Analysis – Executive Summary</h1>",
    f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    "<h2>Key Files</h2>",
    f"<ul><li><a href='competitor_analysis.xlsx'>Download Excel Workbook</a></li>",
    f"<li><a href='lda_topics.txt'>LDA Topics (text)</a></li>",
    f"<li><a href='tfidf_keywords.txt'>TF-IDF Keywords (text)</a></li></ul>",
    "<h2>Visuals</h2>"
]
for p in plots_rel:
    html.append(f"<div class='plot'><img src='{p}'><p><em>{os.path.basename(p)}</em></p></div>")
html.append("</body></html>")
with open(html_path, "w", encoding="utf-8") as f:
    f.write("\n".join(html))

print(f"HTML index saved to: {os.path.abspath(html_path)}")

# =========================
# Zip the whole folder for easy download/share
# =========================
zip_path = OUTPUT_DIR + ".zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(OUTPUT_DIR):
        for fn in files:
            full = os.path.join(root, fn)
            arc = os.path.relpath(full, start=os.path.dirname(OUTPUT_DIR))
            z.write(full, arcname=arc)
print(f"Zipped bundle created: {os.path.abspath(zip_path)}")

# =========================
# Console Summary
# =========================
print("\n=== Analysis Complete ===")
print(f"- Topics discovered: {NUM_TOPICS}")
print(f"- Average price: ${df['price'].mean():.2f}")
print(f"- Output folder: {os.path.abspath(OUTPUT_DIR)}")
print(f"- ZIP bundle:    {os.path.abspath(zip_path)}")
