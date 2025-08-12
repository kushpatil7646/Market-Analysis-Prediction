import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download all required NLTK resources
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading stopwords...")
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading wordnet...")
        nltk.download('wordnet', quiet=True)

    # Handle punkt_tab specifically
    

    try:
        nltk.data.find('tokenizers/punkt/PY3/english.pickle')
    except LookupError:
        print("Downloading punkt language data...")
        nltk.download('punkt_tab', quiet=True)

print("Downloading required NLTK data...")
download_nltk_resources()
print("NLTK data download complete.\n")

# Create output directory if it doesn't exist
output_dir = "competitor_analysis_output"
os.makedirs(output_dir, exist_ok=True)

CSV_PATH = "competitor_data.csv"

# ---------- Generate Enhanced Dummy Competitor Data ----------
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
    
    df_dummy = pd.DataFrame({
        "product_text": products,
        "price": prices,
        "brand": brands
    })
    df_dummy.to_csv(CSV_PATH, index=False)
    print(f"Dummy competitor data created: {CSV_PATH}")

# ---------- Load Data ----------
df = pd.read_csv(CSV_PATH)

# ---------- Text Preprocessing ----------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r"[^a-zA-Z ]", " ", text).lower()
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]

df["tokens"] = df["product_text"].apply(preprocess)

# ---------- LDA Topic Modeling ----------
dictionary = corpora.Dictionary(df["tokens"])
corpus = [dictionary.doc2bow(tokens) for tokens in df["tokens"]]

lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=4,
    passes=15,
    random_state=42,
    alpha="auto"
)

# Save LDA topics to file
topics_file = os.path.join(output_dir, "lda_topics.txt")
with open(topics_file, "w") as f:
    f.write("LDA Topics:\n")
    for i, topic in lda_model.show_topics(num_topics=4, num_words=5, formatted=True):
        f.write(f"Topic {i}: {topic}\n")
print(f"LDA topics saved to: {topics_file}")

# ---------- TF-IDF Analysis ----------
tfidf = TfidfVectorizer(max_features=15, stop_words=list(stop_words))
tfidf_matrix = tfidf.fit_transform(df["product_text"])
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)

# Save TF-IDF keywords to file
keywords_file = os.path.join(output_dir, "tfidf_keywords.txt")
with open(keywords_file, "w") as f:
    f.write("TF-IDF Keywords:\n")
    f.write(", ".join(tfidf.get_feature_names_out()))
print(f"TF-IDF keywords saved to: {keywords_file}")

# ---------- Generate Visualizations ----------
# 1. Word Cloud for Product Descriptions
all_text = " ".join(df["product_text"])
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="viridis",
    max_words=50
).generate(all_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Competitor Product Keywords")
wordcloud_file = os.path.join(output_dir, "wordcloud.png")
plt.savefig(wordcloud_file, dpi=300, bbox_inches="tight")
plt.close()
print(f"Word cloud saved to: {wordcloud_file}")

# 2. Price Distribution by Brand
plt.figure(figsize=(10, 6))
df.boxplot(column="price", by="brand", grid=False)
plt.title("Price Distribution by Brand")
plt.suptitle("")
plt.xlabel("Brand")
plt.ylabel("Price ($)")
price_dist_file = os.path.join(output_dir, "price_distribution.png")
plt.savefig(price_dist_file, dpi=300, bbox_inches="tight")
plt.close()
print(f"Price distribution plot saved to: {price_dist_file}")

# 3. Feature Correlations
plt.figure(figsize=(10, 6))
# Convert brand to numerical values for correlation
df_numeric = df.copy()
df_numeric["brand_code"] = pd.factorize(df["brand"])[0]
corr = df_numeric[["price", "brand_code"]].corr()
plt.matshow(corr, fignum=1, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr.columns)), ["price", "brand"], rotation=45)
plt.yticks(range(len(corr.columns)), ["price", "brand"])
plt.title("Feature Correlation Matrix")
corr_file = os.path.join(output_dir, "feature_correlations.png")
plt.savefig(corr_file, dpi=300, bbox_inches="tight")
plt.close()
print(f"Feature correlation plot saved to: {corr_file}")

# ---------- Save Processed Data ----------
processed_file = os.path.join(output_dir, "processed_competitor_data.csv")
df.to_csv(processed_file, index=False)
print(f"Processed data saved to: {processed_file}")

# ---------- Display Summary ----------
print("\n=== Analysis Complete ===")
print(f"1. Found {len(lda_model.show_topics())} distinct product categories")
print(f"2. Average Product Price: ${df['price'].mean():.2f}")
print("\nAll output files saved to:", os.path.abspath(output_dir))