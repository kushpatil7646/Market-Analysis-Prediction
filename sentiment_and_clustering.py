# sentiment_and_clustering.py
import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------- CONFIG ----------------
CSV_PATH = "processed_competitor_data.csv"                   # input reviews file (1 column: review_text)
OUT_DIR = "data_competitor_analysis"       # output folder
PLOT_DPI = 300
MODEL_NAME = "all-MiniLM-L6-v2"            # SBERT model
NUM_CLUSTERS = 2                           # choose 2-6 depending on your data
RANDOM_SEED = 42
MAX_EXAMPLES_PER_CLUSTER_IN_EXCEL = 10
# ----------------------------------------

# Reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Generate Dummy Reviews if missing ----------
if not os.path.exists(CSV_PATH):
    reviews = [
        "Absolutely love this turmeric powder, very fresh!",
        "Too spicy for my taste, but quality is good.",
        "Great aroma and flavor, will buy again.",
        "Not worth the price, packaging was damaged.",
        "Perfect blend for curries, my family loved it.",
        "Mild taste, could be stronger.",
        "Excellent quality cumin, enhanced my dish!",
        "The masala felt stale; disappointed.",
        "Packaging is premium and the spices are fragrant.",
        "Average product; expected better flavor."
    ]
    pd.DataFrame({"review_text": reviews}).to_csv(CSV_PATH, index=False)
    print(f"Dummy reviews data created: {CSV_PATH}")

# ---------- Load Data ----------
df = pd.read_csv(CSV_PATH)
if "review_text" not in df.columns:
    raise ValueError("Input CSV must have a 'review_text' column.")
df["review_text"] = df["review_text"].fillna("").astype(str)

# ---------- Sentiment Analysis ----------
analyzer = SentimentIntensityAnalyzer()
df["sentiment_score"] = df["review_text"].apply(lambda t: analyzer.polarity_scores(t)["compound"])

# ---------- Embeddings ----------
print("Loading sentence-transformer model...")
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(
    df["review_text"].tolist(),
    batch_size=64,
    show_progress_bar=False,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# ---------- Clustering ----------
km = KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_SEED, n_init="auto")
df["cluster"] = km.fit_predict(embeddings)

# ---------- PCA for 2D visualization ----------
pca = PCA(n_components=2, random_state=RANDOM_SEED)
coords = pca.fit_transform(embeddings)
df["pc1"] = coords[:, 0]
df["pc2"] = coords[:, 1]

# ---------- Save Plot ----------
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df["pc1"], df["pc2"], c=df["cluster"], cmap="viridis")
plt.title("Customer Review Clusters (SBERT + KMeans)")
plt.xlabel("PC1"); plt.ylabel("PC2")
handles, labels = scatter.legend_elements()
plt.legend(handles, [f"Cluster {i}" for i in range(NUM_CLUSTERS)], title="Clusters", loc="best")
plot_path = os.path.join(OUT_DIR, "review_clusters.png")
plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
plt.close()

# ---------- Cluster Summary ----------
summary = (
    df.groupby("cluster")
      .agg(
          reviews_count=("review_text", "count"),
          avg_sentiment=("sentiment_score", "mean")
      )
      .reset_index()
      .sort_values("cluster")
)

# representative examples per cluster
examples = (
    df.sort_values(["cluster", "sentiment_score"], ascending=[True, False])
      .groupby("cluster")
      .head(MAX_EXAMPLES_PER_CLUSTER_IN_EXCEL)
      .copy()
)

# ---------- Save Enriched CSV ----------
enriched_csv = os.path.join(OUT_DIR, "reviews_scored_clustered.csv")
df.to_csv(enriched_csv, index=False)

# ---------- Save Excel (summary + data + examples) ----------
excel_path = os.path.join(OUT_DIR, "reviews_sentiment_clustering.xlsx")
try:
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        summary.to_excel(writer, sheet_name="cluster_summary", index=False)
        # format avg sentiment to 2 decimals
        wb = writer.book
        ws = writer.sheets["cluster_summary"]
        num_fmt = wb.add_format({"num_format": "0.00"})
        ws.set_column("A:A", 10)
        ws.set_column("B:B", 15)
        ws.set_column("C:C", 15, num_fmt)

        examples[["cluster", "sentiment_score", "review_text"]].to_excel(
            writer, sheet_name="top_examples", index=False
        )
        writer.sheets["top_examples"].set_column("A:A", 10)
        writer.sheets["top_examples"].set_column("B:B", 15, num_fmt)
        writer.sheets["top_examples"].set_column("C:C", 100)

        df.to_excel(writer, sheet_name="full_data", index=False)
        writer.sheets["full_data"].set_column("A:Z", 20)
except ModuleNotFoundError:
    # fallback if xlsxwriter isn't installed
    df.to_excel(excel_path, index=False)
    print("Note: Install 'xlsxwriter' for the full multi-sheet Excel export.")

# ---------- Console Summary ----------
print("\n--- Cluster Summary ---")
for _, row in summary.iterrows():
    print(f"Cluster {int(row['cluster'])}: "
          f"count={int(row['reviews_count'])}, "
          f"avg_sentiment={row['avg_sentiment']:.2f}")

print("\nSaved files:")
print(f" - Enriched CSV: {os.path.abspath(enriched_csv)}")
print(f" - Excel report: {os.path.abspath(excel_path)}")
print(f" - Cluster plot: {os.path.abspath(plot_path)}")
