# 📊 Spice Market Analysis & Forecasting

## Overview
This project performs **data-driven market analysis** and **demand forecasting** for spice products using:
- **LSTM-based Time Series Forecasting**
- **NLP-based Competitor Analysis**
- **Customer Review Sentiment Clustering**

The workflow covers:
1. **Demand Forecasting** → Predict next 6 months of product demand using pricing & export index trends.
2. **Competitor Analysis** → Identify product categories, pricing gaps, and keyword opportunities.
3. **Review Sentiment & Clustering** → Group customer feedback into actionable sentiment-based clusters.

---

## 🔮 1. Time-Series Demand Forecasting

### **Approach**
- **Model:** LSTM (Long Short-Term Memory) neural network
- **Data:** 1,000+ months of synthetic demand, pricing, and export index data
- **Features:** 
  - Demand (target)
  - Price
  - Export Index
  - 12-month rolling lookback window
- **Performance:**
  - **MAE:** ~8.62
  - **RMSE:** ~10.54
  - **MAPE:** ~2.54%
- **Benefit:** Detects seasonal patterns & long-term growth trends.

### **Outputs**
| Metric | Value |
|--------|-------|
| MAE    | 8.62  |
| RMSE   | 10.54 |
| MAPE   | 2.54% |

#### **Training History**
![Training History](data_competitor_analysis/training_history.png)

#### **Forecast Plot**
![Demand Forecast](data_competitor_analysis/demand_forecast.png)

---

## 🏷 2. Competitor Product & Price Analysis

### **Approach**
- **NLP Techniques:** Tokenization, Lemmatization, TF-IDF, LDA Topic Modeling
- **Data:** 100+ competitor spice product descriptions
- **Findings:**
  - **4** product categories identified (via LDA)
  - **15** high-impact keywords for SEO & marketing
  - **20%+** price gap opportunities in premium categories

### **Outputs**
#### **Word Cloud of Keywords**
![Word Cloud](data_competitor_analysis/wordcloud.png)

#### **Price Distribution by Brand**
![Price Distribution](data_competitor_analysis/price_distribution.png)

#### **Feature Correlations**
![Feature Correlation Matrix](data_competitor_analysis/feature_correlations.png)

---

## 💬 3. Customer Review Sentiment & Clustering

### **Approach**
- **Sentiment Analysis:** VaderSentiment for polarity scoring
- **Text Embeddings:** Sentence-BERT (`all-MiniLM-L6-v2`)
- **Clustering:** KMeans + PCA for dimensionality reduction
- **Goal:** Identify common themes and sentiment in customer feedback

### **Cluster Labels (Example)**
| Cluster | Theme                          | Avg Sentiment |
|---------|--------------------------------|---------------|
| 0       | Positive Product Experience    | 0.82          |
| 1       | Negative or Critical Feedback  | -0.45         |

### **Cluster Visualization**
![Review Clusters](data_competitor_analysis/review_clusters.png)

---

## 📂 Output Files

| File | Description |
|------|-------------|
| `reviews_sentiment_clustering.xlsx` | Multi-sheet report: cluster summary, top examples, full data |
| `reviews_scored_clustered.csv` | Reviews with sentiment & cluster assignments |
| `training_history.png` | LSTM training loss curve |
| `demand_forecast.png` | Predicted vs actual demand |
| `wordcloud.png` | Competitor product keyword cloud |
| `price_distribution.png` | Brand price comparison |
| `feature_correlations.png` | Correlation heatmap |

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Install dependencies
pip install -r requirements.txt

# Run forecasting
python demand_forecasting.py

# Run competitor analysis
python competitor_analysis.py

# Run sentiment & clustering
python sentiment_and_clustering.py
