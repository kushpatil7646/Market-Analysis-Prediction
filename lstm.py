import os
# ----------------- CONFIG -----------------
FORCE_CPU = False   # set True to force CPU (debugging / cluster mismatch)
PLOT_DPI = 300      # high resolution for plots
RANDOM_SEED = 42    # for reproducibility
MODEL_PATH = "best_model.keras"  # Updated to .keras format
# ------------------------------------------

if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from io import BytesIO
import base64

# ---------- Generate More Realistic Dummy Sales Data ----------
def generate_dummy_data():
    np.random.seed(RANDOM_SEED)
    periods = 1000  # Double the data points for smoother plots
    dates = pd.date_range(start="2018-01-01", periods=periods, freq="M")
    
    # More realistic demand pattern with seasonality and trend
    time = np.arange(periods)
    trend = 0.15 * time
    seasonality = 25 * np.sin(2 * np.pi * time / 12)  # yearly seasonality
    noise = np.random.normal(0, 8, periods)
    demand = np.abs(200 + trend + seasonality + noise)
    
    # Price with some correlation to demand
    price = 50 + 0.2 * demand + np.random.normal(0, 1.5, periods)
    
    # Export index with longer-term trends
    export_index = 100 + 0.5 * time + 10 * np.sin(2 * np.pi * time / 24) + np.random.normal(0, 3, periods)
    
    return pd.DataFrame({
        "date": dates,
        "demand": np.round(demand, 2),
        "price": np.round(price, 2),
        "export_index": np.round(export_index, 2)
    })

CSV_PATH = "sales_data.csv"
if not os.path.exists(CSV_PATH):
    df_dummy = generate_dummy_data()
    df_dummy.to_csv(CSV_PATH, index=False)
    print(f"Dummy sales data created: {CSV_PATH}")

# ---------- Load Data ----------
df = pd.read_csv(CSV_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------- Feature Engineering ----------
features = ["demand", "price", "export_index"]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])

LOOKBACK = 12
def create_sequences(X, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i+lookback])
        ys.append(X[i+lookback, 0])  # demand is first col
    return np.array(Xs), np.array(ys)

X, y = create_sequences(scaled, LOOKBACK)

# Train/test split with more sophisticated approach
test_size = int(len(X) * 0.2)
val_size = int(len(X) * 0.1)

X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

X_train, X_val = X_train[:-val_size], X_train[-val_size:]
y_train, y_val = y_train[:-val_size], y_train[-val_size:]

# Memory optimization
X_train = np.ascontiguousarray(X_train.astype("float32"))
X_val = np.ascontiguousarray(X_val.astype("float32"))
X_test = np.ascontiguousarray(X_test.astype("float32"))
y_train = np.ascontiguousarray(y_train.astype("float32").reshape(-1, 1))
y_val = np.ascontiguousarray(y_val.astype("float32").reshape(-1, 1))
y_test = np.ascontiguousarray(y_test.astype("float32").reshape(-1, 1))

# ---------- Model with Improved Architecture ----------
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),  # Fixed this line
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss="mse",
              metrics=["mae"])

# Callbacks
es = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True, verbose=1)
mc = ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=[es, mc],
    verbose=1
)

# Load the best model
best_model = tf.keras.models.load_model(MODEL_PATH)  # Changed to models.load_model

# ---------- Evaluation ----------
def invert_target(scaled_target):
    dummy = np.zeros((len(scaled_target), scaled.shape[1]))
    dummy[:, 0] = scaled_target.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

y_pred = best_model.predict(X_test)
y_test_inv = invert_target(y_test)
y_pred_inv = invert_target(y_pred)

mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# ---------- Enhanced Visualization ----------
def create_plot_download_link(fig, filename):
    """Create download link for plot"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=PLOT_DPI, bbox_inches="tight")
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/png;base64,{plot_data}" download="{filename}">Download {filename}</a>'

# Plot 1: Training History
plt.figure(figsize=(12, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Training History")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
train_history_plot = create_plot_download_link(plt, "training_history.png")
plt.close()

# Plot 2: Actual vs Predicted
test_dates = df["date"][-len(y_test):]
plt.figure(figsize=(14, 7))
plt.plot(test_dates, y_test_inv, label="Actual Demand", marker="o", markersize=4)
plt.plot(test_dates, y_pred_inv, label="Predicted Demand", linestyle="--", marker="x", markersize=5)
plt.title("Demand Forecast: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
forecast_plot = create_plot_download_link(plt, "demand_forecast.png")
plt.close()

# Plot 3: Feature Correlations
plt.figure(figsize=(10, 6))
corr = df[features].corr()
plt.matshow(corr, fignum=1, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(features)), features, rotation=45)
plt.yticks(range(len(features)), features)
plt.title("Feature Correlation Matrix")
correlation_plot = create_plot_download_link(plt, "feature_correlations.png")
plt.close()

# Display results and download links
print("\n=== Model Evaluation ===")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

print("\n=== Plot Download Links ===")
print("1. Training History:", train_history_plot)
print("2. Demand Forecast:", forecast_plot)
print("3. Feature Correlations:", correlation_plot)

# Save the scaler for future use
import joblib
joblib.dump(scaler, 'scaler.save')