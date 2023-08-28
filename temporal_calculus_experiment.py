
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def model(x, t, theta):
    return theta[0] * x + theta[1] * t

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def compute_gradients(x, t, y_true, theta):
    y_pred = model(x, t, theta)
    error = y_true - y_pred
    dtheta1 = -2 * np.mean(error * x)
    dtheta2 = -2 * np.mean(error * t)
    dt = -2 * np.mean(error)
    return np.array([dtheta1, dtheta2, dt])

def temporal_gradient_descent(x, t, y_true, theta, alpha, beta, epochs):
    loss_history = []
    for epoch in range(epochs):
        gradients = compute_gradients(x, t, y_true, theta)
        theta[:2] = theta[:2] - alpha * gradients[:2]
        theta[1] = theta[1] - alpha * beta * gradients[2]
        loss_history.append(mse_loss(y_true, model(x, t, theta)))
    return theta, loss_history

def traditional_gradient_descent(x, t, y_true, theta, alpha, epochs):
    loss_history = []
    for epoch in range(epochs):
        gradients = compute_gradients(x, t, y_true, theta)[:2]
        theta = theta - alpha * gradients
        loss_history.append(mse_loss(y_true, model(x, t, theta)))
    return theta, loss_history

def create_temporal_features(data, lag_days=5):
    data["Day_Sequence"] = np.arange(len(data))
    for i in range(1, lag_days + 1):
        data[f"Open_Lag_{i}"] = data["Open"].shift(i)
    data = data.dropna().reset_index(drop=True)
    return data

# Load the dataset
data = pd.read_csv('/mnt/data/BTC-USD.csv')

# Data preprocessing and feature engineering
train_size = int(0.7 * len(data))
test_size = int(0.2 * len(data))
train_data = data[:train_size]
test_data = data[train_size:train_size + test_size]
predict_data = data[train_size + test_size:]

# Temporal features
train_data = create_temporal_features(train_data)
test_data = create_temporal_features(test_data)
predict_data = create_temporal_features(predict_data)

# Training Gradient Boosting with Temporal Features
features = ["Open", "Day_Sequence"] + [f"Open_Lag_{i}" for i in range(1, 6)]
X_train_gb = train_data[features].values
y_train_gb = train_data["Close"].values
X_test_gb = test_data[features].values
y_test_gb = test_data["Close"].values

gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_gb, y_train_gb)
y_pred_gb = gb_model.predict(X_test_gb)
mse_gb = mean_squared_error(y_test_gb, y_pred_gb)

# Training Gradient Boosting without Temporal Features
X_train_gb_no_temporal = train_data[["Open"]].values
X_test_gb_no_temporal = test_data[["Open"]].values
gb_model_no_temporal = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model_no_temporal.fit(X_train_gb_no_temporal, y_train_gb)
y_pred_gb_no_temporal = gb_model_no_temporal.predict(X_test_gb_no_temporal)
mse_gb_no_temporal = mean_squared_error(y_test_gb, y_pred_gb_no_temporal)

print(f"Gradient Boosting with Temporal Features MSE: {mse_gb}")
print(f"Gradient Boosting without Temporal Features MSE: {mse_gb_no_temporal}")
