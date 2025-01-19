import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def train_and_save_model(data, save_path):
    if 'close' not in data.columns or 'date' not in data.columns:
        raise KeyError("Les colonnes 'close' et 'date' doivent exister dans les données.")
    
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.dropna(subset=['date', 'close'])
    data['prev_close'] = data['close'].shift(1)
    data['year'] = data['date'].dt.year
    data = data.dropna(subset=['prev_close'])

    X = data[['prev_close', 'year']]
    y = data['close']

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, save_path)

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    return mse

def load_model(load_path):
    return joblib.load(load_path)

def predict_closing_prices(model, future_data):
    future_data['year'] = future_data['date'].dt.year
    last_close = future_data['close'].iloc[-1] if 'close' in future_data else 0
    future_data['prev_close'] = [last_close] * len(future_data)

    predictions = model.predict(future_data[['prev_close', 'year']])
    lower_bound = predictions * 0.98  # ±8%
    upper_bound = predictions * 1.02

    future_data['predicted_close'] = predictions
    future_data['lower_bound'] = lower_bound
    future_data['upper_bound'] = upper_bound

    return future_data

def generate_future_data(last_date, end_year):
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=pd.Timestamp(f"{end_year}-12-31"), freq='D')
    future_data = pd.DataFrame({'date': future_dates})
    return future_data
