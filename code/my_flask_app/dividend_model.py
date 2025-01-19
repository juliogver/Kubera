from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import joblib

def train_and_save_dividend_model(data, closing_prices, save_path):
    """
    Entraîne un modèle pour prédire les dividendes futurs en utilisant les dividendes passés et les prix de clôture.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les colonnes 'date' et 'dividends'.
        closing_prices (pd.Series): Série des prix de clôture correspondants.
        save_path (str): Chemin pour sauvegarder le modèle.
    
    Returns:
        float: Erreur moyenne quadratique (MSE) du modèle.
    """
    if 'dividends' not in data.columns or 'date' not in data.columns:
        raise KeyError("Les colonnes 'dividends' et 'date' doivent exister dans les données.")

    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.dropna(subset=['date', 'dividends'])
    
    # Inclure les prix de clôture comme prédicteur
    data['closing_price'] = closing_prices
    data['prev_dividend'] = data['dividends'].shift(1)
    data = data.dropna(subset=['prev_dividend', 'closing_price'])

    X = data[['prev_dividend', 'closing_price']]
    y = data['dividends']

    # Entraîner le modèle
    model = LinearRegression()
    model.fit(X, y)

    # Sauvegarder le modèle
    joblib.dump(model, save_path)

    # Calculer le MSE
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    return mse


def load_dividend_model(load_path):
    return joblib.load(load_path)

def predict_dividends(model, future_data, future_closing_prices):
    """
    Prédit les dividendes futurs en fonction des prix de clôture et des dividendes passés.
    
    Args:
        model: Modèle entraîné.
        future_data (pd.DataFrame): DataFrame contenant les dates futures pour la prédiction.
        future_closing_prices (pd.Series): Série des prix de clôture futurs.
    
    Returns:
        pd.DataFrame: DataFrame avec les prédictions et la plage d'erreur.
    """
    future_data['prev_dividend'] = [future_data['dividends'].iloc[-1] if 'dividends' in future_data else 0] * len(future_data)
    future_data['closing_price'] = future_closing_prices

    predictions = model.predict(future_data[['prev_dividend', 'closing_price']])
    lower_bound = predictions * 0.92  # ±8%
    upper_bound = predictions * 1.08

    future_data['predicted_dividend'] = predictions
    future_data['lower_bound'] = lower_bound
    future_data['upper_bound'] = upper_bound

    return future_data


def generate_future_dividend_data(last_date, end_year):
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=pd.Timestamp(f"{end_year}-12-31"), freq='D')
    future_data = pd.DataFrame({'date': future_dates})
    return future_data
