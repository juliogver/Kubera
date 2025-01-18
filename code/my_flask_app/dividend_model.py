import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_and_save_dividend_model(data, save_path):
    """
    Entraîne un modèle pour prédire les dividendes futurs.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les colonnes 'date' et 'dividends'.
        save_path (str): Chemin pour sauvegarder le modèle.
    
    Returns:
        float: Erreur moyenne quadratique (MSE) du modèle.
    """
    if 'dividends' not in data.columns or 'date' not in data.columns:
        raise KeyError("Les colonnes 'dividends' et 'date' doivent exister dans les données.")
    
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.dropna(subset=['date', 'dividends'])

    # Créer une variable pour les dividendes passés
    data['prev_dividend'] = data['dividends'].shift(1)
    data['year'] = data['date'].dt.year
    data = data.dropna(subset=['prev_dividend'])

    # Définir X et y
    X = data[['prev_dividend', 'year']]
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
    """
    Charge un modèle sauvegardé.
    
    Args:
        load_path (str): Chemin du modèle sauvegardé.
    
    Returns:
        model: Modèle chargé.
    """
    return joblib.load(load_path)

def predict_dividends(model, future_data):
    """
    Prédit les dividendes futurs.
    
    Args:
        model: Modèle entraîné.
        future_data (pd.DataFrame): DataFrame contenant les dates futures pour la prédiction.
    
    Returns:
        pd.DataFrame: DataFrame avec les prédictions et la plage d'erreur.
    """
    future_data['year'] = future_data['date'].dt.year
    last_dividend = future_data['dividends'].iloc[-1] if 'dividends' in future_data else 0
    future_data['prev_dividend'] = [last_dividend] * len(future_data)
    
    predictions = model.predict(future_data[['prev_dividend', 'year']])
    lower_bound = predictions * 0.98
    upper_bound = predictions * 1.02

    future_data['predicted_dividend'] = predictions
    future_data['lower_bound'] = lower_bound
    future_data['upper_bound'] = upper_bound

    return future_data
