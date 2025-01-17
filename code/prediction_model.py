import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib  # Pour sauvegarder et charger le modèle

def train_and_save_model(data, save_path):
    """
    Entraîne un modèle de régression pour prédire les prix de clôture en fonction des prix de clôture passés et de l'année.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les colonnes 'date' et 'close'.
        save_path (str): Chemin pour sauvegarder le modèle.
    
    Returns:
        float: Erreur moyenne quadratique (MSE) du modèle.
    """
    # Vérifier les colonnes nécessaires
    if 'close' not in data.columns or 'date' not in data.columns:
        raise KeyError("Les colonnes 'close' et 'date' doivent exister dans les données.")
    
    # Convertir la colonne 'date' en datetime
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    
    # Supprimer les lignes où les dates sont invalides ou manquantes
    data = data.dropna(subset=['date', 'close'])
    
    # Créer des variables pour les prix de clôture précédents (par exemple, le prix de clôture d'hier)
    data['prev_close'] = data['close'].shift(1)
    
    # Ajouter une colonne 'year' pour l'année
    data['year'] = data['date'].dt.year
    
    # Supprimer la première ligne, qui n'a pas de valeur 'prev_close'
    data = data.dropna(subset=['prev_close'])
    
    # Définir les variables indépendantes et dépendantes
    X = data[['prev_close', 'year']]  # Utilisation du prix de clôture précédent et de l'année comme prédicteurs
    y = data['close']  # Utilisation du prix de clôture comme variable à prédire
    
    # Entraîner le modèle
    model = LinearRegression()
    model.fit(X, y)
    
    # Sauvegarder le modèle
    joblib.dump(model, save_path)
    
    # Calculer l'erreur moyenne quadratique (MSE)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    
    return mse


def load_model(load_path):
    """
    Charge un modèle sauvegardé.
    
    Args:
        load_path (str): Chemin du modèle sauvegardé.
    
    Returns:
        model: Modèle chargé.
    """
    return joblib.load(load_path)

def predict_closing_prices(model, future_data):
    """
    Prédit les prix de clôture pour les dates futures.
    
    Args:
        model: Modèle de régression entraîné.
        future_data (pd.DataFrame): DataFrame contenant les dates futures pour la prédiction.
    
    Returns:
        pd.DataFrame: DataFrame avec les prédictions et la plage d'erreur.
    """
    # Extraire l'année des dates futures
    future_data['year'] = future_data['date'].dt.year
    
    # Ajouter la colonne 'prev_close' pour la prédiction
    # Utilisez la dernière valeur connue de 'close' pour le 'prev_close'
    last_close = future_data['close'].iloc[-1] if 'close' in future_data else 0
    future_data['prev_close'] = [last_close] * len(future_data)
    
    # Prédire les prix de clôture en utilisant 'prev_close' et 'year'
    predictions = model.predict(future_data[['prev_close', 'year']])
    
    # Calculer la plage d'erreur (±2%)
    lower_bound = predictions * 0.98
    upper_bound = predictions * 1.02
    
    # Ajouter les prédictions dans le DataFrame
    future_data['predicted_close'] = predictions
    future_data['lower_bound'] = lower_bound
    future_data['upper_bound'] = upper_bound
    
    return future_data

