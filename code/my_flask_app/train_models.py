import pandas as pd
from prediction_model import train_and_save_model
from data_preparation import prepare_data

# Chemin du fichier CSV
csv_path = "../../data/magnificent_seven.csv"

# Charger les données
data = prepare_data(csv_path)


# Entraîner et sauvegarder un modèle pour chaque ticker
for ticker, ticker_data in data.items():
    save_path = f"../../models/{ticker}_close_price_model.pkl"
    mse = train_and_save_model(ticker_data, save_path)
    print(f"Modèle pour {ticker} sauvegardé avec MSE : {mse:.4f}")


    
