import pandas as pd
from dividend_model import train_and_save_dividend_model

# Chemin des données de dividendes
dividend_csv_path = "../../data/magnificent_seven_dividends.csv"
data = pd.read_csv(dividend_csv_path)

# Entraîner et sauvegarder un modèle pour chaque ticker
tickers = data['ticker'].unique()
for ticker in tickers:
    ticker_data = data[data['ticker'] == ticker].copy()  # Création d'une copie explicite
    save_path = f"../../models/{ticker}_dividend_model.pkl"
    mse = train_and_save_dividend_model(ticker_data, save_path)
    print(f"Modèle pour les dividendes de {ticker} sauvegardé avec MSE : {mse:.4f}")
