import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score

def main():
    # Chemin vers le CSV contenant la colonne 'dividends' désirée
    dividends_csv_path = "../data/magnificent_seven_dividends.csv"
    # Chemin vers le CSV contenant les autres informations (date, ticker, close, etc.)
    market_csv_path = "../data/magnificent_seven.csv"
    
    # Charger le CSV des dividendes
    dividends_df = pd.read_csv(dividends_csv_path)
    # Vérifier que ce CSV contient bien 'date' et 'dividends'
    if 'date' not in dividends_df.columns or 'dividends' not in dividends_df.columns:
        print("Le fichier magnificent_seven_dividends.csv doit contenir les colonnes 'date' et 'dividends'.")
        return

    # Charger le CSV du marché
    market_df = pd.read_csv(market_csv_path)
    # Vérifier que ce CSV contient au moins 'date' et 'close'
    if 'date' not in market_df.columns or 'close' not in market_df.columns:
        print("Le fichier magnificent_seven.csv doit contenir les colonnes 'date' et 'close'.")
        return

    # Conversion des colonnes 'date' en datetime pour les deux DataFrame
    dividends_df['date'] = pd.to_datetime(dividends_df['date'])
    market_df['date'] = pd.to_datetime(market_df['date'])
    
    # (Optionnel) Si le CSV du marché contient plusieurs tickers, on peut filtrer sur un ticker particulier.
    if 'ticker' in market_df.columns:
        ticker_to_use = "AAPL"  # Par exemple, on choisit AAPL
        market_df = market_df[market_df['ticker'] == ticker_to_use]
    
    # Fusionner les deux DataFrame sur la colonne 'date'
    # Seules les dates communes aux deux fichiers seront conservées (jointure inner)
    merged_df = pd.merge(dividends_df, market_df[['date', 'close']], on='date', how='inner')
    
    # Si la fusion renvoie un DataFrame vide, avertir l'utilisateur
    if merged_df.empty:
        print("La fusion des deux CSV ne retourne aucune donnée commune sur 'date'.")
        return
    
    # Trier par date
    merged_df = merged_df.sort_values(by='date')
    
    # Création d'une variable numérique 'Days' correspondant au nombre de jours écoulés depuis la première date
    base_date = merged_df['date'].min()
    merged_df['Days'] = (merged_df['date'] - base_date).dt.days
    
    # Affichage rapide des premières lignes pour vérification (optionnel)
    print("Aperçu des données fusionnées :")
    print(merged_df.head())
    
    # Préparation des variables pour la régression
    # X : nombre de jours (feature)
    # y : dividendes (target) – on utilise ici la colonne 'dividends' du CSV dédié.
    X = merged_df[['Days']]
    y = merged_df['dividends']
    
    # Entraînement d'un modèle de régression linéaire
    model = LinearRegression()
    model.fit(X, y)
    
    # Prédictions sur l'ensemble des données
    y_pred = model.predict(X)
    
    # Calcul des métriques de performance
    score_r2 = model.score(X, y)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    median_ae = median_absolute_error(y, y_pred)
    if np.all(y != 0):
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
    else:
        mape = None
    explained_variance = explained_variance_score(y, y_pred)
    
    # Affichage des résultats
    print("\nPerformance du modèle sur les données fusionnées :")
    print(f"Score R² : {score_r2:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"Median Absolute Error : {median_ae:.2f}")
    if mape is not None:
        print(f"MAPE : {mape:.2f} %")
    else:
        print("MAPE : Indéfini (les valeurs cibles contiennent zéro)")
    print(f"Explained Variance : {explained_variance:.2f}")
    
    # (Optionnel) Vous pouvez aussi utiliser la colonne 'close' du marché pour d'autres calculs,
    # par exemple, le calcul du rendement en dividende par rapport au prix de clôture.
    # Exemple :
    # merged_df['dividend_yield'] = merged_df['dividends'] / merged_df['close'] * 100
    # print("\nAperçu du rendement en dividende :")
    # print(merged_df[['date', 'dividends', 'close', 'dividend_yield']].head())

if __name__ == "__main__":
    main()
