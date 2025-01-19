import yfinance as yf
import pandas as pd

# Liste des tickers du Magnificent Seven
magnificent_seven = ['TSLA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA']

# Dictionnaire pour stocker les dividendes par ticker
dividends_data = {}

# Récupérer les dividendes pour chaque ticker
for ticker in magnificent_seven:
    # Télécharger les données financières via Yahoo Finance
    stock = yf.Ticker(ticker)
    
    # Extraire les dividendes
    dividends = stock.dividends
    
    # Si des dividendes existent, les stocker dans le dictionnaire
    if not dividends.empty:
        # Ajouter une colonne pour le ticker et ajouter les données dans le dictionnaire
        dividends_data[ticker] = dividends
    else:
        print(f"Aucun dividende trouvé pour {ticker}.")

# Afficher les dividendes récupérés
for ticker, dividends in dividends_data.items():
    print(f"\nDividendes pour {ticker}:")
    print(dividends)

# Optionnel: Convertir les dividendes en DataFrame pour faciliter l'analyse ou exportation
# Nous allons maintenant organiser les données de manière plus lisible
all_dividends = []

for ticker, dividends in dividends_data.items():
    dividends_df = dividends.reset_index()
    dividends_df['Ticker'] = ticker  # Ajouter le ticker pour chaque ligne
    dividends_df['Date'] = pd.to_datetime(dividends_df['Date']).dt.date  # Extraire seulement la date (année, mois, jour)
    all_dividends.append(dividends_df)

# Concaténer toutes les données dans un seul DataFrame
final_dividends_df = pd.concat(all_dividends, ignore_index=True)

# Mettre les noms des colonnes en minuscules
final_dividends_df.columns = [col.lower() for col in final_dividends_df.columns]

# Enregistrer dans un fichier CSV, avec un chemin modifiable (ici, je mets un nom personnalisé)
output_csv_path = "../../data/magnificent_seven_dividends.csv"  # Nom du fichier CSV de sortie
final_dividends_df.to_csv(output_csv_path, index=False)

print(f"\nLes dividendes ont été récupérés et sauvegardés dans '{output_csv_path}'.")
