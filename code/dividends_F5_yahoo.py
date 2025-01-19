import yfinance as yf
import pandas as pd

# Liste des tickers des plus grandes entreprises françaises
french_tickers = ['MC.PA', 'OR.PA', 'TTE.PA', 'SAN.PA', 'AIR.PA']

# Dictionnaire pour stocker les dividendes par ticker
dividends_data = {}

# Récupérer les dividendes pour chaque ticker
for ticker in french_tickers:
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

# Organiser les données des dividendes
all_dividends = []

for ticker, dividends in dividends_data.items():
    dividends_df = dividends.reset_index()
    dividends_df['Ticker'] = ticker
    dividends_df['Date'] = pd.to_datetime(dividends_df['Date']).dt.date
    all_dividends.append(dividends_df)

# Concaténer toutes les données dans un seul DataFrame
final_dividends_df = pd.concat(all_dividends, ignore_index=True)

# Mettre les noms des colonnes en minuscules
final_dividends_df.columns = [col.lower() for col in final_dividends_df.columns]

# Enregistrer dans un fichier CSV
output_csv_path = "../data/french_top5_dividends.csv"
final_dividends_df.to_csv(output_csv_path, index=False)

print(f"\nLes dividendes ont été récupérés et sauvegardés dans '{output_csv_path}'.")
