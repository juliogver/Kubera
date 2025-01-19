import pandas as pd

# Charger le fichier CSV principal contenant toutes les données
csv_path = "../data/all_stock_data.csv"  # Chemin du fichier CSV contenant les données
df = pd.read_csv(csv_path)

# Normaliser les noms des colonnes pour éviter les erreurs de casse
df.columns = [col.strip().lower() for col in df.columns]

# Afficher les premières lignes pour vérification
print(df.head())

# --- Extraire les données pour les 5 plus grosses entreprises françaises ---
# Liste des tickers des 5 plus grosses capitalisations françaises
top_france_tickers = ['MC', 'OR', 'TTE', 'SAN', 'AIR']  # LVMH, L'Oréal, TotalEnergies, Sanofi, Airbus

# Filtrer le DataFrame pour ne conserver que les données des entreprises françaises
df_top_5_france = df[df['ticker'].isin(top_france_tickers)]

# Vérification
print("\nDataFrame des 5 plus grosses entreprises françaises créé.")
print(df_top_5_france.head())

# --- Exporter les données filtrées ---
# Sauvegarder le DataFrame des entreprises françaises dans un fichier CSV
output_csv_path = "../data/top_5_france.csv"
df_top_5_france.to_csv(output_csv_path, index=False)

print(f"\nLes données des 5 plus grandes entreprises françaises ont été sauvegardées dans '{output_csv_path}'.")
