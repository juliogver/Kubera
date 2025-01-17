import pandas as pd

# Charger le fichier CSV
# csv_path = "./data/extracted/all_stock_data.csv"  # Chemin du fichier CSV
csv_path = "../data/all_stock_data.csv"  # Chemin du fichier CSV
df = pd.read_csv(csv_path)

# Normaliser les noms de colonnes
df.columns = [col.strip().lower() for col in df.columns]

# Vérification des premières lignes
print(df.head())

# --- 1. Les "Magnificent Seven" ---
magnificent_seven = ['TSLA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA']
df_magnificent_seven = df[df['ticker'].isin(magnificent_seven)]
print("\nDataFrame des Magnificent Seven créé.")

# --- 2. Les 5 plus grosses entreprises françaises ---
# Liste des tickers des 5 plus grosses entreprises françaises
top_france_tickers = ['LVMH', 'SU', 'TTE', 'SAN', 'AI']  # Tickers : LVMH, Schneider Electric, TotalEnergies, Sanofi, Air Liquide
df_top_5_france = df[df['ticker'].isin(top_france_tickers)]
print("\nDataFrame des 5 plus grosses entreprises françaises créé.")

# --- 4. Entreprises entrées en bourse en 2024 ---
# Identifier les tickers dont la première date d'enregistrement est en 2024
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convertir en datetime
first_dates = df.groupby('ticker')['date'].min().reset_index()  # Obtenir la première date par ticker
first_dates_2024 = first_dates[first_dates['date'].dt.year == 2024]  # Filtrer pour l'année 2024

# Joindre pour obtenir les informations complètes des tickers concernés
df_ipo_2024 = df.merge(first_dates_2024, on='ticker', how='inner', suffixes=('', '_first'))

# Afficher les résultats
if not df_ipo_2024.empty:
    print("\nDataFrame des entreprises avec premier ticker en 2024 créé.")
    print(df_ipo_2024.head())
else:
    print("\nAucun ticker avec première date en 2024 trouvé.")

# --- Exporter ou traiter les DataFrames ---
# Optionnel : Sauvegarder chaque DataFrame en CSV
df_magnificent_seven.to_csv("../data/magnificent_seven.csv", index=False)
df_top_5_france.to_csv("../data/top_5_france.csv", index=False)

df_ipo_2024.to_csv("../data/ipo_2024.csv", index=False)

print("\nTous les DataFrames ont été créés et exportés.")
