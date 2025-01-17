import pandas as pd

def prepare_data(csv_path):
    """
    Prépare les données pour chaque ticker avec les prix de clôture.
    
    Args:
        csv_path (str): Chemin vers le fichier CSV.
    
    Returns:
        dict: Un dictionnaire avec les tickers comme clés et leurs DataFrames respectifs.
    """
    # Charger le fichier CSV
    df = pd.read_csv(csv_path)
    
    # Obtenir les tickers uniques
    tickers = df['ticker'].unique()
    
    # Organiser les données par ticker
    data = {}
    for ticker in tickers:
        ticker_df = df[df['ticker'] == ticker].copy()
        # Garder 'date' comme une colonne normale et pas comme index
        # ticker_df.set_index('date', inplace=True)
        data[ticker] = ticker_df
    
    return data
