import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Charger les données de dividendes depuis le CSV
dividend_csv_path = "../../data/magnificent_seven_dividends.csv"
data_dividends = pd.read_csv(dividend_csv_path)

# Charger les données de prix de clôture depuis un autre CSV
price_csv_path = "../../data/magnificent_seven.csv"
data_prices = pd.read_csv(price_csv_path)

# Filtrer les données pour un ticker spécifique (par exemple AAPL)
ticker = 'AAPL'
data_ticker_dividends = data_dividends[data_dividends['ticker'] == ticker]
data_ticker_prices = data_prices[data_prices['ticker'] == ticker]

# Convertir les dates en objets datetime pour les dividendes
dates_dividends = [datetime.strptime(date, "%Y-%m-%d") for date in data_ticker_dividends['date']]
values_dividends = data_ticker_dividends['dividends'].values

# Convertir les dates en objets datetime pour les prix de clôture
dates_prices = [datetime.strptime(date, "%Y-%m-%d") for date in data_ticker_prices['date']]
close_prices = data_ticker_prices['close'].values

# Transformation des dates en nombre de jours depuis le premier jour
date_numeric_dividends = [(date - dates_dividends[0]).days for date in dates_dividends]
date_numeric_prices = [(date - dates_prices[0]).days for date in dates_prices]

# Créer un modèle polynomial de degré 3 pour les dividendes
poly = PolynomialFeatures(degree=3)
date_numeric_poly_dividends = poly.fit_transform(np.array(date_numeric_dividends).reshape(-1, 1))

# Modèle de régression linéaire pour les dividendes
model_dividends = LinearRegression()
model_dividends.fit(date_numeric_poly_dividends, values_dividends)

# Créer un modèle polynomial de degré 3 pour les prix de clôture
date_numeric_poly_prices = poly.fit_transform(np.array(date_numeric_prices).reshape(-1, 1))

# Modèle de régression linéaire pour les prix de clôture
model_prices = LinearRegression()
model_prices.fit(date_numeric_poly_prices, close_prices)

# Fonction pour faire des prédictions sur de nouvelles dates
def predict_future_values_with_error(model, start_date, end_year, poly, base_dates):
    future_dates = []
    future_date_numeric = []
    
    # Générer des dates futures jusqu'à l'année spécifiée (ex: 2030)
    current_year = start_date.year
    while current_year <= end_year:
        future_date = datetime(current_year, start_date.month, start_date.day)
        future_dates.append(future_date.strftime("%Y-%m-%d"))
        future_date_numeric.append((future_date - base_dates[0]).days)
        current_year += 1
    
    # Appliquer la transformation polynomial et prédire
    future_date_numeric_poly = poly.transform(np.array(future_date_numeric).reshape(-1, 1))
    predicted_values = model.predict(future_date_numeric_poly)
    
    # Ajouter la marge d'erreur de ±8%
    error_margin = 0.08  # 8% error
    lower_bound = predicted_values * (1 - error_margin)
    upper_bound = predicted_values * (1 + error_margin)
    
    return list(zip(future_dates, predicted_values, lower_bound, upper_bound))

# Prédire les valeurs des dividendes et des prix de clôture jusqu'en 2030
predictions_with_error_dividends = predict_future_values_with_error(model_dividends, dates_dividends[-1], 2030, poly, dates_dividends)
predictions_with_error_prices = predict_future_values_with_error(model_prices, dates_prices[-1], 2030, poly, dates_prices)

# Filtrer les dates et les valeurs historiques à partir de 1980
historical_dates_dividends = [date for date in dates_dividends if date.year >= 1980]
historical_values_dividends = values_dividends[-len(historical_dates_dividends):]
historical_dates_prices = [date for date in dates_prices if date.year >= 1980]
historical_values_prices = close_prices[-len(historical_dates_prices):]

# Fusionner les dates historiques et futures pour les dividendes
all_dates_dividends = historical_dates_dividends + [datetime.strptime(date, "%Y-%m-%d") for date, _, _, _ in predictions_with_error_dividends]
all_values_dividends = list(historical_values_dividends) + [value for _, value, _, _ in predictions_with_error_dividends]
lower_bound_values_dividends = [value for _, _, value, _ in predictions_with_error_dividends]
upper_bound_values_dividends = [value for _, _, _, value in predictions_with_error_dividends]

# Fusionner les dates historiques et futures pour les prix de clôture
all_dates_prices = historical_dates_prices + [datetime.strptime(date, "%Y-%m-%d") for date, _, _, _ in predictions_with_error_prices]
all_values_prices = list(historical_values_prices) + [value for _, value, _, _ in predictions_with_error_prices]
lower_bound_values_prices = [value for _, _, value, _ in predictions_with_error_prices]
upper_bound_values_prices = [value for _, _, _, value in predictions_with_error_prices]

# Graphique des données existantes et des prédictions avec la plage d'erreur pour les dividendes
plt.figure(figsize=(10,6))
plt.plot(all_dates_dividends, all_values_dividends, label='Données Historiques et Prédictions Dividendes', color='blue')
plt.fill_between(all_dates_dividends[len(historical_dates_dividends):], lower_bound_values_dividends, upper_bound_values_dividends, color='red', alpha=0.3, label='Plage d\'erreur ±8%')
plt.xlabel('Date')
plt.ylabel('Valeur des Dividendes')
plt.title('Données Historiques et Prédictions des Dividendes avec Plage d\'Erreur ±8%')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Graphique des données existantes et des prédictions avec la plage d'erreur pour les prix de clôture
plt.figure(figsize=(10,6))
plt.plot(all_dates_prices, all_values_prices, label='Données Historiques et Prédictions des Prix de Clôture', color='green')
plt.fill_between(all_dates_prices[len(historical_dates_prices):], lower_bound_values_prices, upper_bound_values_prices, color='orange', alpha=0.3, label='Plage d\'erreur ±8%')
plt.xlabel('Date')
plt.ylabel('Prix de Clôture')
plt.title('Données Historiques et Prédictions des Prix de Clôture avec Plage d\'Erreur ±8%')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Affichage des graphiques
plt.show()
