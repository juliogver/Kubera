import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests

app = Flask(__name__)

# Clé API NewsAPI
NEWS_API_KEY = "57d058c57a1c4ed3aa11d34b0bde1638"

# Tickers des Magnificent Seven
magnificent_seven_tickers = ["AAPL", "MSFT", "META", "AMZN", "GOOGL", "NVDA", "TSLA"]

# Fonction pour récupérer les données de marché via yfinance
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="10y").reset_index()
    df = df.rename(columns={"Date": "Date", "Close": "Close", "Dividends": "Dividends"})
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)  # Convertir en datetime naïf

    # Ajouter un point fictif si les données s’arrêtent avant la date actuelle
    last_date = df['Date'].max()
    if last_date < pd.Timestamp.now() - pd.DateOffset(months=1):
        # Ajouter un point pour janvier 2025 avec le dernier dividende connu
        last_dividend = df['Dividends'].iloc[-1]
        new_point = pd.DataFrame({
            "Date": [last_date + pd.DateOffset(months=1)],
            "Close": [df['Close'].iloc[-1]],  # Dernier prix connu
            "Dividends": [last_dividend]  # Dernier dividende connu
        })
        df = pd.concat([df, new_point], ignore_index=True)

    return df

# Fonction pour récupérer les actualités via NewsAPI
def fetch_news(ticker):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "apiKey": NEWS_API_KEY,
        "language": "fr",
        "sortBy": "publishedAt",
        "pageSize": 5
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        return articles
    else:
        print(f"Erreur lors de la récupération des actualités : {response.status_code}")
        return []

@app.route("/", methods=["GET", "POST"])
def home():
    selected_ticker = "AAPL"  # Par défaut
    news_articles = []
    
    # Valeur par défaut pour le montant investi (ex. 10 000 €)
    invested_amount = 10000.0

    if request.method == "POST":
        selected_ticker = request.form.get("ticker")
        # Récupérer le montant investi depuis le formulaire, s'il est fourni
        invested_amount_str = request.form.get("invested_amount", "10000")
        try:
            invested_amount = float(invested_amount_str)
        except ValueError:
            invested_amount = 10000.0

    # Charger les données de stock pour le ticker sélectionné
    df = get_stock_data(selected_ticker)

    # Supprimer les lignes avec des dividendes à 0 ou manquants
    df = df.dropna(subset=["Close", "Dividends"])
    df = df[df["Dividends"] > 0]

    if df.empty:
        return render_template("index.html", 
                               tickers=magnificent_seven_tickers, 
                               selected_ticker=selected_ticker, 
                               error_message="Pas de dividendes valides pour ce ticker.")

    # Préparer les données pour la régression linéaire sur les dividendes
    base_date = df['Date'].min()
    date_numeric = (df['Date'] - base_date).dt.days.values.reshape(-1, 1)

    # Régression linéaire sur les dividendes
    model_dividends = LinearRegression()
    model_dividends.fit(date_numeric, df['Dividends'])

    # Prédiction des dividendes futurs sur 36 mois (3 ans)
    future_dates = pd.date_range(start=df['Date'].max() + pd.DateOffset(days=1), periods=36, freq="M")
    future_dates = future_dates.tz_localize(None)  # Convertir les futures dates en datetime naïf

    future_date_numeric = (future_dates - base_date).days.values.reshape(-1, 1)
    predicted_dividends = model_dividends.predict(future_date_numeric)

    # Corriger les prédictions négatives si elles existent
    predicted_dividends = [max(0, val) for val in predicted_dividends]

    # Plage d’erreur pour les dividendes (±8 %)
    lower_bound_dividends = [val * 0.92 for val in predicted_dividends]
    upper_bound_dividends = [val * 1.08 for val in predicted_dividends]

    # --- Graphique des Dividendes ---
    fig_dividends = make_subplots()

    # Tracé des dividendes historiques
    fig_dividends.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Dividends"],
        mode='lines+markers',
        name="Dividendes Historiques",
        line=dict(color='green')
    ))

    # Tracé des prédictions des dividendes
    fig_dividends.add_trace(go.Scatter(
        x=future_dates,
        y=predicted_dividends,
        mode='lines',
        name="Dividendes Prévus",
        line=dict(color='blue')
    ))

    # Plage d'erreur des prédictions
    fig_dividends.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(upper_bound_dividends) + list(lower_bound_dividends[::-1]),
        fill='toself',
        fillcolor='rgba(173,216,230,0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="Plage d'erreur ±8%"
    ))

    fig_dividends.update_layout(
        title=f"Prédiction des Dividendes - {selected_ticker}",
        xaxis_title="Date",
        yaxis_title="Dividendes",
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all")
                ]
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    # --- Graphique des Prix Historiques ---
    fig_prices = go.Figure()
    fig_prices.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode='lines',
        name="Prix de Clôture Historiques",
        line=dict(color='orange')
    ))

    fig_prices.update_layout(
        title=f"Prix de Clôture Historiques - {selected_ticker}",
        xaxis_title="Date",
        yaxis_title="Prix de Clôture",
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all")
                ]
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    # --- Calcul du rendement en dividende ---
    # Utiliser le dernier prix connu pour déterminer le nombre d'actions achetées
    current_price = df['Close'].iloc[-1]
    number_of_shares = invested_amount / current_price

    # Calcul total des dividendes par action sur 3 ans (somme des 36 prédictions)
    total_dividend_per_share = sum(predicted_dividends)

    # Dividende total perçu pour le montant investi
    dividend_received = total_dividend_per_share * number_of_shares

    # Rendement en dividende sur 3 ans (en pourcentage)
    dividend_yield = (dividend_received / invested_amount) * 100

    # Convertir les graphiques en HTML
    predicted_dividend_div = fig_dividends.to_html(full_html=False)
    price_history_div = fig_prices.to_html(full_html=False)

    # Récupérer les actualités
    news_articles = fetch_news(selected_ticker)

    return render_template(
        "index.html",
        predicted_dividend_div=predicted_dividend_div,
        price_history_div=price_history_div,
        selected_ticker=selected_ticker,
        tickers=magnificent_seven_tickers,
        news_articles=news_articles,
        invested_amount=invested_amount,
        number_of_shares=number_of_shares,
        dividend_received=dividend_received,
        dividend_yield=dividend_yield
    )

if __name__ == "__main__":
    app.run(debug=True)
