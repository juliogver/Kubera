from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Charger les données
magnificent_csv_path = "../../data/magnificent_seven.csv"
dividend_data_path = "../../data/magnificent_seven_dividends.csv"
top_5_fr_csv_path = "../../data/top_5_fr.csv"
dividend_top_5fr_path = "../../data/french_top5_dividends.csv"

magnificent_data = pd.read_csv(magnificent_csv_path)
dividend_data = pd.read_csv(dividend_data_path)
top_5_fr_data = pd.read_csv(top_5_fr_csv_path)
dividend_top5 = pd.read_csv(dividend_top_5fr_path)

# Convertir les dates
magnificent_data['date'] = pd.to_datetime(magnificent_data['date'], errors='coerce')
dividend_data['date'] = pd.to_datetime(dividend_data['date'], errors='coerce')
top_5_fr_data['date'] = pd.to_datetime(top_5_fr_data['date'], errors='coerce')
dividend_top5['date'] = pd.to_datetime(dividend_top5['date'], errors='coerce')

# Supprimer les colonnes inutiles
magnificent_data = magnificent_data[['date', 'ticker', 'close']]
top_5_fr_data = top_5_fr_data[['date', 'ticker', 'close']]

@app.route("/", methods=["GET", "POST"])
def home():
    tickers = magnificent_data['ticker'].unique()
    top_5_fr_tickers = top_5_fr_data['ticker'].unique()
    selected_ticker = None
    selected_top5_ticker = None
    timeframe = "max"

    real_price_img_path = None
    predicted_dividend_img_path = None

    real_price_top5_img_path = None
    predicted_dividend_top5_img_path = None

    user_investment = None
    dividend_yield = None
    user_investment_top5fr = None
    dividend_yield_top5fr = None
  
    def get_filtered_data(data, timeframe):
        # Utiliser la dernière date disponible dans les données comme référence
        if data['date'].max() is pd.NaT:
            return data  # Si les données sont vides ou incorrectes, renvoyer directement

        last_date = data['date'].max()

        if timeframe == "1D":
            start_date = last_date - pd.Timedelta(days=1)
        elif timeframe == "1W":
            start_date = last_date - pd.Timedelta(weeks=1)
        elif timeframe == "1M":
            start_date = last_date - pd.DateOffset(months=1)
        elif timeframe == "1Y":
            start_date = last_date - pd.DateOffset(years=1)
        else:  # "max"
            return data

        # Filtrer les données
        filtered_data = data[data['date'] >= start_date]
        print(f"Nombre de points après filtrage ({timeframe}): {len(filtered_data)}")  # Debug
        return filtered_data


    # Traitement pour le Magnificent Seven
    if request.method == "POST" and request.form.get("ticker"):
        selected_ticker = request.form.get("ticker")
        timeframe = request.form.get("timeframe")
        user_investment = request.form.get("investment")

        

        try:
            df = magnificent_data[magnificent_data['ticker'] == selected_ticker].copy()
            df_filtered = get_filtered_data(df, timeframe)
            dividends = dividend_data[dividend_data['ticker'] == selected_ticker].copy()

            if not dividends.empty and not df.empty:
                    df = df.dropna(subset=['close'])
                    dividends = dividends.dropna(subset=['dividends'])
                    merged_data = pd.merge(dividends, df, on=['date', 'ticker'], how='inner')

                    if not merged_data.empty:
                        merged_data['dividend_to_price_ratio'] = merged_data['dividends'] / merged_data['close']
                        avg_ratio = merged_data['dividend_to_price_ratio'].mean()

                        if user_investment and avg_ratio:
                            user_investment = float(user_investment)
                            dividend_yield = user_investment * avg_ratio

            if dividends.empty:
                return render_template(
                    "index.html",
                    tickers=tickers,
                    # top_5_fr_tickers=top_5_fr_tickers,
                    # top_5_closing_img_paths=top_5_closing_img_paths,
                    error_message=f"Le ticker {selected_ticker} ne verse pas de dividendes.",
                    selected_ticker=selected_ticker,
                    timeframe=timeframe,
                )

            dividends = dividends.dropna(subset=['dividends'])
            df = df.dropna(subset=['close'])
            merged_data = pd.merge(dividends, df, on=['date', 'ticker'], how='inner')

            if merged_data.empty:
                return render_template(
                    "index.html",
                    tickers=tickers,
                    #top_5_fr_tickers=top_5_fr_tickers,
                    #top_5_closing_img_paths=top_5_closing_img_paths,
                    error_message=f"Aucune donnée commune disponible pour {selected_ticker}.",
                    selected_ticker=selected_ticker,
                    timeframe=timeframe,
                )

            merged_data['dividend_to_price_ratio'] = merged_data['dividends'] / merged_data['close']
            avg_ratio = merged_data['dividend_to_price_ratio'].mean()

            fig_closing, ax_closing = plt.subplots(figsize=(10, 6))
            ax_closing.plot(df_filtered['date'], df_filtered['close'], label="Prix de Clôture Historiques", color="green")
            ax_closing.set_title(f"Prix de Clôture Historiques - {selected_ticker}")
            ax_closing.set_xlabel("Date")
            ax_closing.set_ylabel("Prix de Clôture")
            ax_closing.legend()
            ax_closing.grid(True)
            real_price_img_path = "static/real_price_plot.png"
            fig_closing.savefig(real_price_img_path)

            last_date = df['date'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=pd.Timestamp("2026-12-31"), freq='D')

            base_date = df['date'].min()
            date_numeric = (df['date'] - base_date).dt.days
            poly = PolynomialFeatures(degree=3)
            date_numeric_poly = poly.fit_transform(date_numeric.values.reshape(-1, 1))

            model_prices = LinearRegression()
            model_prices.fit(date_numeric_poly, df['close'])

            future_date_numeric = (future_dates - base_date).days
            future_date_numeric_poly = poly.transform(np.array(future_date_numeric).reshape(-1, 1))
            predicted_prices = model_prices.predict(future_date_numeric_poly)
            predicted_dividends = predicted_prices * avg_ratio

            lower_bound_dividends = predicted_dividends * 0.92
            upper_bound_dividends = predicted_dividends * 1.08

            all_dates = pd.concat([merged_data['date'], pd.Series(future_dates)], ignore_index=True)
            all_dividends = pd.concat([merged_data['dividends'], pd.Series(predicted_dividends)], ignore_index=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(all_dates, all_dividends, label="Dividendes Historiques et Prédits", color="blue")
            ax.fill_between(future_dates, lower_bound_dividends, upper_bound_dividends, color='orange', alpha=0.3, label="Plage d'erreur ±8%")
            ax.set_title(f"Prédiction des Dividendes - {selected_ticker}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Dividendes")
            ax.legend()
            ax.grid(True)
            predicted_dividend_img_path = "static/predicted_dividend_plot.png"
            fig.savefig(predicted_dividend_img_path)

            


        except Exception as e:
            return render_template(
                "index.html",
                tickers=tickers,
                #top_5_fr_tickers=top_5_fr_tickers,
                #top_5_closing_img_paths=top_5_closing_img_paths,
                error_message=f"Une erreur est survenue : {str(e)}",
                selected_ticker=selected_ticker,
                timeframe=timeframe,
            )

     # Gestion de la sélection des tickers pour le Top 5 Français
    if request.method == "POST" and "ticker_top5" in request.form:
            
        selected_top5_ticker = request.form.get("ticker_top5")
        timeframe = request.form.get("timeframe")
        user_investment_top5fr = request.form.get("investment")

        try:
            df = top_5_fr_data[top_5_fr_data['ticker'] == selected_top5_ticker].copy()
            df_filtered = get_filtered_data(df, timeframe)
            dividends = dividend_top5[dividend_top5['ticker'] == selected_top5_ticker].copy()

            if not dividends.empty and not df.empty:
                    df = df.dropna(subset=['close'])
                    dividends = dividends.dropna(subset=['dividends'])
                    merged_data = pd.merge(dividends, df, on=['date', 'ticker'], how='inner')

                    if not merged_data.empty:
                        merged_data['dividend_to_price_ratio'] = merged_data['dividends'] / merged_data['close']
                        avg_ratio = merged_data['dividend_to_price_ratio'].mean()

                        if user_investment_top5fr and avg_ratio:
                            user_investment_top5fr = float(user_investment_top5fr)
                            dividend_yield_top5fr = user_investment_top5fr * avg_ratio

            if dividends.empty:
                return render_template(
                    "index.html",
                    top_5_fr_tickers=top_5_fr_tickers,
                    #top_5_fr_tickers=top_5_fr_tickers,
                    #top_5_closing_img_paths=top_5_closing_img_paths,
                    error_message_fr=f"Le ticker {selected_top5_ticker} ne verse pas de dividendes.",
                    selected_top5_ticker=selected_top5_ticker,
                    timeframe=timeframe,
                )

            dividends = dividends.dropna(subset=['dividends'])
            df = df.dropna(subset=['close'])
            merged_data = pd.merge(dividends, df, on=['date', 'ticker'], how='inner')

            if merged_data.empty:
                return render_template(
                    "index.html",
                    top_5_fr_tickers=top_5_fr_tickers,
                    #top_5_fr_tickers=top_5_fr_tickers,
                    #top_5_closing_img_paths=top_5_closing_img_paths,
                    error_message_fr=f"Aucune donnée commune disponible pour {selected_top5_ticker}.",
                    selected_top5_ticker=selected_top5_ticker,
                    timeframe=timeframe,
                )

            merged_data['dividend_to_price_ratio'] = merged_data['dividends'] / merged_data['close']
            avg_ratio = merged_data['dividend_to_price_ratio'].mean()

            fig_closing, ax_closing = plt.subplots(figsize=(10, 6))
            ax_closing.plot(df_filtered['date'], df_filtered['close'], label="Prix de Clôture Historiques", color="green")
            ax_closing.set_title(f"Prix de Clôture Historiques - {selected_top5_ticker}")
            ax_closing.set_xlabel("Date")
            ax_closing.set_ylabel("Prix de Clôture")
            ax_closing.legend()
            ax_closing.grid(True)
            real_price_top5_img_path = "static/real_price_top5.png"
            fig_closing.savefig(real_price_top5_img_path)

            last_date = df['date'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=pd.Timestamp("2026-12-31"), freq='D')

            base_date = df['date'].min()
            date_numeric = (df['date'] - base_date).dt.days
            poly = PolynomialFeatures(degree=3)
            date_numeric_poly = poly.fit_transform(date_numeric.values.reshape(-1, 1))

            model_prices = LinearRegression()
            model_prices.fit(date_numeric_poly, df['close'])

            future_date_numeric = (future_dates - base_date).days
            future_date_numeric_poly = poly.transform(np.array(future_date_numeric).reshape(-1, 1))
            predicted_prices = model_prices.predict(future_date_numeric_poly)
            predicted_dividends = predicted_prices * avg_ratio

            lower_bound_dividends = predicted_dividends * 0.92
            upper_bound_dividends = predicted_dividends * 1.08

            all_dates = pd.concat([merged_data['date'], pd.Series(future_dates)], ignore_index=True)
            all_dividends = pd.concat([merged_data['dividends'], pd.Series(predicted_dividends)], ignore_index=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(all_dates, all_dividends, label="Dividendes Historiques et Prédits", color="blue")
            ax.fill_between(future_dates, lower_bound_dividends, upper_bound_dividends, color='orange', alpha=0.3, label="Plage d'erreur ±8%")
            ax.set_title(f"Prédiction des Dividendes - {selected_top5_ticker}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Dividendes")
            ax.legend()
            ax.grid(True)
            predicted_dividend_top5_img_path = "static/predicted_dividend_top5_plot.png"
            fig.savefig(predicted_dividend_top5_img_path)


        except Exception as e:
            return render_template(
                "index.html",
                top_5_fr_tickers=top_5_fr_tickers,
                error_message_fr=f"Une erreur est survenue : {str(e)}",
                selected_top5_ticker=selected_top5_ticker,
                timeframe=timeframe,
            )


    return render_template(
        "index.html",
        tickers=tickers,
        top_5_fr_tickers=top_5_fr_tickers,
        selected_ticker=selected_ticker,
        selected_top5_ticker=selected_top5_ticker,
        timeframe=timeframe,
        real_price_img_path=real_price_img_path,
        predicted_dividend_img_path=predicted_dividend_img_path,
        real_price_top5_img_path=real_price_top5_img_path,
        predicted_dividend_top5_img_path=predicted_dividend_top5_img_path,
        dividend_yield=dividend_yield,
        dividend_yield_top5fr=dividend_yield_top5fr,
        user_investment_top5fr=user_investment_top5fr,
        user_investment=user_investment
    )

if __name__ == "__main__":
    app.run(debug=True)

