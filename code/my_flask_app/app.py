from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from prediction_model import load_model, predict_closing_prices
from data_preparation import prepare_data

app = Flask(__name__)

# Charger les données
csv_path = "../../data/magnificent_seven.csv"
data = prepare_data(csv_path)

# Filtrer les lignes vides dans chaque DataFrame pour chaque ticker
for ticker in data.keys():
    data[ticker] = data[ticker].dropna(how="all")  # Supprimer les lignes vides

@app.route("/", methods=["GET", "POST"])
def home():
    tickers = list(data.keys())
    selected_ticker = None
    timeframe = "max"  # Valeur par défaut

    # Si l'utilisateur a soumis le formulaire
    if request.method == "POST":
        selected_ticker = request.form.get("ticker")
        timeframe = request.form.get("timeframe")  # Obtenir la temporalité choisie
        model_path = f"../../models/{selected_ticker}_close_price_model.pkl"

        try:
            # Charger le modèle
            model = load_model(model_path)

            # Préparer les données pour le ticker sélectionné
            df = data[selected_ticker].copy()
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])

            # Affichage des dates disponibles dans le DataFrame pour diagnostic
            print(f"Dates disponibles dans le dataset pour {selected_ticker}:")
            print(df['date'].min(), df['date'].max())  # Affiche la première et dernière date du DataFrame

            # Ajuster les données selon la temporalité choisie
            end_date = pd.Timestamp.now()  # Date actuelle
            if timeframe == "1D":
                start_date = end_date - pd.Timedelta(days=1)
            elif timeframe == "1W":
                start_date = end_date - pd.Timedelta(days=7)
            elif timeframe == "1M":
                start_date = end_date - pd.Timedelta(days=30)  # Approximativement un mois
            elif timeframe == "1Y":
                start_date = end_date - pd.Timedelta(days=365)  # Approximativement un an
            elif timeframe == "max":
                start_date = df["date"].min()

            # Affichage des dates de début et de fin pour vérifier le filtrage
            print(f"start_date: {start_date}, end_date: {end_date}")

            # Filtrer les données pour la période choisie
            df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

            '''# Vérification : Si aucune donnée n'est disponible, avertir l'utilisateur
            if df_filtered.empty:
                error_message = f"Aucune donnée disponible pour {selected_ticker} sur la période sélectionnée."
                print(f"Pas de données pour la période demandée: {error_message}")
                return render_template("index.html", tickers=tickers, error_message=error_message, selected_ticker=selected_ticker)

            # Sauvegarder les données filtrées dans un CSV pour diagnostic
            output_csv_path = f"static/{selected_ticker}_filtered_data_{timeframe}.csv"
            df_filtered.to_csv(output_csv_path, index=False)
            print(f"Données filtrées sauvegardées dans : {output_csv_path}")'''

            # Vous pouvez ensuite visualiser ce fichier CSV dans un tableur ou l'ouvrir pour vérifier les données

            # Vérification : Si aucune donnée n'est disponible, avertir l'utilisateur
            if df_filtered.empty:
                error_message = f"Aucune donnée disponible pour {selected_ticker} sur la période sélectionnée."
                print(f"Pas de données pour la période demandée: {error_message}")
                return render_template("index.html", tickers=tickers, error_message=error_message, selected_ticker=selected_ticker)

            # Définir le format de la date pour l'axe des X
            if timeframe == "1D" or timeframe == "1W":
                date_format = "%Y-%m-%d"  # Format jour-mois-année
            elif timeframe == "1M":
                date_format = "%Y-%m"  # Format mois-année
            elif timeframe == "1Y":
                date_format = "%Y-%m"  # Format année uniquement
            else:  # Pour "max" ou autre
                date_format = "%Y"

            # Créer le graphique des prix réels
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(df_filtered['date'], df_filtered['close'], label="Prix de Clôture Réels", color="green")
            ax1.set_title(f"Prix de Clôture Réels ({timeframe}) - {selected_ticker}")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Prix de Clôture")
            ax1.legend()
            ax1.grid(True)
            ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(date_format))
            fig1.autofmt_xdate()

            real_price_img_path = "static/real_price_plot.png"
            fig1.savefig(real_price_img_path)

            # Prédire les prix futurs
            last_date = df_filtered['date'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=pd.to_datetime('2026-12-31'), freq='D')
            future_data = pd.DataFrame({'date': future_dates})
            predictions = predict_closing_prices(model, future_data)

            # Créer le graphique des prédictions
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(predictions['date'], predictions['predicted_close'], label="Prix de Clôture Prévus", color="blue")
            ax2.fill_between(predictions['date'], predictions['lower_bound'], predictions['upper_bound'], 
                            color='blue', alpha=0.2, label='Plage d\'erreur (±2%)')
            ax2.set_title(f"Prédiction des Prix de Clôture - {selected_ticker}")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Prix de Clôture")
            ax2.legend()
            ax2.grid(True)

            predicted_price_img_path = "static/predicted_price_plot.png"
            fig2.savefig(predicted_price_img_path)

            return render_template(
                "index.html", 
                tickers=tickers, 
                selected_ticker=selected_ticker, 
                timeframe=timeframe, 
                real_price_img_path=real_price_img_path, 
                predicted_price_img_path=predicted_price_img_path
            )

        except FileNotFoundError:
            error_message = f"Modèle pour {selected_ticker} non trouvé. Entraînez-le d'abord."
            return render_template("index.html", tickers=tickers, error_message=error_message, selected_ticker=selected_ticker, timeframe=timeframe)
        except Exception as e:
            return render_template("index.html", tickers=tickers, error_message=str(e), selected_ticker=selected_ticker, timeframe=timeframe)

    return render_template("index.html", tickers=tickers, selected_ticker=selected_ticker, timeframe=timeframe)


if __name__ == "__main__":
    app.run(debug=True)
