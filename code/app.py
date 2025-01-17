import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from prediction_model import load_model, predict_closing_prices
from data_preparation import prepare_data

# Charger les données
csv_path = "../data/magnificent_seven.csv"
data = prepare_data(csv_path)

# Filtrer les lignes vides dans chaque DataFrame pour chaque ticker
for ticker in data.keys():
    data[ticker] = data[ticker].dropna(how='all')  # Supprimer les lignes vides

# Interface Streamlit
st.title("Prédiction des Prix de Clôture des Magnificent Seven")

# Sélectionner un ticker
tickers = list(data.keys())
selected_ticker = st.selectbox("Sélectionnez un ticker :", tickers)

# Charger le modèle pour le ticker sélectionné
model_path = f"../models/{selected_ticker}_close_price_model.pkl"
try:
    model = load_model(model_path)
    st.success(f"Modèle chargé pour {selected_ticker}")
    
    # Convertir les dates dans le DataFrame en datetime (si ce n'est pas déjà fait)
    data[selected_ticker]['date'] = pd.to_datetime(data[selected_ticker]['date'], errors='coerce')

    # Vérifier et afficher les dates invalides
    invalid_dates = data[selected_ticker][data[selected_ticker]['date'].isna()]
    if not invalid_dates.empty:
        st.warning(f"Certaines dates n'ont pas pu être converties en datetime :\n{invalid_dates}")
    
    # Exclure les lignes où la date est NaT
    data[selected_ticker] = data[selected_ticker].dropna(subset=['date'])

    # Vérifier si la colonne 'date' contient des dates valides
    if pd.api.types.is_datetime64_any_dtype(data[selected_ticker]['date']):
        last_date = data[selected_ticker]['date'].max()
        st.write(f"Dernière date : {last_date}")
    else:
        raise ValueError("La dernière date trouvée n'est pas de type datetime.")
    
    # Calculer la date du jour suivant
    start_date = last_date + pd.Timedelta(days=1)  # Le jour suivant la dernière date
    end_date = pd.to_datetime('2026-12-31')  # Jusqu'à fin 2028
    
    # Générer les dates futures (quotidiennement)
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Préparer les données des dates futures pour les prédictions
    future_data = pd.DataFrame({'date': future_dates})
    
    # Faire les prédictions
    predictions = predict_closing_prices(model, future_data)
    
    # Afficher les prédictions sous forme de tableau
    st.subheader(f"Prédictions des Prix de Clôture pour {selected_ticker} (2024 - 2028)")
    st.write(predictions)
    
    # Visualiser les prédictions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(predictions['date'], predictions['predicted_close'], label='Prix de Clôture Prévus', color='blue')
    ax.fill_between(predictions['date'], predictions['lower_bound'], predictions['upper_bound'], 
                    color='blue', alpha=0.2, label='Plage d\'erreur (±2%)')
    
    ax.set_title(f"Prédiction des Prix de Clôture pour {selected_ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix de Clôture")
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)

except FileNotFoundError:
    st.error(f"Modèle pour {selected_ticker} non trouvé. Entraînez-le d'abord.")
except Exception as e:
    st.error(f"Erreur : {e}")
