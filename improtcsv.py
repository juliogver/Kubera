import os
import zipfile
import pandas as pd

# Configuration Kaggle : Assurez-vous que kaggle.json est configuré correctement
os.environ["KAGGLE_CONFIG_DIR"] = r"C:\Users\Enzo\.kaggle"

# Nom du dataset Kaggle
dataset_name = "jakewright/9000-tickers-of-stock-market-data-full-history"

# Répertoires pour les fichiers
download_dir = "./data"
zip_file_name = "9000-tickers-of-stock-market-data-full-history.zip"  # Nom du fichier ZIP
zip_file_path = os.path.join(download_dir, zip_file_name)
extracted_dir = os.path.join(download_dir, "extracted")

# Créer les répertoires si nécessaire
if not os.path.exists(download_dir):
    os.makedirs(download_dir)
if not os.path.exists(extracted_dir):
    os.makedirs(extracted_dir)

# Télécharger le dataset depuis Kaggle
print("Téléchargement du dataset depuis Kaggle...")
download_command = f"kaggle datasets download -d {dataset_name} -p {download_dir} --force"
os.system(download_command)

# Vérifier si le fichier ZIP existe
if os.path.exists(zip_file_path):
    print("Fichier ZIP téléchargé avec succès.")

    # Décompression du fichier ZIP
    print("Décompression du fichier ZIP...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)
    print(f"Fichiers décompressés dans : {extracted_dir}")

    # Charger le fichier CSV
    csv_file_path = os.path.join(extracted_dir, "all_stock_data.csv")
    if os.path.exists(csv_file_path):
        print("Chargement du fichier CSV...")
        df = pd.read_csv(csv_file_path)
        print(df.head())  # Afficher les premières lignes du fichier
    else:
        print("Erreur : Fichier CSV non trouvé après décompression.")
else:
    print("Erreur : Le fichier ZIP n'a pas été téléchargé.")

# Nettoyage (Optionnel) : Supprimez le fichier ZIP si nécessaire
if os.path.exists(zip_file_path):
    os.remove(zip_file_path)
    print("Fichier ZIP supprimé après extraction.")
else:
    print("Aucun fichier ZIP à supprimer.")
