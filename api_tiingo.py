import requests
from tokens import jules_token

headers = {
    'Content-Type': 'application/json'
}

# Utilisation d'une f-string pour insérer jules_token dans l'URL
url = f"https://api.tiingo.com/tiingo/corporate-actions/splits?exDate=2023-9-28&token={jules_token}"

requestResponse = requests.get(url, headers=headers)
print(requestResponse.json())
