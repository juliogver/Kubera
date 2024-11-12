import requests
headers = {
    'Content-Type': 'application/json'
}
requestResponse = requests.get("https://api.tiingo.com/tiingo/corporate-actions/splits?exDate=2023-9-28&token=276bb16b9cda9fb2da63e8de01d1d46de70364e4", headers=headers)
print(requestResponse.json())