import requests

url = "http://127.0.0.1:5001/predict"

data = {
    "features": [3, 0, 22, 7.25]
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:", response.text)