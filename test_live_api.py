import requests

url = "https://mlops-project-production.up.railway.app/predict"

data = {
    "features": [3, 0, 22, 7.25]
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:", response.text)