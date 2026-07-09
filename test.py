import requests

data = {
    "pclass": 3,
    "age": 40,
    "fare": 7,
    "sex": 0,
    "sibsp": 0
}

response = requests.post("http://127.0.0.1:5000/predict", json=data)
print(response.json())