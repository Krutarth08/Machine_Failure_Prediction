import requests, json

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json=json.dumps({"Temperature":"68", "Humidity":"78"}))

print(r)