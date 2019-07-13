import requests

url = 'http://127.0.0.1:5000/'
params ={'query': [4.6,  3.2,  1.4,  0.2] }
response = requests.get(url, params)
response.json()
