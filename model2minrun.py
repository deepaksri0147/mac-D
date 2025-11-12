import requests
import time

URL = "http://ollama-keda.mobiusdtaas.ai/api/tags"

def call_api():
    try:
        response = requests.get(URL)
        print(f"Status Code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")

if __name__ == "__main__":
    while True:
        call_api()
        time.sleep(120)  # Wait for 2 minutes