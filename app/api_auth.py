import os

def validate_api_key(key):
    return key == os.environ.get("API_KEY")
