import os

def auth_api_key(key):
    if key == os.environ.get("API_KEY"):
        return True
    else:
        return False
