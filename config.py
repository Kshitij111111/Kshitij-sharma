import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
    API_KEY = os.getenv('API_KEY', 'your-api-key-here')
    API_URL = os.getenv('API_URL', 'https://api.provider.com/v1/chat/completions')  # Update with your API URL