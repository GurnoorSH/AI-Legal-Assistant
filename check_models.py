import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in your .env file.")
else:
    genai.configure(api_key=api_key)
    print("Fetching available models...\n")
    for m in genai.list_models():
      # Check if the model supports the 'generateContent' method
      if 'generateContent' in m.supported_generation_methods:
        print(f"Model Name: {m.name}")