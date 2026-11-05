from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

def getModels():
    gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    modelsList = gemini.models.list()
    print("Available Models:")
    for model in modelsList:
        print(f"{model.name}")

getModels()