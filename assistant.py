from openai import OpenAI
from dotenv import load_dotenv
import os
import google.generativeai as genai


# Constants 
MODEL_GPT = 'gpt-4o-mini'
MODEL_LLAMA = 'llama3.2'
MODEL_GEMINI = 'gemini-1.5-flash'  

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
openai = OpenAI()

google_api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=google_api_key)
gemini = genai.GenerativeModel(MODEL_GEMINI)


sysPrompt = "You are an expert at answering technical questions. "
sysPrompt += "Provide clear, accurate, and well-structured explanations that help the user understand complex concepts easily. "
sysPrompt += "If you do not know something then make it clear to the user that you do not have an answer."
