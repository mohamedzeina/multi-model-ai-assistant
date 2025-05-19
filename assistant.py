from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
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


def talker(message):
    try:
        response = openai.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=message
        )
        audio_stream = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_stream, format="mp3")
        play(audio)
    except Exception as e:
        print(f"[Talker Error] {e}")
        

def transcribe(audio_path):
    """Function to transcribe audio using OpenAI's Whisper API"""
    with open(audio_path, "rb") as audio_file:
        transcription = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcription.text