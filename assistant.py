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






def chat(history, selected_model):
    """Generate AI responses based on chat history and selected model"""
    if not history:
        return history
        
    # Filter conversation history (excluding system messages)
    chat_history = [msg for msg in history if msg["role"] in ["user", "assistant"]]
    
    if selected_model == "GPT":
        # For GPT, include the system message
        messages = [{"role": "system", "content": sysPrompt}] + chat_history
        stream = openai.chat.completions.create(model=MODEL_GPT, messages=messages, stream=True)
                                               
        response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                response += delta.content
                yield history + [{"role": "assistant", "content": response}]
        talker(response)
                
    elif selected_model == "Ollama":
        # For Ollama, format the system prompt as first message
        messages = [{"role": "system", "content": sysPrompt}] + chat_history
        stream = ollama.chat(model=MODEL_LLAMA, messages=messages, stream=True)
        response = ""
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                response += chunk['message']['content']
                yield history + [{"role": "assistant", "content": response}]
        talker(response)
                    
    elif selected_model == "Gemini":
        # For Gemini, convert to its specific format
        gemini_messages = convert_to_gemini_format(chat_history)
        
        if not gemini_messages:
            return history
            
        try:
            stream = gemini.generate_content(gemini_messages, stream=True)
            
            response = ""
            for chunk in stream:
                if hasattr(chunk, "text") and chunk.text:
                    response += chunk.text
                    yield history + [{"role": "assistant", "content": response}]
        except Exception as e:
            yield history + [{"role": "assistant", "content": f"Error with Gemini model: {str(e)}\n\nPlease try another model or rephrase your question."}]
                
        talker(response)









