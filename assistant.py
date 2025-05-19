from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
import os
import gradio as gr
import ollama
import google.generativeai as genai
import subprocess


# First verify ffmpeg/ffprobe is installed and get exact paths
try:
    ffmpeg_path = subprocess.check_output(['which', 'ffmpeg']).decode('utf-8').strip()
    ffprobe_path = subprocess.check_output(['which', 'ffprobe']).decode('utf-8').strip()
    print(f"Found ffmpeg at: {ffmpeg_path}")
    print(f"Found ffprobe at: {ffprobe_path}")
except Exception as e:
    print(f"Error finding ffmpeg/ffprobe: {e}")
    print("Falling back to default paths")
    ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
    ffprobe_path = "/opt/homebrew/bin/ffprobe"

# Configure paths before importing pydub
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
os.environ["FFMPEG_BINARY"] = ffmpeg_path
os.environ["FFPROBE_BINARY"] = ffprobe_path

# Configure pydub directly
AudioSegment.converter = ffmpeg_path
AudioSegment.ffmpeg = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path


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


def convert_to_gemini_format(history):
    """Convert chat history to Gemini's expected format"""
    gemini_messages = []
    for message in history:
        if message["role"] == "user":
            gemini_messages.append({"role": "user", "parts": [message["content"]]})
        elif message["role"] == "assistant":
            gemini_messages.append({"role": "model", "parts": [message["content"]]})
    
    # Add system prompt to first user message if available
    if gemini_messages and gemini_messages[0]["role"] == "user":
        gemini_messages[0]["parts"][0] = f"Please follow these instructions: {sysPrompt}\n\nUser's question: {gemini_messages[0]['parts'][0]}"
    
    return gemini_messages



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
        
# Gradio UI
with gr.Blocks(title="AI Assistant Hub") as ui:
    gr.Markdown("# AI Assistant Hub\nChat with multiple AI models using text or voice")
    
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages", elem_id="chatbox")
        
    with gr.Row():
        with gr.Column(scale=6):
            entry = gr.Textbox(
            label=None,
            placeholder="Type your message here..."
            )

        with gr.Column(scale=1):
            submit_btn = gr.Button("Send", variant="primary")
            
    with gr.Row():
        audio_input = gr.Microphone(
        label="Speak to the assistant", 
        type="filepath"
        )
        model_selector = gr.Dropdown(
            ["GPT", "Ollama", "Gemini"], 
            label="Select model", 
            value="GPT",
            interactive=True
        )
        clear = gr.Button("Clear Chat", variant="secondary")
    
    # Define handler functions within the Blocks context
    def handle_text_input(message, history):
        """Handle text input from the user"""
        history = history or []
        history.append({"role": "user", "content": message})
        return "", history
            
    def handle_audio_input(audio_path, history):
        """Handle audio input from the user"""
        if not audio_path or not isinstance(audio_path, str) or not os.path.exists(audio_path):
            return gr.update(), history  # Prevent triggering downstream if cleared
    
        history = history or []
        text = transcribe(audio_path)
        history.append({"role": "user", "content": text})
        return gr.update(value=None), history  # Clear audio after handling

    
    # Event handlers
    entry.submit(
        handle_text_input, 
        inputs=[entry, chatbot], 
        outputs=[entry, chatbot]
    ).then(
        chat, 
        inputs=[chatbot, model_selector], 
        outputs=chatbot
    )
    
    submit_btn.click(
        handle_text_input, 
        inputs=[entry, chatbot], 
        outputs=[entry, chatbot]
    ).then(
        chat, 
        inputs=[chatbot, model_selector], 
        outputs=chatbot
    )
    
    audio_input.change(
        handle_audio_input, 
        inputs=[audio_input, chatbot], 
        outputs=[audio_input, chatbot]
    ).then(
        chat, 
        inputs=[chatbot, model_selector], 
        outputs=chatbot
)
    
    # Clear button handler
    clear.click(lambda: [], outputs=chatbot, queue=False)
    
    # Reset chatbot history when the model is changed
    model_selector.change(lambda: ([], None, None), outputs=[chatbot, audio_input, entry], queue=False)

# Launch the Gradio UI
ui.launch(inbrowser=True)









