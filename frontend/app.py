import sys
import os

# Fix OpenMP conflict permanently
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import threading
import uuid
import shutil
import subprocess
import json
from ollama_speech_analyzer import OllamaSpeechAnalyzer
from tts_synthesizer import synthesize_speech, play_audio
import requests
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa

UPLOAD_FOLDER = 'audio_files'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory chat history (for demo; use DB for production)
chat_histories = {}

# Track analysis status
analysis_status = {}

def get_feedback_history():
    chat_id = session.get('chat_id')
    if not hasattr(app, 'feedback_histories'):
        app.feedback_histories = {}
    if chat_id not in app.feedback_histories:
        app.feedback_histories[chat_id] = []
    return app.feedback_histories[chat_id]

def get_analysis_status():
    chat_id = session.get('chat_id')
    if not chat_id:
        return {'status': 'no_session'}
    return analysis_status.get(chat_id, {'status': 'not_started'})

def clear_output_directories():
    """Clear output directories and files from previous analysis runs"""
    directories_to_clear = ['mfa_output', 'mfa_chunks', 'opensmile_features']
    files_to_remove = [
        'whisper_words.json', 'temporal_phrase_analysis.json', 
        'mispronunciations.json', 'segments_words.csv', 'segments_phones.csv',
        'ollama_analysis.txt'
    ]
    
    # Clear directories
    for directory in directories_to_clear:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Cleared directory: {directory}")
    
    # Remove files
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed file: {file}")

# Helper to get chat history for a session
def get_chat_history():
    chat_id = session.get('chat_id')
    if not chat_id:
        chat_id = str(uuid.uuid4())
        session['chat_id'] = chat_id
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    return chat_histories[chat_id]

def wav2vec2_transcribe(audio_path):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    speech, rate = librosa.load(audio_path, sr=16000, mono=True)
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()

# Fast Ollama: transcribe and respond quickly using Wav2Vec2
def fast_ollama_response(audio_path, chat_history):
    try:
        text = wav2vec2_transcribe(audio_path)
        if not text.strip():
            text = '[No speech detected]'
    except Exception as e:
        text = '[Transcription failed]'
    analyzer = OllamaSpeechAnalyzer()
    conversation = ""
    for msg in chat_history:
        if msg['sender'] == 'User':
            conversation += f"User: {msg['text']}\n"
        elif msg['sender'] == 'Fast Ollama':
            conversation += f"Assistant: {msg['text']}\n"
    conversation += f"User: {text}\n"
    prompt = f"""You are an AI language tutor. Continue the conversation naturally. Keep it around 20 words maximum, but fewer is better to convey the message:\n\n{conversation}Assistant:"""
    ollama_reply = analyzer.query_ollama(prompt)
    return text, ollama_reply

# Slow Ollama: run full analysis in background
def run_slow_ollama(chat_id, audio_path):
    # Set status to running
    analysis_status[chat_id] = {'status': 'running', 'message': 'Analysis in progress...'}
    
    try:
        clear_output_directories()
        dest_path = os.path.join(UPLOAD_FOLDER, 'text_audio.wav')
        shutil.copy(audio_path, dest_path)
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        subprocess.run(['python', 'run_pipeline.py'], check=True, env=env)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        try:
            with open('ollama_analysis.txt', 'r') as f:
                analysis = f.read()
        except FileNotFoundError:
            analysis = "Analysis completed but results file not found."
        # Store in feedback_histories, not chat_histories
        if not hasattr(app, 'feedback_histories'):
            app.feedback_histories = {}
        if chat_id not in app.feedback_histories:
            app.feedback_histories[chat_id] = []
        app.feedback_histories[chat_id].append(analysis)
        # Set status to complete
        analysis_status[chat_id] = {'status': 'complete', 'message': 'Analysis complete!'}
    except Exception as e:
        if not hasattr(app, 'feedback_histories'):
            app.feedback_histories = {}
        if chat_id not in app.feedback_histories:
            app.feedback_histories[chat_id] = []
        app.feedback_histories[chat_id].append(f'Analysis failed: {str(e)}')
        # Set status to failed
        analysis_status[chat_id] = {'status': 'failed', 'message': f'Analysis failed: {str(e)}'}

@app.route('/', methods=['GET'])
def index():
    chat_history = get_chat_history()
    return render_template('index.html', chat_history=chat_history)

@app.route('/send_audio', methods=['POST'])
def send_audio():
    chat_history = get_chat_history()
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    # Fast Ollama response (now conversational)
    user_text, ollama_response = fast_ollama_response(filepath, chat_history)
    chat_history.append({'sender': 'User', 'text': user_text, 'audio_path': filepath})
    chat_history.append({'sender': 'Fast Ollama', 'text': ollama_response})
    # Start TTS in a background thread after response is sent
    def tts_playback():
        try:
            print(f"Synthesizing Ollama response with macOS 'say': {ollama_response}")
            tts_path = synthesize_speech(ollama_response, output_path="tts_output/ollama_response.aiff")
            play_audio(tts_path)
        except Exception as e:
            print(f"TTS error: {e}")
    threading.Thread(target=tts_playback, daemon=True).start()
    return jsonify({'chat_history': chat_history})

@app.route('/request_slow_feedback', methods=['POST'])
def request_slow_feedback():
    chat_history = get_chat_history()
    user_messages = [msg for msg in chat_history if msg['sender'] == 'User' and 'audio_path' in msg]
    if not user_messages:
        return jsonify({'error': 'No audio to analyze'}), 400
    latest_audio = user_messages[-1]['audio_path']
    chat_id = session.get('chat_id')
    
    # Check if analysis is already running
    current_status = get_analysis_status()
    if current_status.get('status') == 'running':
        return jsonify({'error': 'Analysis already in progress'}), 400
    
    threading.Thread(target=run_slow_ollama, args=(chat_id, latest_audio)).start()
    return jsonify({'status': 'started', 'message': 'Analysis started'})

@app.route('/get_analysis_status', methods=['GET'])
def get_analysis_status_route():
    return jsonify(get_analysis_status())

@app.route('/get_chat', methods=['GET'])
def get_chat():
    chat_history = get_chat_history()
    feedback_history = get_feedback_history()
    latest_feedback = feedback_history[-1] if feedback_history else None
    analysis_status_info = get_analysis_status()
    return jsonify({
        'chat_history': chat_history, 
        'latest_feedback': latest_feedback,
        'analysis_status': analysis_status_info
    })

if __name__ == '__main__':
    app.run(debug=True) 