import torch
import torchaudio
import os
import json
from collections import defaultdict

# --- Setup ---
torch.set_num_threads(1)
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, read_audio, _, _) = utils

AUDIO_PATH = 'audio.wav'
sample_rate = 16000
audio = read_audio(AUDIO_PATH, sampling_rate=sample_rate)

# --- Load Whisper word-level transcript ---
with open("whisper_words.json", "r") as f:
    whisper_words = json.load(f)

# --- Get Silero VAD segments ---
speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=sample_rate)
vad_segments = [
    {'start': ts['start'] / sample_rate, 'end': ts['end'] / sample_rate}
    for ts in speech_timestamps
]

# --- Create output folder ---
os.makedirs("mfa_chunks", exist_ok=True)

def find_pause_boundaries(words, pause_threshold=0.5):
    """
    Find natural speech boundaries based on pauses between words.
    A new chunk starts when there's a pause longer than the threshold.
    """
    if not words:
        return []
    
    chunks = []
    current_chunk = [words[0]]
    
    for i in range(1, len(words)):
        current_word = words[i]
        prev_word = words[i-1]
        
        # Calculate gap between words
        gap = current_word['start'] - prev_word['end']
        
        # Start new chunk if there's a significant pause
        if gap > pause_threshold:
            chunks.append(current_chunk)
            current_chunk = [current_word]
        else:
            current_chunk.append(current_word)
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# Group words into chunks based on pauses
chunks = find_pause_boundaries(whisper_words, pause_threshold=0.5)

# --- Create chunks from pause boundaries ---
for i, chunk in enumerate(chunks):
    if not chunk:
        continue
        
    # Get start and end times with buffer
    BUFFER = 0.3  # seconds before and after
    start_time = max(0, chunk[0]['start'] - BUFFER)
    end_time = min(len(audio) / sample_rate, chunk[-1]['end'] + BUFFER)
    
    # Convert to samples
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    # Extract audio chunk
    chunk_audio = audio[start_sample:end_sample].unsqueeze(0)
    
    # Save audio chunk
    chunk_path = f"mfa_chunks/chunk_{i}.wav"
    torchaudio.save(chunk_path, chunk_audio, sample_rate)
    
    # Save transcript
    transcript = [w['word'].strip() for w in chunk]
    with open(f"mfa_chunks/chunk_{i}.lab", "w") as f:
        f.write(" ".join(transcript))
    
    print(f"Saved {chunk_path} with transcript: {' '.join(transcript)}")

# --- Handle any remaining words that might have been missed ---
all_processed_words = set()
for chunk in chunks:
    for word in chunk:
        all_processed_words.add(word['word'])

missed_words = [w for w in whisper_words if w['word'] not in all_processed_words]

# Process missed words
for i, word in enumerate(missed_words):
    start_time = max(0, word['start'] - BUFFER)
    end_time = min(len(audio) / sample_rate, word['end'] + BUFFER)
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    chunk = audio[start_sample:end_sample].unsqueeze(0)
    
    chunk_path = f"mfa_chunks/missed_{i}.wav"
    torchaudio.save(chunk_path, chunk, sample_rate)
    
    with open(f"mfa_chunks/missed_{i}.lab", "w") as f:
        f.write(word['word'].strip())
    
    print(f"Saved {chunk_path} with transcript: {word['word']}")