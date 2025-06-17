import torch
import torchaudio
import os
import json

# Load Silero VAD
torch.set_num_threads(1)
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, read_audio, _, _) = utils

# Load audio
AUDIO_PATH = 'audio.wav'
audio = read_audio(AUDIO_PATH, sampling_rate=16000)
sample_rate = 16000

# Get speech timestamps
speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=sample_rate)

# Convert sample indices to seconds
vad_segments = [
    {'start': ts['start'] / sample_rate, 'end': ts['end'] / sample_rate}
    for ts in speech_timestamps
]

# Load Whisper transcript
with open("whisper_words.json", "r") as f:
    whisper_words = json.load(f)

# Output folder for MFA chunks
os.makedirs("mfa_chunks", exist_ok=True)

# Slice audio + create .lab file for each chunk
for i, seg in enumerate(vad_segments):
    start_sample = int(seg['start'] * sample_rate)
    end_sample = int(seg['end'] * sample_rate)
    chunk = audio[start_sample:end_sample].unsqueeze(0)
    
    chunk_path = f"mfa_chunks/chunk_{i}.wav"
    torchaudio.save(chunk_path, chunk, sample_rate)

    # Get Whisper words in this segment
    transcript = [
        w['word']
        for w in whisper_words
        if seg['start'] <= w['start'] <= seg['end']
    ]
    
    # Save transcript as .lab file
    with open(f"mfa_chunks/chunk_{i}.lab", "w") as f:
        f.write(" ".join(transcript))

    print(f"Saved {chunk_path} with transcript: {' '.join(transcript)}")
