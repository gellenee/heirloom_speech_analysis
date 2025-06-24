#!/usr/bin/env python3
"""
Whisper Transcription Script
Transcribes audio using OpenAI Whisper with word-level timestamps
"""

import whisper
import json
import os
import sys

# Configuration - matches pipeline settings
AUDIO_PATH = sys.argv[1] if len(sys.argv) > 1 else "audio_files/text_audio.mp3"
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "whisper_words.json"

def transcribe_audio():
    """Transcribe audio using Whisper with word timestamps"""
    print(f"Loading Whisper model...")
    model = whisper.load_model("base")
    
    print(f"Transcribing {AUDIO_PATH}...")
    result = model.transcribe(AUDIO_PATH, word_timestamps=True)
    
    # Extract word-level timestamps
    words = []
    for segment in result["segments"]:
        for word_info in segment["words"]:
            words.append({
                "word": word_info["word"].strip(),
                "start": word_info["start"],
                "end": word_info["end"]
            })
    
    # Save word-level data
    with open(OUTPUT_PATH, "w") as f:
        json.dump(words, f, indent=2)
    
    print(f"Transcription complete! Found {len(words)} words")
    print(f"Saved word timestamps to {OUTPUT_PATH}")
    
    return words

if __name__ == "__main__":
    if not os.path.exists(AUDIO_PATH):
        print(f"Error: {AUDIO_PATH} not found!")
        print(f"Please ensure your audio file is in the correct directory")
        exit(1)
    
    transcribe_audio()
