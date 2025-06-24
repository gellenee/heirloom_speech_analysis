#!/usr/bin/env python3
"""
Pause Segmentation Script (Single Chunk Version)
Always creates a single chunk for the entire audio for MFA processing
"""

import librosa
import soundfile as sf
import os
import json

AUDIO_DIR = "audio_files"
AUDIO_FILENAME = "text_audio.mp3"
AUDIO_PATH = os.path.join(AUDIO_DIR, AUDIO_FILENAME)


def create_single_chunk():
    """Create a single chunk for the entire audio file"""
    print(f"Creating single chunk from {AUDIO_PATH}...")
    # Load audio
    y, sr = librosa.load(AUDIO_PATH, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Create output directory
    os.makedirs("mfa_chunks", exist_ok=True)

    # Save entire audio as single chunk
    chunk_path = "mfa_chunks/chunk_0.wav"
    sf.write(chunk_path, y, sr)

    # Create chunk boundaries
    chunk_boundaries = [{
        "chunk": "chunk_0",
        "start": 0.0,
        "end": float(duration)
    }]

    # Save chunk boundaries
    with open("mfa_chunks/chunk_boundaries.json", "w") as f:
        json.dump(chunk_boundaries, f, indent=2)

    # Create transcript file for MFA
    with open("mfa_chunks/transcript.txt", "w") as f:
        f.write("chunk_0\t<placeholder_text>\n")

    print(f"Created single chunk: {chunk_path} ({duration:.2f}s)")
    print("Created MFA transcript file: mfa_chunks/transcript.txt")
    print("Note: You'll need to replace placeholder text with actual transcript")
    return [chunk_path]


def main():
    if not os.path.exists(AUDIO_PATH):
        print(f"Error: {AUDIO_PATH} not found!")
        print(f"Please ensure your audio file is in the {AUDIO_DIR}/ directory")
        exit(1)
    try:
        create_single_chunk()
        print("Single-chunk segmentation complete!")
    except Exception as e:
        print(f"Error during segmentation: {e}")
        exit(1)

if __name__ == "__main__":
    main()