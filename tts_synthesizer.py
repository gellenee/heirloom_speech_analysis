#!/usr/bin/env python3
"""
Text-to-Speech Synthesizer using macOS 'say' command
Provides basic offline speech synthesis for Ollama responses (macOS only)
"""

import os
import subprocess
import platform

def synthesize_speech(text, output_path="tts_output/ollama_response.aiff", audio_prompt_path=None):
    """
    Synthesize speech from text using macOS 'say' command
    Args:
        text: Text to synthesize
        output_path: Path to save the generated audio file (AIFF format)
        audio_prompt_path: Ignored (for compatibility)
    Returns:
        Path to the generated audio file, or None if failed
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Synthesizing with macOS 'say': '{text}'")
        subprocess.run(["say", "-o", output_path, text], check=True)
        print(f"TTS output saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error during TTS synthesis: {e}")
        import traceback
        traceback.print_exc()
        return None

def play_audio(audio_path):
    """Play audio file using system audio player (afplay on macOS)"""
    try:
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", str(audio_path)], check=True)
        else:
            print(f"Audio playback not supported on {system}")
    except Exception as e:
        print(f"Error playing audio: {e}")

if __name__ == "__main__":
    # Test the TTS system
    test_text = "Hello! This is a test of macOS say text-to-speech."
    out_path = synthesize_speech(test_text, output_path="tts_output/test_say.aiff")
    if out_path:
        play_audio(out_path) 