import whisper
import json

# Load Whisper model
model = whisper.load_model("medium")  # or "small", "large"


# Transcribe with word timestamps
result = model.transcribe("audio.wav", word_timestamps=True)

# Save output to JSON
with open("whisper_words.json", "w") as f:
    word_data = []
    for segment in result['segments']:
        for word in segment['words']:
            word_data.append({
                "word": word['word'].strip('.,?!').lower(),
                "start": word['start'],
                "end": word['end']
            })
    json.dump(word_data, f, indent=2)

# Also print nicely
for word in word_data:
    print(f"{word['word']} — start: {word['start']:.2f}s, end: {word['end']:.2f}s")
