import os
import pandas as pd
import json

# Directory where openSMILE .csv outputs are saved
input_dir = "opensmile_features"
output_json = "llm_ready_pronunciation_data.json"

# Feature mapping from openSMILE to simplified keys
feature_map = {
    "F0semitoneFrom27.5Hz_sma3nz_amean": "F0",
    "jitterLocal_sma3nz_amean": "jitter",
    "shimmerLocaldB_sma3nz_amean": "shimmer",
    "loudness_sma3_amean": "loudness"
}

data = []

for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path, comment='@')

        # Skip if the file is empty
        if df.empty:
            continue

        features = df.iloc[0]
        pronunciation = {}
        for smile_name, simple_name in feature_map.items():
            pronunciation[simple_name] = float(features.get(smile_name, 0.0))

        # Extract word from filename (e.g., "audio_word_5.csv" → "word")
        parts = file.replace(".csv", "").split("_")
        word = parts[1] if len(parts) > 1 else "unknown"

        entry = {
            "word": word,
            "pronunciation_features": pronunciation,
            "expected_features": {
                "F0": 120.0,
                "jitter": 0.01,
                "shimmer": 0.005,
                "loudness": 0.25
            },
            "transcript": word,
            "feedback_goal": "help the user speak more confidently and with correct vowel stress"
        }
        data.append(entry)

# Save to JSON
with open(output_json, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved {len(data)} entries to {output_json}")
