import glob
import pandas as pd
import numpy as np
import json

def parse_opensmile_arff_csv(csv_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    attributes = [line.split()[1] for line in lines if line.startswith('@attribute')]
    data_idx = next((i for i, line in enumerate(lines) if line.strip() == '@data'), None)
    value_line = None
    if data_idx is not None:
        for i in range(data_idx + 1, len(lines)):
            if lines[i].strip() and not lines[i].startswith('%'):
                value_line = lines[i].strip()
                break
    if value_line:
        values = value_line.split(',')
        attributes = [a.strip() for a in attributes]
        values = [v.strip() for v in values]
        features = dict(zip(attributes, values))
        return features
    else:
        return {}

def get_speaking_rate_from_whisper():
    try:
        with open("whisper_words.json", "r") as f:
            words = json.load(f)
        if not words:
            return 0.0
        first_word_start = min(w["start"] for w in words)
        last_word_end = max(w["end"] for w in words)
        total_duration = last_word_end - first_word_start
        if total_duration > 0:
            return len(words) / total_duration
        else:
            return 0.0
    except Exception as e:
        print(f"Warning: Could not calculate speaking rate from whisper_words.json: {e}")
        return 0.0

# Collect features from all word segments in temporal order
temporal_features = []
word_sequence = []

for file in sorted(glob.glob("opensmile_features/*.csv")):
    features = parse_opensmile_arff_csv(file)
    # Remove non-numeric fields
    core_features = {k: v for k, v in features.items() if k not in ['name', 'class']}
    
    # Convert to float
    feature_vector = {}
    for k in core_features:
        try:
            feature_vector[k] = float(core_features[k])
        except ValueError:
            feature_vector[k] = None
    
    temporal_features.append(feature_vector)
    word_sequence.append(file.split('/')[-1].replace('.csv', ''))

# Create temporal DataFrame
df_temporal = pd.DataFrame(temporal_features, index=word_sequence)

# Analyze temporal patterns
temporal_analysis = {
    'word_sequence': word_sequence,
    'temporal_features': df_temporal.to_dict('index'),
    'feature_trajectories': {}
}

# Calculate temporal patterns for key features
key_features = ['F0semitoneFrom27.5Hz_sma3nz_amean', 'loudness_sma3_amean', 
                'mfcc1_sma3_amean', 'jitterLocal_sma3nz_amean', 'shimmerLocaldB_sma3nz_amean']

for feature in key_features:
    if feature in df_temporal.columns:
        values = df_temporal[feature].dropna()
        if len(values) > 0:
            temporal_analysis['feature_trajectories'][feature] = {
                'values': values.tolist(),
                'word_positions': values.index.tolist(),
                'trend': 'increasing' if values.iloc[-1] > values.iloc[0] else 'decreasing',
                'variability': values.std(),
                'range': values.max() - values.min()
            }

# Calculate speaking rate using global word times from whisper_words.json
speaking_rate = get_speaking_rate_from_whisper()
temporal_analysis['speaking_rate'] = speaking_rate

# Save temporal analysis
with open("temporal_phrase_analysis.json", "w") as f:
    json.dump(temporal_analysis, f, indent=2)

print("Temporal analysis saved to temporal_phrase_analysis.json")
print(f"Analyzed {len(word_sequence)} words in sequence")
print(f"Key features tracked: {list(temporal_analysis['feature_trajectories'].keys())}")

# Print temporal patterns
for feature, data in temporal_analysis['feature_trajectories'].items():
    print(f"\n{feature}:")
    print(f"  Trend: {data['trend']}")
    print(f"  Variability: {data['variability']:.4f}")
    print(f"  Range: {data['range']:.4f}")
    print(f"  Values over time: {[f'{v:.2f}' for v in data['values']]}")