import pandas as pd
import textgrid
import os

# === 1. Parse openSMILE CSV (e.g., from IS13_ComParE.conf or emobase.conf) ===
def parse_opensmile_csv(csv_path):
    # Try both ARFF/CSV and semicolon-delimited CSV
    try:
        df = pd.read_csv(csv_path, delimiter=';')
    except Exception:
        df = pd.read_csv(csv_path)
    if 'name' in df.columns:
        df = df.drop(columns=['name'])
    return df

# === 2. Parse MFA TextGrid and extract alignment stats ===
def parse_textgrid(textgrid_path):
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    alignment_info = []

    for tier in tg.tiers:
        if "word" in tier.name.lower():
            for interval in tier.intervals:
                word = interval.mark.strip()
                duration = interval.maxTime - interval.minTime
                if word:  # skip silence
                    alignment_info.append({
                        'word': word,
                        'start': interval.minTime,
                        'end': interval.maxTime,
                        'duration': duration
                    })

    df_align = pd.DataFrame(alignment_info)
    return df_align

# === 3. Example usage ===
# Update these paths to your actual files
opensmile_csv_path = "opensmile_features/chunk_0_word_4.csv"  # Example file
textgrid_path = "mfa_output/chunk_0.TextGrid"  # Example file

# Load data
opensmile_features = parse_opensmile_csv(opensmile_csv_path)
alignment_data = parse_textgrid(textgrid_path)

# Show parsed data
print("OpenSMILE Features:")
print(opensmile_features.head())
print("\nMFA Alignment Data:")
print(alignment_data.head()) 