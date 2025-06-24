import pandas as pd
import subprocess
import os

# Paths
csv_path = "segments_words.csv"  # or segments_phones.csv
opensmile_config = "/Users/aishanibal/opensmile-3.0.2-macos-armv8/config/egemaps/v01a/eGeMAPSv01a.conf"
output_dir = "opensmile_features"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)

for i, row in df.iterrows():
    wav_path = row["wav_path"]
    start = row["start"]
    end = row["end"]
    unit = row["unit"]
    segment_type = row["type"]

    output_csv = os.path.join(output_dir, f"{os.path.basename(wav_path).replace('.wav','')}_{segment_type}_{i}.csv")

    subprocess.run([
        "/Users/aishanibal/opensmile-3.0.2-macos-armv8/bin/SMILExtract",
        "-C", opensmile_config,
        "-I", wav_path,
        "-start", str(start),
        "-end", str(end),
        "-O", output_csv
    ])
