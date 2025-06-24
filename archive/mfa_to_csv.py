import os
import pandas as pd
from textgrid import TextGrid

# === Set Paths ===
textgrid_dir = "mfa_output"       # MFA TextGrid output directory
wav_dir = "mfa_chunks"            # Original audio files used in MFA
word_output_csv = "segments_words.csv"
phone_output_csv = "segments_phones.csv"

word_segments = []
phone_segments = []

# === Extract from Each TextGrid ===
for tg_file in os.listdir(textgrid_dir):
    if tg_file.endswith(".TextGrid"):
        tg_path = os.path.join(textgrid_dir, tg_file)
        audio_basename = os.path.splitext(tg_file)[0]
        wav_path = os.path.join(wav_dir, audio_basename + ".wav")

        tg = TextGrid.fromFile(tg_path)

        for tier in tg.tiers:
            tier_name = tier.name.lower()
            for interval in tier.intervals:
                label = interval.mark.strip()
                start = interval.minTime
                end = interval.maxTime

                if label == "":  # skip silences
                    continue

                if tier_name == "words":
                    word_segments.append({
                        "wav_path": wav_path,
                        "unit": label,
                        "start": start,
                        "end": end,
                        "type": "word"
                    })

                elif tier_name == "phones":
                    phone_segments.append({
                        "wav_path": wav_path,
                        "unit": label,
                        "start": start,
                        "end": end,
                        "type": "phone"
                    })

# === Save to CSVs ===
df_words = pd.DataFrame(word_segments)
df_phones = pd.DataFrame(phone_segments)

df_words.to_csv(word_output_csv, index=False)
df_phones.to_csv(phone_output_csv, index=False)

print(f"[✓] Saved {len(df_words)} word segments to {word_output_csv}")
print(f"[✓] Saved {len(df_phones)} phone segments to {phone_output_csv}")
