#!/usr/bin/env python3
"""
Detect Mispronunciations by comparing expected phoneme sequences (from CMUdict)
to the MFA alignment for each word in segments_words.csv and segments_phones.csv.
Outputs a report and a JSON file with detected mispronunciations.
Uses Levenshtein distance to allow for natural variation.
"""
import pandas as pd
import json
import os
from collections import defaultdict

try:
    import nltk
    nltk.data.find('corpora/cmudict')
except LookupError:
    import nltk
    nltk.download('cmudict')
from nltk.corpus import cmudict

def levenshtein(seq1, seq2):
    """Compute Levenshtein distance between two sequences."""
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # deletion
                d[i][j - 1] + 1,      # insertion
                d[i - 1][j - 1] + cost  # substitution
            )
    return d[n][m]

# Load CMU Pronouncing Dictionary
cmu = cmudict.dict()

# Load word and phone segments
words_df = pd.read_csv("segments_words.csv")
phones_df = pd.read_csv("segments_phones.csv")

mispronunciations = []
report_lines = []

for _, word_row in words_df.iterrows():
    word = word_row['unit'].lower()
    wav_path = word_row['wav_path']
    w_start = word_row['start']
    w_end = word_row['end']

    # Find all phones in the same wav_path that overlap with the word interval
    aligned_phones = [
        row['unit'].lower()
        for _, row in phones_df.iterrows()
        if row['wav_path'] == wav_path and row['start'] >= w_start and row['end'] <= w_end
    ]

    # Get expected phonemes from CMUdict
    expected_prons = cmu.get(word, [])
    if not expected_prons:
        # Not in CMUdict, skip
        continue
    # Use the first pronunciation variant
    expected_phones = [p.lower() for p in expected_prons[0]]

    # Compute Levenshtein distance
    if not aligned_phones:
        edit_distance = len(expected_phones)
    else:
        edit_distance = levenshtein(expected_phones, aligned_phones)

    # Flag as mispronounced if edit distance > 3 or no aligned phones
    if edit_distance > 3 or not aligned_phones:
        mispronunciations.append({
            "word": word,
            "wav_path": wav_path,
            "start": w_start,
            "end": w_end,
            "expected_phones": expected_phones,
            "aligned_phones": aligned_phones,
            "edit_distance": edit_distance
        })
        report_lines.append(
            f"Word: '{word}'\n  Expected: {expected_phones}\n  Aligned:  {aligned_phones}\n  Edit distance: {edit_distance}\n  File: {wav_path}  Start: {w_start:.2f}  End: {w_end:.2f}\n"
        )

# Output report
with open("mispronunciation_report.txt", "w") as f:
    if mispronunciations:
        f.write("MISPRONUNCIATIONS DETECTED (edit distance > 3 or no aligned phones):\n\n")
        f.write("\n".join(report_lines))
    else:
        f.write("No mispronunciations detected.\n")

with open("mispronunciations.json", "w") as f:
    json.dump(mispronunciations, f, indent=2)

print(f"Checked {len(words_df)} words. Found {len(mispronunciations)} possible mispronunciations.")
print("See mispronunciation_report.txt and mispronunciations.json for details.") 