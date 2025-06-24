#!/usr/bin/env python3
"""
Generate MFA chunk transcripts and .lab files from Whisper word-level output and chunk boundaries.
For each chunk, assign the words whose timestamps fall within the chunk's start/end times.
Writes mfa_chunks/transcript.txt and .lab files for MFA alignment.
"""
import json
import os

# Load Whisper word-level output
with open("whisper_words.json", "r") as f:
    words = json.load(f)

# Load chunk boundaries
with open("mfa_chunks/chunk_boundaries.json", "r") as f:
    chunk_boundaries = json.load(f)

chunk_dir = "mfa_chunks"

# Assign words to chunks using robust overlap (at least half the word's duration overlaps with the chunk)
chunk_transcripts = {chunk_info["chunk"]: [] for chunk_info in chunk_boundaries}
for word in words:
    w_start = word["start"]
    w_end = word["end"]
    w_dur = w_end - w_start
    for chunk_info in chunk_boundaries:
        chunk_name = chunk_info["chunk"]
        c_start = chunk_info["start"]
        c_end = chunk_info["end"]
        # Calculate overlap
        overlap = max(0, min(w_end, c_end) - max(w_start, c_start))
        if w_dur > 0 and overlap / w_dur >= 0.5:
            chunk_transcripts[chunk_name].append(word["word"])
            break

# Write transcript.txt
with open(os.path.join(chunk_dir, "transcript.txt"), "w") as f:
    for chunk_name in chunk_transcripts:
        transcript = " ".join(chunk_transcripts[chunk_name])
        if not transcript:
            transcript = "<placeholder_text>"
        f.write(f"{chunk_name}\t{transcript}\n")

# Write .lab files for each chunk
for chunk_name, words_list in chunk_transcripts.items():
    transcript = " ".join(words_list)
    if not transcript:
        transcript = "<placeholder_text>"
    lab_path = os.path.join(chunk_dir, f"{chunk_name}.lab")
    with open(lab_path, "w") as lab_file:
        lab_file.write(transcript.strip() + "\n")

print(f"Wrote MFA transcript and .lab files for {len(chunk_transcripts)} chunks using chunk boundaries.") 