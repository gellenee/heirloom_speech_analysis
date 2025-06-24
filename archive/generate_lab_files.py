#!/usr/bin/env python3
"""
Generate .lab files for each chunk in mfa_chunks/ using transcript.txt.
Each .lab file will contain the transcript for its corresponding chunk.
"""
import os

chunk_dir = "mfa_chunks"
transcript_path = os.path.join(chunk_dir, "transcript.txt")

with open(transcript_path, "r") as f:
    for line in f:
        if not line.strip():
            continue
        chunk_name, transcript = line.strip().split("\t", 1)
        lab_path = os.path.join(chunk_dir, f"{chunk_name}.lab")
        with open(lab_path, "w") as lab_file:
            lab_file.write(transcript.strip() + "\n")
print("Generated .lab files for all chunks.") 