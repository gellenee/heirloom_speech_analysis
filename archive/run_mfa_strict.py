import subprocess
import os

# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
corpus_dir = os.path.join(current_dir, "mfa_chunks")
dictionary_path = "english_us_mfa"
acoustic_model = "english_mfa"
output_dir = os.path.join(current_dir, "mfa_output")

# Ensure output directory exists and is empty
os.makedirs(output_dir, exist_ok=True)
for file in os.listdir(output_dir):
    os.remove(os.path.join(output_dir, file))

# Clean MFA cache
clean_command = ["mfa", "clean"]
subprocess.run(clean_command, capture_output=True, text=True)

# Command to run MFA alignment with strict settings
command = [
    "mfa", "align",
    "--clean",  # Clean the output directory before processing
    "--overwrite",  # Overwrite existing files
    corpus_dir,
    dictionary_path,
    acoustic_model,
    output_dir
]

# Run the command
result = subprocess.run(command, capture_output=True, text=True)

# Print the output or error for logging
print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)