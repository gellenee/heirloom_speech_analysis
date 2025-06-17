import subprocess

# Set paths
corpus_dir = "mfa_chunks"
acoustic_model = "english_us_mfa"
dictionary_path = "english_mfa"
output_dir = "mfa_output"

# Command to run MFA alignment
command = [
    "mfa", "align",
    corpus_dir,
    acoustic_model,
    dictionary_path,
    output_dir
]

# Run the command
result = subprocess.run(command, capture_output=True, text=True)

# Print the output or error for logging
print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)
