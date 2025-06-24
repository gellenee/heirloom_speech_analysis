def parse_opensmile_arff_csv(csv_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    # Extract attribute names
    attributes = [line.split()[1] for line in lines if line.startswith('@attribute')]
    # Find the @data line
    data_idx = next((i for i, line in enumerate(lines) if line.strip() == '@data'), None)
    # Find the first non-empty line after @data
    value_line = None
    if data_idx is not None:
        for i in range(data_idx + 1, len(lines)):
            if lines[i].strip() and not lines[i].startswith('%'):
                value_line = lines[i].strip()
                break
    if value_line:
        values = value_line.split(',')
        # Remove leading/trailing whitespace from attributes and values
        attributes = [a.strip() for a in attributes]
        values = [v.strip() for v in values]
        # Map attributes to values
        features = dict(zip(attributes, values))
        return features
    else:
        print("No data found after @data line.")
        return {}

if __name__ == "__main__":
    csv_path = "opensmile_features/chunk_0_word_8.csv"
    features = parse_opensmile_arff_csv(csv_path)
    print("Available features:", list(features.keys()))
    print("\nFeature values:")
    for k, v in features.items():
        print(f"{k}: {v}")