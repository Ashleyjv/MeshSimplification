import json
import os

# Path template for the dataset JSON files
json_file_template = "dataset/city.stl_{}.json"

# Output dictionary to store extracted data
results = []

# Iterate through the files
for i in range(1, 11):
    file_path = json_file_template.format(i)
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)

                # Extract relevant fields
                original = data.get("original", {})
                simplified = data.get("simplified", {})

                results.append({
                    "file": file_path,
                    "original": {
                        "rrtdist": original.get("rrtdist"),
                        "rrttime": original.get("rrttime"),
                        "astardist": original.get("astardist"),
                        "astartime": original.get("astartime"),
                    },
                    "simplified": {
                        "rrtdist": simplified.get("rrtdist"),
                        "rrttime": simplified.get("rrttime"),
                        "astardist": simplified.get("astardist"),
                        "astartime": simplified.get("astartime"),
                    }
                })
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")

# Save the results to a new JSON file for convenience
output_path = "extracted_results.json"
with open(output_path, 'w') as outfile:
    json.dump(results, outfile, indent=4)

print(f"Extraction complete. Results saved to {output_path}.")
