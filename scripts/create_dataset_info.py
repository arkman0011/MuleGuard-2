from pathlib import Path
import json


data_folder = Path(r"D:\MUEL GUARD LLM FT\data\llama_factory")

dataset_info = {
    "muleguard_train": {
        "file_name": "muleguard_train.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    },
    "muleguard_validation": {
        "file_name": "muleguard_validation.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    },
    "muleguard_test": {
        "file_name": "muleguard_test.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }
}

output_file = data_folder / "dataset_info.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(dataset_info, file, indent=2)

print(f"Created: {output_file}")
