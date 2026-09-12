from pathlib import Path
import json
import random


DATA_FOLDER = Path(r"D:\MUEL GUARD LLM FT\data\llama_factory")
INPUT_FILE = DATA_FOLDER / "muleguard_train.json"
OUTPUT_FILE = DATA_FOLDER / "muleguard_train_balanced.json"

random.seed(42)

examples = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))

mule_examples = [example for example in examples if example["output"] == "MULE"]
not_mule_examples = [example for example in examples if example["output"] == "NOT_MULE"]

if not mule_examples or not not_mule_examples:
    raise ValueError("Both MULE and NOT_MULE examples are required.")

# Repeat MULE examples until the two classes are close in size.
balanced_mule_examples = []
while len(balanced_mule_examples) < len(not_mule_examples):
    balanced_mule_examples.extend(mule_examples)

balanced_mule_examples = balanced_mule_examples[:len(not_mule_examples)]

balanced_examples = not_mule_examples + balanced_mule_examples
random.shuffle(balanced_examples)

OUTPUT_FILE.write_text(
    json.dumps(balanced_examples, indent=2),
    encoding="utf-8"
)

print(f"Original training examples: {len(examples)}")
print(f"Original MULE examples: {len(mule_examples)}")
print(f"Original NOT_MULE examples: {len(not_mule_examples)}")
print(f"Balanced training examples: {len(balanced_examples)}")
print(f"Balanced MULE examples: {sum(e['output'] == 'MULE' for e in balanced_examples)}")
print(f"Balanced NOT_MULE examples: {sum(e['output'] == 'NOT_MULE' for e in balanced_examples)}")
print(f"Created: {OUTPUT_FILE}")
