from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


# Original dataset
SOURCE_FILE = Path(r"D:\MUEL GUARD LLM FT\data\raw\muleguard_account_features.csv")

# Output folder
CSV_OUTPUT_FOLDER = Path(r"D:\MUEL GUARD LLM FT\data\processed")
JSON_OUTPUT_FOLDER = Path(r"D:\MUEL GUARD LLM FT\data\llama_factory")
CSV_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
JSON_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# Read the original CSV
df = pd.read_csv(SOURCE_FILE)

# Confirm required columns exist
required_columns = {"ACCOUNT_ID", "IS_MULE"}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Check one row per account
if df["ACCOUNT_ID"].duplicated().any():
    raise ValueError("Duplicate ACCOUNT_ID values found.")

# Check labels
valid_labels = {0, 1}

actual_labels = set(df["IS_MULE"].dropna().astype(int).unique())

if not actual_labels.issubset(valid_labels):
    raise ValueError(f"IS_MULE must contain only 0 and 1. Found: {actual_labels}")

# Check missing values
if df.isnull().any().any():
    missing_counts = df.isnull().sum()
    print("Missing values found:")
    print(missing_counts[missing_counts > 0])
    raise ValueError("Fix missing values before continuing.")

# First split:
# 70% training, 30% temporary
train_df, temporary_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["IS_MULE"]
)

# Second split:
# Temporary data becomes 15% validation and 15% test
validation_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=42,
    stratify=temporary_df["IS_MULE"]
)

# Save CSV files
train_df.to_csv(CSV_OUTPUT_FOLDER / "muleguard_train.csv", index=False)
validation_df.to_csv(CSV_OUTPUT_FOLDER / "muleguard_validation.csv", index=False)
test_df.to_csv(CSV_OUTPUT_FOLDER / "muleguard_test.csv", index=False)


# Columns used as model inputs
feature_columns = [
    column
    for column in df.columns
    if column not in ["ACCOUNT_ID", "IS_MULE"]
]


def convert_row_to_llm_example(row):
    feature_lines = []

    for column in feature_columns:
        value = row[column]
        feature_lines.append(f"{column}: {value}")

    return {
        "instruction": (
            "Classify this account as MULE or NOT_MULE. "
            "Return only one label."
        ),
        "input": "\n".join(feature_lines),
        "output": "MULE" if int(row["IS_MULE"]) == 1 else "NOT_MULE"
    }


def convert_dataframe_to_json(dataframe, output_file):
    examples = []

    for _, row in dataframe.iterrows():
        examples.append(convert_row_to_llm_example(row))

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(examples, file, indent=2)


# Save LLaMA Factory JSON files
convert_dataframe_to_json(
    train_df,
    JSON_OUTPUT_FOLDER / "muleguard_train.json"
)

convert_dataframe_to_json(
    validation_df,
    JSON_OUTPUT_FOLDER / "muleguard_validation.json"
)

convert_dataframe_to_json(
    test_df,
    JSON_OUTPUT_FOLDER / "muleguard_test.json"
)


# Print summary
print("Dataset preparation completed.")
print(f"Training rows: {len(train_df)}")
print(f"Validation rows: {len(validation_df)}")
print(f"Test rows: {len(test_df)}")

print("\nTraining labels:")
print(train_df["IS_MULE"].value_counts())

print("\nValidation labels:")
print(validation_df["IS_MULE"].value_counts())

print("\nTest labels:")
print(test_df["IS_MULE"].value_counts())

print(f"\nCSV files created in: {CSV_OUTPUT_FOLDER}")
print(f"JSON files created in: {JSON_OUTPUT_FOLDER}")
