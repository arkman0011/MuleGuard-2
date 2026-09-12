from pathlib import Path
from datetime import datetime
import csv
import json

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT = Path(r"D:\MUEL GUARD LLM FT")
ADAPTER_FOLDER = PROJECT / "models" / "run_clean"
TEST_FILE = PROJECT / "data" / "llama_factory" / "muleguard_test.json"
LABELS = ["NOT_MULE", "MULE", "INVALID"]


def main():
    # Read the test examples and check their answer labels.
    examples = json.loads(TEST_FILE.read_text(encoding="utf-8-sig"))
    if not examples:
        raise ValueError("The test file is empty.")
    for example in examples:
        if example["output"] not in LABELS[:2]:
            raise ValueError("Test answers must be MULE or NOT_MULE.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use the training Python environment.")

    # Load the saved tokenizer, matching base model, and trained adapter.
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_FOLDER))
    adapter_config = PeftConfig.from_pretrained(str(ADAPTER_FOLDER))
    base_model = AutoModelForCausalLM.from_pretrained(
        adapter_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_FOLDER))
    model.eval()

    true_labels = []
    predicted_labels = []
    predictions = []

    # Predict one account at a time to limit GPU memory use.
    for number, example in enumerate(examples, start=1):
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": example["instruction"] + "\n" + example["input"],
            },
        ]
        # The correct output is never included in the model's prompt.
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        prompt_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            output_ids[0, prompt_length:], skip_special_tokens=True
        ).strip()
        prediction = answer if answer in LABELS[:2] else "INVALID"
        true_labels.append(example["output"])
        predicted_labels.append(prediction)
        predictions.append({
            "test_row": number,
            "true_label": example["output"],
            "predicted_label": prediction,
            "raw_answer": answer,
            "correct": prediction == example["output"],
        })
        print(f"Account {number}/{len(examples)}: {prediction}", flush=True)

    # INVALID answers remain in the results and count as incorrect.
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(
        true_labels, predicted_labels, labels=LABELS[:2],
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=LABELS)[:2, :]
    invalid_count = predicted_labels.count("INVALID")
    summary = {
        "test_accounts": len(examples),
        "accuracy": accuracy,
        "invalid_answers": invalid_count,
        "per_class_metrics": {label: report[label] for label in LABELS[:2]},
        "macro_f1": sum(report[label]["f1-score"] for label in LABELS[:2]) / 2,
        "confusion_matrix": matrix.tolist(),
        "matrix_rows_actual": LABELS[:2],
        "matrix_columns_predicted": LABELS,
        "adapter_folder": str(ADAPTER_FOLDER),
        "base_model": adapter_config.base_model_name_or_path,
    }

    # Each evaluation gets a new folder so previous results are preserved.
    results_folder = PROJECT / "test_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    results_folder.mkdir(parents=True, exist_ok=False)
    with (results_folder / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    (results_folder / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Show a normal 2x2 matrix unless invalid answers require a third column.
    visible_matrix = matrix if invalid_count else matrix[:, :2]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.imshow(visible_matrix, cmap="Blues")
    ax.set_xticks(range(visible_matrix.shape[1]), LABELS[:visible_matrix.shape[1]])
    ax.set_yticks(range(2), LABELS[:2])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("MuleGuard 2: test confusion matrix")
    for row in range(2):
        for column in range(visible_matrix.shape[1]):
            value = int(visible_matrix[row, column])
            color = "white" if value > visible_matrix.max() / 2 else "black"
            ax.text(column, row, value, ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(results_folder / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Mule precision: {report['MULE']['precision']:.2%}")
    print(f"Mule recall: {report['MULE']['recall']:.2%}")
    print(f"Mule F1: {report['MULE']['f1-score']:.2%}")
    print(f"Invalid answers: {invalid_count}")
    print("Confusion matrix (rows: actual; columns: NOT_MULE, MULE, INVALID):")
    print(matrix)
    print(f"Results saved in: {results_folder}")


if __name__ == "__main__":
    main()
