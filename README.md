# MuleGuard 2

MuleGuard 2 is an experimental LLM-based classification pipeline for identifying whether an account is likely to be a mule account from engineered account-level transaction features.

## Project summary

- Dataset: 1,000 account records.
- Input: 38 engineered account-level features. `ACCOUNT_ID` is used as an identifier and `IS_MULE` is the target label.
- Splits: 700 training records, 150 validation records, and 150 test records.
- Model: `unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit`.
- Fine-tuning: supervised fine-tuning with LoRA, 4-bit quantization, LLaMA Factory, and Unsloth GPU optimization.
- Evaluation: label parsing, accuracy, classification report, confusion matrix, and prediction export.

## Repository structure

```text
data/
  raw/                 Original account feature CSV
  processed/           Train, validation, and test CSV files
  llama_factory/       LLaMA Factory JSON datasets and dataset_info.json
configs/               Fine-tuning YAML configuration
scripts/               Dataset preparation, balancing, and evaluation scripts
results/               Saved evaluation outputs and plots
```

The local Python environment, downloaded model weights, training checkpoints, and caches are intentionally excluded from Git. They can be recreated locally by installing the required dependencies and downloading the base model.

## Reproduce the workflow

Run commands from PowerShell:

```powershell
cd "D:\MUEL GUARD LLM FT\LlamaFactory"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Prepare or refresh the dataset:

```powershell
\.venv\Scripts\python.exe "D:\MUEL GUARD LLM FT\scripts\prepare_muleguard_dataset.py"
```

Start fine-tuning with LLaMA Factory:

```powershell
\.venv\Scripts\llamafactory-cli.exe train "D:\MUEL GUARD LLM FT\configs\muleguard_ft_clean.yaml"
```

Evaluate the saved adapter on the test set:

```powershell
\.venv\Scripts\python.exe "D:\MUEL GUARD LLM FT\scripts\evaluate_muleguard_test.py"
```

The evaluation script saves predictions, metrics, and a confusion-matrix image under `results/`.

## Measured baseline run

The completed baseline run trained for 3 epochs. Training took 20 minutes 51.88 seconds, and evaluation processed 150 test accounts in 27.46 seconds, approximately 183 milliseconds per account. The measured test accuracy was 84%.

The test confusion matrix showed that this baseline predicted every test record as `NOT_MULE`. Therefore, the accuracy reflects the class distribution and does not demonstrate useful mule detection. The project should be treated as an experimental baseline, with class balancing and further calibration required before deployment or use in a high-stakes decision.

## Data and responsible use

Do not commit personally identifiable, confidential, or production banking data. This repository is for experimentation and reproducibility. Model outputs must not be used as the sole basis for blocking, closing, or otherwise taking action against an account.
