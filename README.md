# Cancer Family History Extraction From Japanese Electronic Health Records

[日本語](README.ja.md)

Code for the manuscript "Performance and Error Analysis of Large Language Models for Extracting Cancer Family History From Japanese Electronic Health Records of Patients With Colorectal Cancer."

No patient data, model weights, or experimental outputs are included. The clinical data cannot be shared. Comments and messages in the code are written in Japanese.

## Contents

| Path | Purpose |
|---|---|
| `src/family_history/selection.py` | Selection of evaluation and rule-generation patients |
| `src/family_history/prompts.py` | Prompts |
| `src/family_history/inference.py` | Extraction with LLMs |
| `src/family_history/normalize.py` | Output validation and normalization |
| `src/family_history/simple_rules.py` | Simple rules |
| `src/family_history/rule_generation.py` | Rule generation by Qwen3.5 |
| `src/family_history/rule_engine.py` | Rule validation and application |
| `src/family_history/evaluation.py` | Evaluation |
| `scripts/compute_bootstrap_ci.py` | 95% CIs |
| `scripts/summarize_age_sex.py` | Sex and age of the evaluated patients |

## Input

- Candidate texts: one JSON file per patient (`<patient ID>.json`) whose keys are dates and whose values are `{"fam_text": text}`. The texts are progress notes retrieved by keyword search.
- Reference standard: a CSV file with the columns `patient_id`, `date`, `decision` (`family_history`, `no_history`, or `undecidable`), and `gold_family_history_json`.

## Usage

Run the commands from this directory with `PYTHONPATH=src`. Each command shows its options with `--help`.

```
PYTHONPATH=src python -m family_history select-evaluation       # select the evaluation patients
PYTHONPATH=src python -m family_history infer                   # extraction with an LLM (4 prompting conditions)
PYTHONPATH=src python -m family_history simple-rules            # extraction with the simple rules
PYTHONPATH=src python -m family_history select-development      # select the rule-generation patients
PYTHONPATH=src python -m family_history prepare-rule-generation # pseudo-labels and example selection
PYTHONPATH=src python -m family_history generate-rules          # rule generation and one regeneration
PYTHONPATH=src python -m family_history apply-rules             # application of the generated rules
PYTHONPATH=src python -m family_history evaluate                # evaluation against the reference standard
python scripts/compute_bootstrap_ci.py                          # 95% CIs
```

The context length of `infer` (`--n-ctx`) was 4096 for LLM-jp and 8192 for the other models.

LLM inference and rule generation require `llama-cpp-python` and a CUDA-capable GPU, rule application requires `regex`, and the bootstrap requires `numpy`. Model weights are available from the model cards cited in the manuscript and are subject to their own licenses.

## License

The code is released under the MIT License (see `LICENSE`). The license does not cover model weights or clinical data.
