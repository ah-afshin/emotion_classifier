# Emotion Classifier

A multilabel classification ML project to classify emotions in given English sentences
using PyTorch and HuggingFace Transformers.

- [Overview](#emotion-classifier)
- [Getting Started](#getting-started)
- [Components](#components)
- [Results](#results)
- [Issues and Future Work](#issues-and-future-work)
- [License](#license)

---

## Getting Started
The project is formatted as a Python package. After cloning the repository and opening its root directory, install it in a virtual environment with:
```bash
pip install -e .
```

Configure a model in a YAML file ([config.yaml](config.yaml) is an example), then train it with the following command. Pass your configuration file and output directory as arguments. The best checkpoint (`best_model.pt`) and training log will be saved in a new run directory:
```bash
emotion-train --config <your-config-path> --output-dir <your-outputs-path>
```

Next, find the optimal thresholds for your model. Pass the run directory as the model path; the thresholds will be saved to `thresholds.json` in that directory:
```bash
emotion-tune-thresholds --model-path <your-run-path>
```

Evaluate the model with the following command. Results will be saved in an `eval/` subdirectory of the run directory:
```bash
emotion-eval --model-path <your-run-path>
```

You can also try a prediction on a sentence:
```bash
emotion-predict --model-path <your-run-path> --text "<your sentence>"
```

---

## Components
- `cli/` here are all the entry points, there's a file for each. these commands just call the corespondig runner.
- `data/` there's a preprocessor that processes the data and loads it into a DataLoader.
- `evaluation/` here are all the codes for evaluations and tests called by runner functions in `runner.py`.
  - `evaluator.py` contains the functions to test trained model.
  - `threshold_tuner.py` contains the functions to find the optimal threshold for each model.
  - `dataset_analysis.py` contains functions to diagnose inherited structures in dataset.
- `inference/` here are the functions used to predict emotions in a sentece.
- `models/` model architectures and model factory are here.
- `training/` here are the functions used to train the models.
- `utils/` side utilities and helper functions are here.
- `tools/visualisation/` some tools to visualize the outputs.

---

## Results
[here](report.md) is the full report on the experiments and the results.

I used the GoEmotions dataset, an imbalanced multilabel collection of English sentences. The [report's dataset section](report.md#data-and-evaluation) includes the dataset analysis plots.

| Approach | Test run | Micro-F1 | Macro-F1 | Hamming loss | Reported training time* |
|---|---|---:|---:|---:|---:|
| BiLSTM, last-token | [2025-11-29](outputs/bilstm-last-token/2025-11-29_13-08/eval/test_results.json) | 48.98% | 39.01% | 0.0496 | 1:26 |
| BiLSTM, max-pool | [2025-11-28](outputs/bilstm-max-pool/2025-11-28_23-30/eval/test_results.json) | 54.76% | 44.01% | 0.0425 | 0:46 |
| Transformer, feature extraction | [2025-12-23](outputs/transformer-feature-extract/2025-12-23_15-50/eval/test_results.json) | 49.37% | 38.51% | 0.0489 | 1:15 |
| Transformer, fine-tuning | [2026-01-03](outputs/transformer-fine-tune/2026-01-03_20-47/eval/test_results.json) | 59.87% | 50.40% | 0.0364 | 0:32 |

Fine-tuning achieved the highest micro-F1. The November 2025 fine-tuning run achieved the highest macro-F1 (51.38%), while max-pooling outperformed last-token BiLSTM in the saved experiments. The reported training times are retained from the original comparison; hardware and timing procedure were not recorded, so they should be treated as rough historical figures rather than reproducible benchmarks. For the full comparison and experiment notes, see the [report](report.md#test-results).

---

## Issues and Future Work
The models still have room for improvement. Also the code could be more dinamic if loss functions and optimizers weren't hardcoded and could be changed in the configurations.
For further expansions we could try othe models and architectures or try to improve the already existing models (especially transformer fine-tuning), seeking a solution for its over-fitting problem.
Also a test unit is needed for because manually checking everything is becoming difficult.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project was created for learning purposes.
I’ve tried to write clear and well-documented code.
If you notice any issues or have suggestions, please let me know! 🌱