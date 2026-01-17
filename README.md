# Emotion Classifier

A multilabel classification ML project to classify emotions in given English sentences
using PyTorch and HuggingFace Transformers.

- [Overview](#emotion-classifier)
- [Getting Started](#getting-started)
- [Components](#components)
- [Results](#results)
  - [dataset](#dataset)
  - [bilstm last-token](#bilstm-last-token)
  - [bilstm max-pool](#bilstm-max-pool)
  - [transformer feature-extract](#transformer-feature-extract)
  - [transformer fine-tune](#transformer-fine-tune)
  - [comparison](#comparison)
- [Issues and Further Expantion]()
- [License](#license)

---

## Getting Started
The project is formated as a python package, Once you clone it on your machine and open the root directory,
you can install it using pip in your virtual environment using the following command:
```bash
pip install -e .
```

Configure a model in a YAML file ([here](config.yaml) is an example of a config file)
and train your model using this command (pass your config file and output directory as arguments).
The trained model will be saved at a new run directory (as `best_model.pt`) next to training logs:
```bash
emotion-train --config <your-config-path> --output-dir <your-outputs-path>
```

Then you need to find optimal thresholds for your model, use this command, pass the run directory as model path,
the results (optimal thresholds) will be saved in `thresholds.json` at model's path:
```bash
emotion-tune-thresholds --model-path <your-run-path>
```

Then you can evaluate your model as follow and the results will be saved at a sub directory `eval/` at the model's path.
```bash
emotion-eval --model-path <your-run-path>
```

You can also test it yourself to see how good it can predict the feelings in your sentence like this:
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
- `uitls/` side utilities and helper functions are here.
- `tools/visualisation/` some tools to visualize the outputs. ***(WIP)***

---

## Results
These are the results I achived from my experiments:

[here](report.md) is the full report.

### dataset
I used GoEmotions dataset for this project, which is a multilabel dataset of sentences by Google.
It's an imbalanced dataset with an inherited cooccurance of labels.
[here](/outputs/dataset/note.md) is an analysis of this dataset.

### BiLSTM last-token
Well this was the first thing and easiest thing that came to my mind, I tried it and it wasn't bad actually,
it jsut took longer to train than i thought it would. These are the results that I got:
f1 micro | f1 macro | precision | Recall | hamming loss
:---:|:---:|:---:|:---:|:---:
~48|~39|~42|~57|0.049

[here](report.md#bilstm-last-token) is a full report on this model.

### BiLSTM max-pool
The next thing that came to my mind, was this (and honestly didn't think it work).
It was better both in training time and performance. These are the results that I got:
f1 micro | f1 macro | precision | Recall | hamming loss
:---:|:---:|:---:|:---:|:---:
~48|~44|~42|~57|0.042

[here](report.md#bilstm-max-pool) is a full report on this model.

### Transformer feature-extract
I added a two layer fully-connected neural-network to a pretrained Encoder (distil-BERT) as a feature extractor head.
I did a few experiments and here is the (best) results that I got:
f1 micro | f1 macro | precision | Recall | hamming loss
:---:|:---:|:---:|:---:|:---:
~47|~38|~43|~53|0.048

[here](report.md#transformer-feature-extract) is a full report on this model.

### Transformer fine-tune
f1 micro | f1 macro | precision | Recall | hamming loss
:---:|:---:|:---:|:---:|:---:
~59|~50|~55|~56|0.036

Fine-tuning is well known as a way to specialize a pretrained model for a task,
so I added a head to a bert model and fine-tuned it 

[here](report.md#transformer-fine-tune) is a full report on this model..

On my first attempt I Added a to layer the pretrained distil-BERT and
trained it while fine-tuning at the same time (with a relatively hight learning-rate)
and well it didn't go very well as the model quikely over-fitted and learning was too unstable
(early stopping triggered at epoch 6 which means only three epohs of actaul learning).
[here](report.md#run-2025-11-29_16-26) is some more info about this attempt.

I decreased threshold for evaluation during training to prevent early-stopping unless it's needed,
this technique worked very well with feature-extracted variant of this model but didn't work for this one.
In fact it over-fitted even quicker and stopped-early ar epoch 5.
[here](report.md#run-2025-12-23_18-12) is some more info about this attempt

Finally I changed every thing from model architecture to training loop to overcome this.
The problem, clearly was over-fitting, so I added a warm-up and a scheduler,
reduced head from two layers to one layer and changed the dropout rate (from 0.3 to 0.1).
with all these changes, the model trained much better althou it did became over-fitted so quickly again.
[here](report.md#run-2026-01-03_20-47) is a full report on last version of this model.


### Comparison
this tabel is a brief comparison of models. look [here](report.md#comparison) for a full comparison.
(*TL;DR*: fine-tuned transformer is clearly superior becuase of a shorter training time and higher metrics.)

model | training time | f1 score | precision | Recall
---|:---:|:---:|:---:|:---:
bilstm lasttoken|1:26|48|42|57
bilstm maxpool|0:46|54|49|61
transformer featureextract|1:15|47|43|53
transformer finetune|0:32|59|55|56


---

## Issues and Further Expansion
The most important issue is of course the low accuracy of models.
Also the code could be more dinamic if loss functions and optimizers weren't hardcoded and
could be changed in the configurations.
For further expansions we could try othe models and architectures or
try to train the already existing models (transformer-fine-tune especially) batter,
seeking a solution for its over-fitting problem.
Also a test unit is needed for all these stuff because the project is kinda large now
and testing every thing by hand is quite hard.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project was created for learning purposes.
I’ve tried to write clear and well-documented code.
If you notice any issues or have suggestions, please let me know! 🌱