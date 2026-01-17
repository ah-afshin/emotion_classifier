# **Report**
This is a full report on experiments with Emotion Classifier models, results, diagnostics and comparisions and dataset analysis.

**note**: in all of these experiments, evaluation and analysis is done on best model, not last model.

- [dataset](#dataset)
- [BiLSTM last-token](#bilstm-last-token)
    - [1st run](#run-2025-11-29_13-08)
- [BiLSTM max-pool](#bilstm-max-pool)
    - [1st run](#run-2025-11-28_23-30)
- [Transformer feature-extract](#transformer-feature-extract)
    - [1st run](#run-2025-11-29_15-22)
    - [2nd run](#run-2025-12-23_15-50)
    - [3rd run](#run-2026-01-04_22-06)
- [Transformer fine-tune](#transformer-fine-tune)
    - [1st run](#run-2025-11-29_16-26)
    - [2nd run](#run-2025-12-23_18-12)
    - [3rd run](#run-2026-01-03_20-47)
- [Comparison](#comparison)

# Dataset
I used GoEmotions dataset for this project, which is a multilabel dataset of sentences (gathered from reddit comments)
by Google. Each sentence either has one or more emotions or is labeled as *neutral*.
There are 27 emotions (plus neutral), and each sentence is labeled with these 28 classes.
This dataset is splited to training, evaluation and test sets by authors (proporsions are 80/10/10).
<!-- You can find more about it on its [official website](link). -->

GoEmotion is an imbalanced dataset with an inherited cooccurance of labels.
Simply some labels occure more and some less, and some occure together more often.
You can see that inherited structure on the charts below:

![number of label samples in train set](/outputs/dataset/train/plots/num_label_samples.png)
![number of label samples in eval set](/outputs/dataset/eval/plots/num_label_samples.png)
![number of label samples in test set](/outputs/dataset/test/plots/num_label_samples.png)


![label cooccurance heatmap - train set](/outputs/dataset/train/plots/cooccurance_heatmap.png)
![label cooccurance heatmap - eval set](/outputs/dataset/eval/plots/cooccurance_heatmap.png)
![label cooccurance heatmap - test set](/outputs/dataset/test/plots/cooccurance_heatmap.png)

---

# BiLSTM last-token
Bi-Directional Long Short-Term Memory Models are one of the models used for NLP,
they process sentences like time series in two directions.
Here I used a *bert-base-uncased* tokenizer to break sentences into tokens,
the model embeds each token into a 128-dimentional vactor and processes these vectors as a time serie
through a three layer LSMT model (each with 256 hidden units and 0.25 dropout rate)
and chooses the last token to percive the meaning in the sentence,
it passes that tokens vector to a Linear layer that extracts the emotions and classifies the sentence.
[This](/src/emotion_classifier/models/bilstm.py) is the code if you wanted to check it out.

## run 2025-11-29_13-08
trained with 5e-4 learning-rate and adam optimizer for 15 epochs (wasn't stopped earlier) and achived the expected metrics.
[Here](/outputs/bilstm-last-token/2025-11-29_13-08/train.log) is the training log.

### Training
### Evaluation Result
### Observed Beheivior
### Analysis and Diagnostics

---

# BiLSTM max-pool
Bi-Directional Long Short-Term Memory Models are one of the models used for NLP,
they process sentences like time series in two directions.
Here I used a *bert-base-uncased* tokenizer to break sentences into tokens,
the model embeds each token into a 128-dimentional vactor and processes these vectors as a time serie
through a three layer LSMT model (each with 256 hidden units and 0.25 dropout rate)
and chooses the token with maximum value to percive the meaning in the sentence,
it passes that tokens vector to a Linear layer that extracts the emotions and classifies the sentence.
[This](/src/emotion_classifier/models/bilstm.py) is the code if you wanted to check it out.

## run 2025-11-28_23-30
trained with 5e-4 learning-rate and adam optimizer for 11 epochs (stopped earlier, best model saved at epoch 8)
and surprisingly achived better metrics the than last-token variant.
[Here](/outputs/bilstm-max-pool/2025-11-28_23-30/train.log) is the training log.

### Training
### Evaluation Result
### Observed Beheivior
### Analysis and Diagnostics

---

# Transformer feature-extract
In my first try with transformers i added a two layer fully-connected neural-network to a pretrained Encoder (distil-BERT)
as a feature extractor that uses BERT's output to find out the emotions in the sentence.

## run 2025-11-29_15-22
trained with 5e-4 learning-rate and adam optimizer for 7 epochs (stopped earlier, best model saved at epoch 4)
well it didn't work as well as i thought and was stopped too early, probably beacause of a bad choice of threshold
that effected evaluation and triggered early stopping. 

### Training
### Evaluation Result
### Observed Beheivior
### Known Issues

## run 2025-12-23_15-50
I changed the configurations and tried again, which turned out to be the best performance of this model.
I lowered the threshold for evaluation during training to prevent early-stopping unless it's needed
(the threshold for evaluation during training was 0.6 before that wasn't a good idea really, decreased to 0.3).

### Training
### Evaluation Result
### Observed Beheivior
### Analysis and Diagnostics

## run 2026-01-04_22-06
later when I changed learning process and model architecture for fine-tuned model
(reduced the head  to one layer and lower dropout rate),
I trained this model for last time as well but it became worse (probably because the head wasn't deep enough).

### Training
### Evaluation Result
### Observed Beheivior
### Analysis and Diagnostics
### Known Issues

---


# Transformer fine-tune
Fine-tuning is well known as a way to specialize a pretrained model for a task,
so I Added a two layer head to pretrained distil-BERT encoder and
trained it while fine-tuning BERT model at the same time (with a relatively lower learning-rate)

## run 2025-11-29_16-26
and well it didn't go very well as the model quikely over-fitted and learning was too unstable
(early stopping triggered at epoch 6 which means only three epohs of actaul learning).

### Training
### Evaluation Result
### Observed Beheivior
### Known Issues

## run 2025-12-23_18-12
I decreased threshold for evaluation during training to prevent early-stopping unless it's needed,
this technique worked very well with feature-extracted variant of this model but didn't work for this one.
In fact it over-fitted even quicker and stopped-early ar epoch 5.

### Training
### Evaluation Result
### Observed Beheivior
### Known Issues

## run 2026-01-03_20-47
Finally I changed every thing from model architecture to training loop to overcome this.
The problem, clearly was over-fitting, so I added a warm-up and a scheduler,
reduced head from two layers to one layer and changed the dropout rate (from 0.3 to 0.1).
with all these changes, the model trained much better althou it did became over-fitted so quickly again.

### Training
### Evaluation Result
### Observed Beheivior
### Analysis and Diagnostics

---

# Comparison
*TL;DR*: fine-tuned transformer is clearly superior becuase of a shorter training time and higher metrics.

model | training time | f1 score | precision | Recall
---|:---:|:---:|:---:|:---:
bilstm lasttoken|1:26|48|42|57
bilstm maxpool|0:46|54|49|61
transformer featureextract|1:15|47|43|53
transformer finetune|0:32|59|55|56
