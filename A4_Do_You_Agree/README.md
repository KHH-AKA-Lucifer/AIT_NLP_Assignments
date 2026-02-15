#  Assignment 4: "Do You Agree?"

This project implements a complete Natural Language Inference (NLI)
pipeline, starting from building BERT from scratch to deploying a simple
web application that predicts whether two sentences entail, contradict,
or are neutral with respect to each other. The work is structured into
four main tasks that progressively build on each other.

The objective of this assignment was not only to use pretrained models,
but to understand how transformer-based architectures work internally
and how sentence-level representations can be adapted for downstream
tasks.

------------------------------------------------------------------------

## Task 1 -- Implementing BERT from Scratch

In Task 1, a simplified version of BERT was implemented from scratch
using PyTorch. The model includes token embeddings, positional
embeddings, segment embeddings, multi-head self-attention, feed-forward
layers, layer normalization, and residual connections.

The model was trained using Masked Language Modeling (MLM) and Next
Sentence Prediction (NSP). The dataset was sourced from a publicly
available corpus (BookCorpus subset). Text preprocessing involved
lowercasing, punctuation removal, whitespace tokenization, and
vocabulary construction with special tokens such as \[PAD\], \[CLS\],
\[SEP\], \[MASK\], and \[UNK\].

After training, the model weights were saved in `artefacts/bert_mlm.pt`.
The encoder portion was exported as `artefacts/bert_encoder.pt` for
reuse.

------------------------------------------------------------------------

## Task 2 -- Sentence-BERT (SBERT) Fine-Tuning

Task 2 adapted the pretrained BERT encoder into a Sentence-BERT
architecture. Mean pooling was applied to generate fixed-size sentence
embeddings.

For a pair of sentences (premise and hypothesis), embeddings were
computed independently in a Siamese structure. The feature vector was
constructed using u, v, and \|u − v\| and passed into a linear
classifier trained with softmax cross-entropy loss, following the
Sentence-BERT paper (Reimers & Gurevych, 2019).

The SNLI dataset was used for fine-tuning. The fine-tuned model was
saved as `artefacts/sbert_softmax_snli.pt`.

------------------------------------------------------------------------

## Task 3 -- Evaluation

The fine-tuned SBERT model was evaluated on validation and test splits
of the SNLI dataset using accuracy, precision, recall, F1-score, and
confusion matrix.

The model achieved approximately 48% accuracy, which is above the random
baseline (\~33%). Stronger performance was observed for contradiction
detection, while entailment recall remained lower. The moderate
performance is largely due to vocabulary mismatch between the
pretraining corpus and SNLI.

------------------------------------------------------------------------

## Task 4 -- Web Application Deployment

A Flask-based web application was developed to deploy the trained SBERT
model. Users input a premise and hypothesis, and the system predicts
Entailment, Neutral, or Contradiction.

This completes the full pipeline from model implementation to real-world
deployment.

------------------------------------------------------------------------

## Project Structure

A4_Do_you_AGREE/ task1.ipynb task2.ipynb task3.ipynb artefacts/
bert_mlm.pt bert_encoder.pt sbert_softmax_snli.pt app/ main.py
templates/ index.html

------------------------------------------------------------------------

## How to Run

1.  Install dependencies from requirements.txt.
2.  Navigate to the app directory.
3.  Run: python main.py
4.  Open http://127.0.0.1:5001 in the browser of choice.

------------------------------------------------------------------------

## Conclusion

This project demonstrates the complete lifecycle of a transformer-based
NLP system, from implementing BERT from scratch to fine-tuning for
sentence-level inference and deploying a working web application.

## Output
