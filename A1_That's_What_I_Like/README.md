# A1 That's What I Like

### 1. Introduction
The purpose of this assignment is to gain hands-on experience with fundamental word embedding techniques by implementing Skip-gram, Skip-gram with Negative Sampling, and GloVe models from scratch.


    1. We utilized the Reuters dataset from the NLTK library. The preprocessing pipeline involved:

    2. Downloading the corpus and removing stopwords/punctuation to extract unique words.

    3. Tokenizing and numericalizing the text to structure the vocabulary.

    4. Generating context-target pairs based on a defined window size.

### 2. Implementation Details
We implemented and trained three distinct models to compare their efficiency and performance:

Standard Skip-gram: Implemented using a full Softmax calculation.

Skip-gram with Negative Sampling (NEG): Implemented to improve computational efficiency by approximating the normalization factor.

GloVe (Global Vectors): Implemented to capture global co-occurrence statistics.

Hyperparameters:

Window Size: 2 (Context window)

Embedding Size: 100 dimensions

Epochs: 2000 for Skip-gram variants; 100 for GloVe

Batch Size: 256 for Skip-gram; 512 for GloVe

### 3. Model Comparison & Analysis
To evaluate our models, we compared them against a pre-trained Gensim GloVe model (trained on Wikipedia + Gigaword).

A. Training Loss & Time
Standard Skip-gram: Showed the highest training loss (~20.3) due to the difficulty of calculating Softmax over the entire vocabulary.

Negative Sampling: Significantly reduced the loss (~2.4) and training time, demonstrating the efficiency of the NEG approach.

GloVe: Achieved a very low regression loss (~0.004), indicating it successfully fitted the log-co-occurrence matrix.

B. Syntactic & Semantic Accuracy
We evaluated the models using the standard analogy test (e.g., King - Man + Woman = Queen).

Observation: Our custom models achieved nearly 0% accuracy on Semantic (Capital-Country) tasks and very low accuracy on Syntactic tasks.

Reasoning: This performance gap is due to the domain limitation of the Reuters dataset (Financial News), which lacks the general world knowledge (e.g., Capitals) found in the massive Wikipedia corpus used by Gensim. Additionally, a window size of 2 captures local syntax better than long-range semantic relationships.

C. Similarity Analysis (Spearman Correlation)
We assessed if our embeddings correlate with human judgment using the WordSim353 dataset.

Our models showed a positive correlation, confirming they learned meaningful relationships within the financial domain.

However, the Gensim model outperformed custom models significantly because human judgment relies on general knowledge, not just financial context.

### 4. Web Application (Search Engine)
Finally, we developed a semantic search engine using Flask and HTML/CSS.

Key Features:

Paragraph Vectors: The system pre-computes vector representations for thousands of Reuters documents by averaging the word vectors in each paragraph.

Multi-Model Selector: Users can select between our custom models (Skip-gram, GloVe) and the pre-trained Gensim model via a dropdown menu to compare search results dynamically.

Cosine Similarity: When a user enters a query (e.g., "market crisis"), the system converts the query into a vector and finds the top 5 most similar documents in the corpus using cosine similarity.

### File Structure 

```
search_app/
├── app.py                  # The main Flask backend script
├── artefacts/              # Folder for your saved model files
│   ├── skipgram.pkl
│   ├── skipgramneg.pkl
│   └── glove.pkl
└── templates/              # Folder for HTML files (Required by Flask)
    └── index.html          # The search engine frontend interface
```

In order to sync with the environment set up please run `pip install -m requirements.txt` first. Then go to app folder and start the web app with `python app.py`.