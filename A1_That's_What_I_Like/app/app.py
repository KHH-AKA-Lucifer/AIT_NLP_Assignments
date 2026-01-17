from flask import Flask, request, render_template
import numpy as np
import pickle
import nltk
from nltk.corpus import reuters
import gensim.downloader as api

app = Flask(__name__)

# --- CONFIGURATION ---
MODELS = {} # Store all model data here
DOC_MATRICES = {} # Store pre-computed paragraph vectors here

# Ensure Data Exists
try:
    nltk.data.find('corpora/reuters.zip')
except LookupError:
    nltk.download('reuters')

# --- 1. LOAD MODELS ---
print("--- STARTUP: LOADING MODELS ---")

# A. Load Gensim (The Benchmark)
print("1. Loading Gensim...")
gensim_model = api.load("glove-wiki-gigaword-50")
MODELS['gensim'] = {'type': 'gensim', 'model': gensim_model}

# B. Load Custom Models (The Ones You Saved)
def load_custom_model(name, filename):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"2. Loading {name}...")
        MODELS[name] = {
            'type': 'custom',
            'vectors': data['vectors'],
            'word2index': data['word2index']
        }
    except FileNotFoundError:
        print(f"WARNING: {filename} not found. Skipping.")

load_custom_model('skipgram', '../artefacts/skipgram.pkl')
load_custom_model('skipgram_neg', '../artefacts/skipgramneg.pkl')
load_custom_model('glove', '../artefacts/glove.pkl')

# --- 2. PRE-COMPUTE PARAGRAPH VECTORS (FOR ALL MODELS) ---
print("--- INDEXING REUTERS CORPUS (This happens once) ---")
file_ids = reuters.fileids()
documents = []

# Get raw text once
for fid in file_ids:
    documents.append(reuters.raw(fid)) # Store full text

# Helper to vectorize a doc
def get_doc_vector(words, model_key):
    model_data = MODELS[model_key]
    vecs = []
    
    for w in words:
        w = w.lower()
        # Gensim Logic
        if model_data['type'] == 'gensim':
            if w in model_data['model']:
                vecs.append(model_data['model'][w])
        # Custom Model Logic
        else:
            if w in model_data['word2index']:
                idx = model_data['word2index'][w]
                vecs.append(model_data['vectors'][idx])
    
    if len(vecs) > 0:
        return np.mean(vecs, axis=0)
    return np.zeros(50) if model_data['type'] == 'gensim' else np.zeros(100)

# Build Matrices for each model
for model_name in MODELS.keys():
    print(f"Indexing for {model_name}...")
    vec_list = []
    for fid in file_ids:
        words = reuters.words(fid)
        vec_list.append(get_doc_vector(words, model_name))
    
    # Normalize Matrix
    matrix = np.array(vec_list)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    DOC_MATRICES[model_name] = matrix / norms

print("--- SYSTEM READY ---")

# --- 3. SEARCH FUNCTION ---
def search(query, model_name):
    if model_name not in MODELS:
        return ["Error: Model not selected"]
    
    # Vectorize Query
    query_words = query.lower().split()
    q_vec = get_doc_vector(query_words, model_name)
    
    # Cosine Sim
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    scores = np.dot(DOC_MATRICES[model_name], q_norm)
    
    # Top 5
    top_indices = np.argsort(scores)[::-1][:10]
    
    results = []
    for idx in top_indices:
        results.append({
            "score": float(scores[idx]),
            "text": documents[idx][:400] + "..." # Snippet
        })
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query = ""
    selected_model = "gensim"
    
    if request.method == 'POST':
        query = request.form.get('query', '')
        selected_model = request.form.get('model_selector', 'gensim')
        results = search(query, selected_model)
        
    return render_template('index.html', query=query, results=results, 
                           selected_model=selected_model, models=MODELS.keys())

if __name__ == '__main__':
    app.run(debug=True, port=5001)