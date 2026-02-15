# --------------------------------------------------
# Task 4 - Simple Web Application for NLI Prediction
# --------------------------------------------------

from flask import Flask, render_template, request
import torch
import torch.nn as nn
import re

app = Flask(__name__)

device = torch.device("cpu")  # keep CPU for deployment simplicity

# --------------------------------------------------
# Load Artefacts
# --------------------------------------------------

ckpt = torch.load("../artefacts/bert_mlm.pt", map_location=device)
config = ckpt["config"]
word2id = ckpt["word2id"]

PAD_ID = word2id["[PAD]"]
UNK_ID = word2id["[UNK]"]
MAX_LEN = config["max_len"]
H = config["d_model"]

# Load encoder
encoder = torch.jit.load("../artefacts/bert_encoder.pt", map_location=device)
encoder.eval()

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def clean_text(s: str):
    s = s.lower()
    s = re.sub(r"[.,!\-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def encode_sentence(sentence):
    sentence = clean_text(sentence)
    tokens = sentence.split()

    ids = [word2id.get(w, UNK_ID) for w in tokens][:MAX_LEN]
    attn = [1] * len(ids)

    while len(ids) < MAX_LEN:
        ids.append(PAD_ID)
        attn.append(0)

    return torch.tensor([ids]), torch.tensor([attn])

def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (token_embeddings * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return summed / count

# --------------------------------------------------
# SBERT Classifier
# --------------------------------------------------

class SBERTSoftmax(nn.Module):
    def __init__(self, encoder, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size * 3, 3)

    def encode(self, input_ids, attention_mask):
        segment_ids = torch.zeros_like(input_ids)
        hidden = self.encoder(input_ids, segment_ids)
        return mean_pooling(hidden, attention_mask)

    def forward(self, prem_ids, prem_attn, hyp_ids, hyp_attn):
        u = self.encode(prem_ids, prem_attn)
        v = self.encode(hyp_ids, hyp_attn)
        feats = torch.cat([u, v, torch.abs(u - v)], dim=1)
        return self.classifier(feats)

# Load fine-tuned weights
model = SBERTSoftmax(encoder, H)
sbert_ckpt = torch.load("../artefacts/sbert_softmax_snli.pt", map_location=device)
model.load_state_dict(sbert_ckpt["sbert_state_dict"])
model.eval()

label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

# --------------------------------------------------
# Web Routes
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        premise = request.form["premise"]
        hypothesis = request.form["hypothesis"]

        prem_ids, prem_attn = encode_sentence(premise)
        hyp_ids, hyp_attn = encode_sentence(hypothesis)

        with torch.no_grad():
            logits = model(prem_ids, prem_attn, hyp_ids, hyp_attn)
            pred = torch.argmax(logits, dim=1).item()

        prediction = label_map[pred]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
