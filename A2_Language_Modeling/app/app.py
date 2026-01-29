import json
import torch
import torch.nn as nn
from pathlib import Path 
import torch.nn.functional as F 
from torchtext.data.utils import get_tokenizer
from flask import Flask, render_template, request


# configuration
base_dir = Path(__file__).resolve().parent.parent
print("base directory", base_dir)
artefact_dir = base_dir / 'artefacts'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMB_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2

# -----------------------------
# Model definition (MUST match notebook)
# -----------------------------
class LSTMLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256, num_layers=2, dropout=0.2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

# -----------------------------
# Load vocab
# -----------------------------
vocab_path = artefact_dir / 'vocab.json'
with vocab_path.open('r', encoding='utf-8') as f:
    vocab_data = json.load(f)

itos = vocab_data['itos']            # index -> token
stoi = vocab_data['stoi']            # token -> index

UNK_ID = stoi.get('<unk>', stoi.get('<unk>', 1))  # fallback
PAD_ID = stoi.get('<pad>', stoi.get('<pad>', 0))

VOCAB_SIZE = len(itos)

tokenizer = get_tokenizer('basic_english')

def basic_tokenize(text: str):
    '''
    Keep tokenization consistent with training.
    If you trained with torchtext basic_english, this is a close approximation.
    '''
    # Simple fallback tokenizer: lowercase + split on whitespace
    # If your notebook used torchtext basic_english, consider copying that logic here later.
    return tokenizer(text)

def encode_tokens(tokens):
    return [stoi.get(t, UNK_ID) for t in tokens]

def decode_ids(ids):
    return ' '.join(itos[i] for i in ids)

# -----------------------------
# Load model
# -----------------------------
model = LSTMLM(
    vocab_size=VOCAB_SIZE,
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    pad_idx=PAD_ID
).to(device)

state_path = artefact_dir / 'starwars_lstm_model.pt'
model.load_state_dict(torch.load(state_path, map_location=device))
model.eval()

# -----------------------------
# Generation
# -----------------------------
@torch.no_grad()
def generate_text(prompt: str, max_new_tokens: int = 50, temperature: float = 1.0):
    prompt = prompt.strip()
    if not prompt:
        return ''

    tokens = basic_tokenize(prompt)
    ids = encode_tokens(tokens)

    # start with full prompt as input, then feed one token at a time
    x = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = None

    # Warm-up pass: run prompt through model to set hidden state
    logits, hidden = model(x, hidden)

    for _ in range(max_new_tokens):
        next_logits = logits[0, -1, :] / max(float(temperature), 1e-8)
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        ids.append(next_id)

        x = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, hidden = model(x, hidden)

    return decode_ids(ids)

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    generated = ''
    prompt = ''
    max_new_tokens = 60
    temperature = 0.9

    if request.method == 'POST':
        prompt = request.form.get('prompt', '')
        max_new_tokens = int(request.form.get('max_new_tokens', 60))
        temperature = float(request.form.get('temperature', 0.9))

        generated = generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

    return render_template(
        'index.html',
        prompt=prompt,
        generated=generated,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

if __name__ == '__main__':
    # For local dev
    app.run(host='0.0.0.0', port=5001, debug=True)