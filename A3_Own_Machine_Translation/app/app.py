from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import json
import numpy as np
from pathlib import Path

# ---- IDs must match training ----
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

ARTEFACTS = Path("../artefacts")
CFG = json.loads((ARTEFACTS / "config.json").read_text(encoding="utf-8"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Attention modules ----
class GeneralAttention(nn.Module):
    def forward(self, dec_hidden, enc_outputs, enc_mask):
        scores = torch.bmm(enc_outputs, dec_hidden.unsqueeze(2)).squeeze(2)  # [B,S]
        scores = scores.masked_fill(enc_mask == 0, -1e9)
        attn = F.softmax(scores, dim=1)
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn

class AdditiveAttention(nn.Module):
    def __init__(self, hid_dim, attn_dim):
        super().__init__()
        self.W1 = nn.Linear(hid_dim, attn_dim, bias=False)
        self.W2 = nn.Linear(hid_dim, attn_dim, bias=False)
        self.v  = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, enc_mask):
        S = enc_outputs.size(1)
        dec_expand = dec_hidden.unsqueeze(1).expand(-1, S, -1)
        energy = torch.tanh(self.W1(enc_outputs) + self.W2(dec_expand))
        scores = self.v(energy).squeeze(2)
        scores = scores.masked_fill(enc_mask == 0, -1e9)
        attn = F.softmax(scores, dim=1)
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn

# ---- Encoder/Decoder ----
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc  = nn.Linear(hid_dim * 2, hid_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, src, src_len):
        x = self.drop(self.emb(src))
        packed = nn.utils.rnn.pack_padded_sequence(x, src_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        h = torch.tanh(self.fc(torch.cat([h[-2], h[-1]], dim=1)))
        c = torch.tanh(self.fc(torch.cat([c[-2], c[-1]], dim=1)))
        return out, (h.unsqueeze(0), c.unsqueeze(0))

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, attention, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.attention = attention
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim + hid_dim + emb_dim, vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_tok, hidden, enc_outputs_proj, enc_mask):
        emb = self.drop(self.emb(input_tok)).unsqueeze(1)
        dec_h = hidden[0].squeeze(0)
        context, attn = self.attention(dec_h, enc_outputs_proj, enc_mask)
        rnn_in = torch.cat([emb, context.unsqueeze(1)], dim=2)
        out, hidden = self.rnn(rnn_in, hidden)
        out = out.squeeze(1)
        pred = self.fc_out(torch.cat([out, context, emb.squeeze(1)], dim=1))
        return pred, hidden, attn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, enc_proj):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enc_proj = enc_proj

# ---- Load SentencePiece ----
sp = spm.SentencePieceProcessor()
sp.load(str(ARTEFACTS / "spm_my_en.model"))

# ---- Build model ----
V = CFG["vocab_size"]
EMB = CFG["emb_dim"]
HID = CFG["hid_dim"]

enc = Encoder(V, EMB, HID).to(device)
enc_proj = nn.Linear(HID*2, HID).to(device)

if CFG["best_name"] == "general":
    attn = GeneralAttention().to(device)
else:
    attn = AdditiveAttention(HID, attn_dim=64).to(device)

dec = Decoder(V, EMB, HID, attn).to(device)
model = Seq2Seq(enc, dec, enc_proj).to(device)

state = torch.load(str(ARTEFACTS / "seq2seq_additive.pt"), map_location=device)
model.load_state_dict(state)
model.eval()

@torch.no_grad()
def translate(src_sentence, max_len=CFG["max_decode_len"]):
    src_ids = [BOS_ID] + sp.encode(src_sentence, out_type=int) + [EOS_ID]
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_len = torch.tensor([len(src_ids)], dtype=torch.long).to(device)

    enc_out, hidden = model.encoder(src, src_len)
    enc_out = model.enc_proj(enc_out)
    enc_mask = (src != PAD_ID).long()

    tok = torch.tensor([BOS_ID], dtype=torch.long).to(device)
    gen = []

    for _ in range(max_len):
        pred, hidden, _ = model.decoder(tok, hidden, enc_out, enc_mask)
        nxt = int(pred.argmax(dim=1).item())
        if nxt == EOS_ID:
            break
        gen.append(nxt)
        tok = torch.tensor([nxt], dtype=torch.long).to(device)

    return sp.decode(gen)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    src_text = ""
    out_text = ""
    if request.method == "POST":
        src_text = request.form.get("src", "").strip()
        if src_text:
            out_text = translate(src_text)
    return render_template("index.html", src_text=src_text, out_text=out_text)

if __name__ == "__main__":
    app.run(debug=True)
