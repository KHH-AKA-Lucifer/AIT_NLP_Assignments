from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import numpy as np
from pathlib import Path

# ---- IDs must match training ----
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# ---- Paths (robust) ----
BASE_DIR = Path(__file__).resolve().parent
ARTEFACTS = (BASE_DIR / ".." / "artefacts").resolve()

# Choose your deployed checkpoint here
CKPT_PATH = ARTEFACTS / "seq2seq_additive.pt"   # or "seq2seq_general.pt"
SPM_PATH  = ARTEFACTS / "spm_my_en.model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Helpers: infer hyperparams from state_dict
# -------------------------
def infer_cfg_from_state(state: dict) -> dict:
    """
    Infers vocab_size, emb_dim, hid_dim from the saved weights.
    Works when you only saved state_dict (no config.json).
    """
    # encoder embedding: [V, EMB]
    V, EMB = state["encoder.emb.weight"].shape

    # LSTM weight_ih_l0: [4*HID, EMB]
    hid4, emb2 = state["encoder.rnn.weight_ih_l0"].shape
    if int(emb2) != int(EMB):
        raise ValueError(f"Embedding dim mismatch: emb={EMB}, rnn expects={emb2}")

    if hid4 % 4 != 0:
        raise ValueError("Unexpected LSTM weight shape; cannot infer hid_dim cleanly.")
    HID = hid4 // 4

    return {
        "vocab_size": int(V),
        "emb_dim": int(EMB),
        "hid_dim": int(HID),
        "max_decode_len": 60,  # sensible default; you can change
    }


def infer_attn_type_from_ckpt_path(path: Path) -> str:
    """
    We can't reliably infer attention type from weights alone.
    So pick based on filename convention.
    """
    name = path.name.lower()
    if "general" in name:
        return "general"
    if "additive" in name:
        return "additive"
    # fallback: assume additive (common choice)
    return "additive"


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
        self.v = nn.Linear(attn_dim, 1, bias=False)

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
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, src, src_len):
        x = self.drop(self.emb(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h, c) = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Combine final forward/backward states for biLSTM
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
        emb = self.drop(self.emb(input_tok)).unsqueeze(1)  # [B,1,EMB]
        dec_h = hidden[0].squeeze(0)  # [B,HID]

        context, attn = self.attention(dec_h, enc_outputs_proj, enc_mask)  # [B,HID], [B,S]

        rnn_in = torch.cat([emb, context.unsqueeze(1)], dim=2)  # [B,1,EMB+HID]
        out, hidden = self.rnn(rnn_in, hidden)                  # out: [B,1,HID]
        out = out.squeeze(1)                                    # [B,HID]

        pred = self.fc_out(torch.cat([out, context, emb.squeeze(1)], dim=1))  # [B,V]
        return pred, hidden, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, enc_proj):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enc_proj = enc_proj


# ---- Load SentencePiece ----
sp = spm.SentencePieceProcessor()
if not SPM_PATH.exists():
    raise FileNotFoundError(f"SentencePiece model not found: {SPM_PATH}")
sp.load(str(SPM_PATH))


# ---- Load state dict and infer config ----
if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

state = torch.load(str(CKPT_PATH), map_location=device)
# If you ever saved a dict like {"state_dict":..., "cfg":...}, handle it too:
if isinstance(state, dict) and "state_dict" in state:
    # advanced checkpoint format
    state_dict = state["state_dict"]
    CFG = state.get("cfg", infer_cfg_from_state(state_dict))
else:
    # plain state_dict format
    state_dict = state
    CFG = infer_cfg_from_state(state_dict)

V = CFG["vocab_size"]
EMB = CFG["emb_dim"]
HID = CFG["hid_dim"]
MAX_DECODE_LEN = int(CFG.get("max_decode_len", 60))

# ---- Build model ----
enc = Encoder(V, EMB, HID).to(device)
enc_proj = nn.Linear(HID * 2, HID).to(device)

attn_type = CFG.get("attn_type") or infer_attn_type_from_ckpt_path(CKPT_PATH)
if attn_type == "general":
    attn = GeneralAttention().to(device)
else:
    attn = AdditiveAttention(HID, attn_dim=64).to(device)

dec = Decoder(V, EMB, HID, attn).to(device)
model = Seq2Seq(enc, dec, enc_proj).to(device)

model.load_state_dict(state_dict)
model.eval()


@torch.no_grad()
def translate(src_sentence: str, max_len: int = MAX_DECODE_LEN) -> str:
    # Encode source
    src_ids = [BOS_ID] + sp.encode(src_sentence, out_type=int) + [EOS_ID]
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)  # [1,S]
    src_len = torch.tensor([len(src_ids)], dtype=torch.long).to(device)

    # Encode
    enc_out, hidden = model.encoder(src, src_len)     # enc_out: [1,S,2H]
    enc_out = model.enc_proj(enc_out)                 # [1,S,H]
    enc_mask = (src != PAD_ID).long()                 # [1,S]

    # Decode
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


# ---- Flask app ----
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
    # debug=True for dev only
    app.run(host="0.0.0.0", port=5001, debug=True)
