"""The choice model: scores each candidate of a page given the committed text and the letters typed, for the keystroke path.

Unlike `neural/model.py`'s character LM it has no vocabulary head. It reads `[context] [letters] [candidate] [END]` causally and turns the hidden state at END into one score per candidate (with `lm_term`, plus the candidate's own character log-likelihood through the tied table, one dot product per character); a page is ranked by those scores. The layout keeps the two savings the LM runtime relies on: the context is encoded once when text is committed and passed back in as per-layer keys and values, and each candidate sees only the context, the letters and its own earlier characters, so candidates never see one another and can be scored in one batch. See `docs/choice-model.md`.

Two entry points, each exported as its own ONNX graph:
- `encode_context(context)` -> past keys and values, run when text is committed.
- `score(past_k, past_v, letters, candidates, ends)` -> one score per candidate, run on every keystroke.
"""

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn

PAD, UNK, END = 0, 1, 2
RESERVED = ["<pad>", "<unk>", "<end>"]
# Letters as typed: a-z, the apostrophe syllable separator, and anything else as one shared slot.
LETTERS = "abcdefghijklmnopqrstuvwxyz'"
LETTER_OTHER = len(LETTERS) + 1  # 0 is padding
SEGMENTS = {"context": 0, "letters": 1, "candidate": 2}


@dataclasses.dataclass
class ChoiceConfig:
    vocab: int = 12288
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    max_positions: int = 256
    dropout: float = 0.0
    # Add each candidate's own character log-likelihood, read off the tied character table, to the END score. Training keeps the table's log normaliser near zero (`lm_loss(z=...)`), so the keystroke path sums one dot product per character instead of a softmax over the vocabulary.
    lm_term: bool = False


def letter_ids(text):
    return [LETTERS.index(c) + 1 if c in LETTERS else LETTER_OTHER for c in text.lower()]


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(approximate="tanh"), nn.Linear(4 * cfg.n_embd, cfg.n_embd))

    def heads(self, t):
        b, n, c = t.shape
        return t.view(b, n, self.n_head, c // self.n_head).transpose(1, 2)

    def forward(self, x, prefix_k, prefix_v):
        """x: [B, N, C] new tokens; prefix_k / prefix_v: [B, H, P, D] keys and values they may all see. Returns the new tokens' output and their own keys and values."""
        b, n, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(c, dim=-1)
        q, k, v = self.heads(q), self.heads(k), self.heads(v)
        keys = torch.cat([prefix_k, k], dim=2)
        values = torch.cat([prefix_v, v], dim=2)
        p = prefix_k.shape[2]
        # Every new token sees the whole prefix and the new tokens up to itself.
        rows = torch.arange(n, device=x.device).unsqueeze(1)
        cols = torch.arange(p + n, device=x.device).unsqueeze(0)
        mask = cols <= rows + p
        y = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
        x = x + self.proj(y.transpose(1, 2).reshape(b, n, c))
        x = x + self.mlp(self.ln2(x))
        return x, k, v


class ChoiceModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.chars = nn.Embedding(cfg.vocab, cfg.n_embd)
        self.letters = nn.Embedding(LETTER_OTHER + 1, cfg.n_embd)
        self.segments = nn.Embedding(len(SEGMENTS), cfg.n_embd)
        self.positions = nn.Embedding(cfg.max_positions, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, 1)
        self.apply(self._init)
        if cfg.lm_term:
            self.lm_scale = nn.Parameter(torch.ones(()))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def non_embedding_parameters(self):
        return sum(p.numel() for name, p in self.named_parameters() if not name.split(".")[0] in ("chars", "letters", "segments", "positions"))

    def embed(self, tokens, table, segment, start):
        n = tokens.shape[-1]
        pos = torch.arange(start, start + n, device=tokens.device)
        return table(tokens) + self.segments.weight[SEGMENTS[segment]] + self.positions(pos)

    def empty(self, b, device):
        h = self.cfg.n_head
        d = self.cfg.n_embd // h
        return torch.zeros(b, h, 0, d, device=device)

    def encode_context(self, context):
        """context: [L] character ids. Returns past_k, past_v: [layers, H, L, D]."""
        x = self.embed(context.unsqueeze(0), self.chars, "context", 0)
        ks, vs = [], []
        for block in self.blocks:
            x, k, v = block(x, self.empty(1, x.device), self.empty(1, x.device))
            ks.append(k[0])
            vs.append(v[0])
        return torch.stack(ks), torch.stack(vs)

    def score(self, past_k, past_v, letters, candidates, ends):
        """past_k / past_v: [layers, H, Lc, D] from encode_context; letters: [La] letter ids; candidates: [B, Lb] character ids, each followed by END and padded; ends: [B] index of each row's END. Returns [B] scores."""
        lc = past_k.shape[2]
        b = candidates.shape[0]
        # The letters, once, after the context.
        x = self.embed(letters.unsqueeze(0), self.letters, "letters", lc)
        prefix_k, prefix_v = [], []
        for i, block in enumerate(self.blocks):
            pk, pv = past_k[i].unsqueeze(0), past_v[i].unsqueeze(0)
            x, k, v = block(x, pk, pv)
            prefix_k.append(torch.cat([pk, k], dim=2))
            prefix_v.append(torch.cat([pv, v], dim=2))
        # Every candidate after context + letters, all in one batch.
        y = self.embed(candidates, self.chars, "candidate", lc + letters.shape[0])
        for i, block in enumerate(self.blocks):
            pk = prefix_k[i].expand(b, -1, -1, -1)
            pv = prefix_v[i].expand(b, -1, -1, -1)
            y, _, _ = block(y, pk, pv)
        hy = self.ln_f(y)
        out = self.head(hy[torch.arange(b, device=y.device), ends]).squeeze(-1)
        if self.cfg.lm_term:
            # The last letter predicts each candidate's first character, every candidate character the next one, up to END.
            pred = torch.cat([self.ln_f(x[:, -1:]).expand(b, -1, -1), hy[:, :-1]], dim=1)
            dots = (pred * self.chars(candidates)).sum(-1)
            mask = torch.arange(candidates.shape[1], device=y.device).unsqueeze(0) < ends.unsqueeze(1)
            out = out + self.lm_scale * (dots * mask).sum(-1)
        return out


    def rows(self, chars, letters, segments, ends, hidden=False):
        """Training form of `score`: one row per candidate holding its own copy of the context and letters, `[context][letters][candidate][END]`, right-padded. chars / letters / segments: [R, N]; a position takes its embedding from `letters` where segments is 1 and from `chars` elsewhere. Attention is causal, so each row computes exactly what `encode_context` + `score` compute for that candidate. Returns [R] scores, and with `hidden` the final hidden states [R, N, C] too."""
        r, n = chars.shape
        pos = torch.arange(n, device=chars.device)
        x = torch.where((segments == SEGMENTS["letters"]).unsqueeze(-1), self.letters(letters), self.chars(chars))
        x = x + self.segments(segments) + self.positions(pos)
        for block in self.blocks:
            x, _, _ = block(x, self.empty(r, x.device), self.empty(r, x.device))
        h = self.ln_f(x)
        out = self.head(h[torch.arange(r, device=x.device), ends]).squeeze(-1)
        if not self.cfg.lm_term:
            return (out, h) if hidden else out
        targets = chars[:, 1:]
        mask = (segments[:, 1:] == SEGMENTS["candidate"]) & (targets != END) & (targets != PAD)
        dots = (h[:, :-1] * self.chars(targets)).sum(-1)
        out = out + self.lm_scale * (dots * mask).sum(-1)
        return (out, h) if hidden else out

    def candidate_lm_loss(self, h, chars, segments, rows, z=0.0):
        """The gold candidates' characters as a conditioned LM: each predicted from the context, the letters and the characters before it, through the tied table. h: `rows(..., hidden=True)`'s hidden states; rows: [G] the gold rows."""
        targets = chars[rows, 1:]
        mask = (segments[rows, 1:] == SEGMENTS["candidate"]) & (targets != END) & (targets != PAD)
        return self.tied_loss(h[rows, :-1][mask], targets[mask], z)

    def tied_loss(self, h, y, z):
        """Cross-entropy through the tied character table, plus `z * logZ^2` so the unnormalised dot product approximates the log-probability."""
        logits = h @ self.chars.weight.t()
        lse = torch.logsumexp(logits.float(), dim=-1)
        ce = lse - logits.gather(-1, y.unsqueeze(-1)).squeeze(-1).float()
        return ce.mean() + z * (lse * lse).mean() if z else ce.mean()

    def lm_loss(self, x, y, z=0.0):
        """Pretraining as a character LM over context-segment text, through the character table (tied); without `lm_term` the table is discarded from the keystroke path, so this costs nothing at inference. `z` adds the self-normalisation penalty of `tied_loss`."""
        h = self.embed(x, self.chars, "context", 0)
        for block in self.blocks:
            h, _, _ = block(h, self.empty(x.shape[0], h.device), self.empty(x.shape[0], h.device))
        h = self.ln_f(h)
        if not z:
            logits = h @ self.chars.weight.t()
            return F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1), ignore_index=PAD)
        keep = y != PAD
        return self.tied_loss(h[keep], y[keep], z)


class ContextGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, context):
        return self.model.encode_context(context)


class KeystrokeGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, past_k, past_v, letters, candidates, ends):
        return self.model.score(past_k, past_v, letters, candidates, ends)
