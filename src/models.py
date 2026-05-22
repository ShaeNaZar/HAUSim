"""
Model definitions and training utilities for HSTU / ARGUS.

Both architectures share:
  - ContextEncoder with a shared token embedding table
  - The same event embedding: word_emb + context_emb (mean-pool) + action_emb + delta_emb
  - SequentialModel base class, same batch format, same save/load

Loss (both archs):
  - BCE on action (binary classification)
  - MSE of sigmoid(logit) vs action_prob where not NaN (soft recall regression)
  ARGUS additionally uses an in-batch NIP contrastive loss.

Public API
----------
model = build_model(ModelConfig(arch='argus'), vocab)

save_model(model, 'model.pt')
model = load_model('model.pt')

# Convenience namespace kept for backward compat
hstu.save(model, path)
model = hstu.load(path)

predictor = HSTUPredictor(model)
p  = predictor.predict_action(history, context, word)
ps = predictor.predict_action_batch(history, [(word, context), ...])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score

from .sequence_dataset import encode_tokens, log_bucketize_delta


# ── Configs ───────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Unified config for HSTU / ARGUS. Set `arch` to switch."""
    arch: str = 'hstu'          # 'hstu' | 'argus'
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    max_seq_len: int = 128
    max_context_tokens: int = 16
    n_time_buckets: int = 32
    n_pos_buckets: int = 32
    n_action_buckets: int = 2   # binary: 0=wrong, 1=correct
    dropout: float = 0.1
    argus_nip_weight: float = 0.5

    @property
    def d_head(self):
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 3
    grad_clip: float = 1.0


# ── Shared: relative attention bias ──────────────────────────────────────────

class RelativeAttentionBias(nn.Module):
    def __init__(self, n_heads: int, n_pos_buckets: int):
        super().__init__()
        self.n = n_pos_buckets
        self.emb = nn.Embedding(n_pos_buckets, n_heads)

    def forward(self, L: int, device):
        i = torch.arange(L, device=device)[:, None]
        j = torch.arange(L, device=device)[None, :]
        rel = (i - j).clamp(min=0).float()
        b = (torch.log1p(rel) / math.log1p(self.n * 4) * (self.n - 1)).long().clamp(0, self.n - 1)
        return self.emb(b).permute(2, 0, 1)  # (h, L, L)


# ── HSTU block ────────────────────────────────────────────────────────────────

class HSTUBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h, dh, dm = cfg.n_heads, cfg.d_head, cfg.d_model
        self.h, self.dh = h, dh
        self.f1 = nn.Linear(dm, 4 * h * dh)
        self.f2 = nn.Linear(h * dh, dm)
        self.norm_pre  = nn.LayerNorm(dm)
        self.norm_post = nn.LayerNorm(h * dh)
        self.drop = nn.Dropout(cfg.dropout)
        self.rab  = RelativeAttentionBias(h, cfg.n_pos_buckets)

    def forward(self, x, time_bias=None, attn_mask=None):
        B, L, _ = x.shape
        uvqk = F.silu(self.f1(self.norm_pre(x)))
        u, v, q, k = uvqk.split(self.h * self.dh, dim=-1)

        def heads(t):
            return t.view(B, L, self.h, self.dh).transpose(1, 2)

        scores = heads(q) @ heads(k).transpose(-1, -2) + self.rab(L, x.device).unsqueeze(0)
        if time_bias is not None:
            scores = scores + time_bias
        causal = torch.ones(L, L, device=x.device, dtype=torch.bool).triu(1)
        a = F.silu(scores) / L
        a = a.masked_fill(causal[None, None], 0.0)
        if attn_mask is not None:
            a = a.masked_fill(attn_mask[:, None, None], 0.0)
        av = (self.drop(a) @ heads(v)).transpose(1, 2).contiguous().view(B, L, -1)
        return x + self.drop(self.f2(self.norm_post(av) * u))


# ── ARGUS block ───────────────────────────────────────────────────────────────

class ARGUSBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        dm = cfg.d_model
        self.attn = nn.MultiheadAttention(dm, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(dm, 4 * dm), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(4 * dm, dm)
        )
        self.n1   = nn.LayerNorm(dm)
        self.n2   = nn.LayerNorm(dm)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, key_padding_mask=None):
        L = x.size(1)
        causal = torch.ones(L, L, device=x.device, dtype=torch.bool).triu(1)
        n = self.n1(x)
        attn_out, _ = self.attn(n, n, n, attn_mask=causal,
                                key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(attn_out)
        return x + self.drop(self.ff(self.n2(x)))


# ── Shared context encoder ────────────────────────────────────────────────────

class ContextEncoder(nn.Module):
    def __init__(self, n_tokens: int, d_model: int):
        super().__init__()
        self.token_emb = nn.Embedding(n_tokens, d_model, padding_idx=0)
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))

    def encode_context(self, ctx):
        emb  = self.token_emb(ctx)
        mask = (ctx != 0).float().unsqueeze(-1)
        return self.proj((emb * mask).sum(-2) / mask.sum(-2).clamp(min=1.0))

    def encode_word(self, w):
        return self.token_emb(w)


# ── Base sequential model ─────────────────────────────────────────────────────

class SequentialModel(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab: dict):
        super().__init__()
        self.cfg, self.vocab = cfg, vocab
        self.ctx_enc     = ContextEncoder(vocab['n_tokens'], cfg.d_model)
        self.action_emb  = nn.Embedding(cfg.n_action_buckets, cfg.d_model)
        self.delta_emb   = nn.Embedding(cfg.n_time_buckets, cfg.d_model)
        self.time_bias_e = nn.Embedding(cfg.n_time_buckets, cfg.n_heads)
        self.final_norm  = nn.LayerNorm(cfg.d_model)

    def _embed(self, hw, hc, ha, hd):
        x = (self.ctx_enc.encode_word(hw) + self.ctx_enc.encode_context(hc)
             + self.action_emb(ha) + self.delta_emb(hd))
        pad = (hw == 0)
        tb  = self.time_bias_e(hd).permute(0, 2, 1).unsqueeze(2)
        return x, pad, tb

    def encode_history(self, hw, hc, ha, hd):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError


# ── HSTU model ────────────────────────────────────────────────────────────────

class HSTUModel(SequentialModel):
    def __init__(self, cfg: ModelConfig, vocab: dict):
        super().__init__(cfg, vocab)
        self.blocks = nn.ModuleList([HSTUBlock(cfg) for _ in range(cfg.n_layers)])
        self.head = nn.Sequential(
            nn.Linear(4 * cfg.d_model, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, 1)
        )

    def encode_history(self, hw, hc, ha, hd):
        x, pad, tb = self._embed(hw, hc, ha, hd)
        for blk in self.blocks:
            x = blk(x, time_bias=tb, attn_mask=pad)
        return self.final_norm(x)

    def forward(self, batch):
        last = self.encode_history(batch['hist_word'], batch['hist_context'],
                                    batch['hist_action'], batch['hist_delta'])[:, -1]
        tw = self.ctx_enc.encode_word(batch['target_word'])
        tc = self.ctx_enc.encode_context(batch['target_context'])
        td = self.delta_emb(batch['target_delta'])
        return self.head(torch.cat([last, tw, tc, td], -1)).squeeze(-1)


# ── ARGUS item tower ──────────────────────────────────────────────────────────

class ItemTower(nn.Module):
    """Lightweight item encoder with independent weights from the user tower."""

    def __init__(self, n_tokens: int, d_model: int):
        super().__init__()
        self.word_emb = nn.Embedding(n_tokens, d_model, padding_idx=0)
        self.ctx_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.out_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def encode_context(self, ctx):
        emb  = self.word_emb(ctx)
        mask = (ctx != 0).float().unsqueeze(-1)
        pool = (emb * mask).sum(-2) / mask.sum(-2).clamp(min=1.0)
        return self.ctx_proj(pool)

    def forward(self, word, context):
        w = self.word_emb(word)
        c = self.encode_context(context)
        return self.norm(self.out_proj(torch.cat([w, c], dim=-1)))


# ── ARGUS model ───────────────────────────────────────────────────────────────

class ARGUSModel(SequentialModel):
    """
    Two-tower ARGUS with late binding.

    User tower  — causal transformer over history  → h_user
    Item tower  — lightweight MLP (independent weights) → h_item
    FP head     — MLP(cat[h_user, h_item]) → P(correct)
    NIP head    — cosine(h_user_before_item, h_item) for in-batch softmax
    """

    def __init__(self, cfg: ModelConfig, vocab: dict):
        super().__init__(cfg, vocab)
        self.blocks     = nn.ModuleList([ARGUSBlock(cfg) for _ in range(cfg.n_layers)])
        self.item_tower = ItemTower(vocab['n_tokens'], cfg.d_model)
        self.fp_head = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 1)
        )
        self.nip_user_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.log_temp      = nn.Parameter(torch.tensor(math.log(0.07)))

    def encode_history(self, hw, hc, ha, hd):
        x, pad, _ = self._embed(hw, hc, ha, hd)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad)
        return self.final_norm(x)

    def encode_user(self, hw, hc, ha, hd):
        return self.encode_history(hw, hc, ha, hd)[:, -1]

    def encode_item(self, word, context):
        return self.item_tower(word, context)

    def forward(self, batch):
        h_user = self.encode_user(batch['hist_word'], batch['hist_context'],
                                   batch['hist_action'], batch['hist_delta'])
        h_item = self.encode_item(batch['target_word'], batch['target_context'])
        return self.fp_head(torch.cat([h_user, h_item], dim=-1)).squeeze(-1)

    def forward_dual(self, batch):
        h = self.encode_history(batch['hist_word'], batch['hist_context'],
                                 batch['hist_action'], batch['hist_delta'])
        h_user_last   = h[:, -1]
        h_user_before = h[:, -2] if h.size(1) >= 2 else h_user_last
        h_item = self.encode_item(batch['target_word'], batch['target_context'])
        fp_logits = self.fp_head(torch.cat([h_user_last, h_item], dim=-1)).squeeze(-1)
        nip_user  = F.normalize(self.nip_user_proj(h_user_before), dim=-1)
        nip_item  = F.normalize(h_item, dim=-1)
        return fp_logits, nip_user, nip_item


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(cfg: ModelConfig, vocab: dict) -> SequentialModel:
    return {'hstu': HSTUModel, 'argus': ARGUSModel}[cfg.arch](cfg, vocab)


# ── Loss / train / eval ───────────────────────────────────────────────────────

def prob_regression_loss(logits: torch.Tensor, batch: dict) -> torch.Tensor:
    """MSE between sigmoid(logits) and target_prob; skipped where target_prob is NaN."""
    if 'target_prob' not in batch:
        return logits.new_tensor(0.0)
    target = batch['target_prob']
    mask = ~torch.isnan(target)
    if not mask.any():
        return logits.new_tensor(0.0)
    return F.mse_loss(torch.sigmoid(logits[mask]), target[mask])


def argus_loss(model: ARGUSModel, batch: dict, nip_weight: Optional[float] = None):
    if nip_weight is None:
        nip_weight = model.cfg.argus_nip_weight
    fp_logits, nip_user, nip_item = model.forward_dual(batch)
    fp_loss = F.binary_cross_entropy_with_logits(fp_logits, batch['target_label'])
    B = nip_user.size(0)
    temp = torch.clamp(model.log_temp.exp(), 0.01, 100.0)
    logits_nip = (nip_user @ nip_item.T) / temp
    nip_loss = F.cross_entropy(logits_nip, torch.arange(B, device=nip_user.device))
    reg_loss = prob_regression_loss(fp_logits, batch)
    total = fp_loss + nip_weight * nip_loss + reg_loss
    return total, fp_loss, nip_loss


def compute_loss(model: SequentialModel, batch: dict) -> torch.Tensor:
    if isinstance(model, ARGUSModel):
        total, _, _ = argus_loss(model, batch)
        return total
    logits = model(batch)
    return F.binary_cross_entropy_with_logits(logits, batch['target_label']) + prob_regression_loss(logits, batch)


def evaluate(model: SequentialModel, loader, device='cpu') -> dict:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            all_logits.append(model(batch).cpu().numpy())
            all_labels.append(batch['target_label'].cpu().numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs  = 1 / (1 + np.exp(-logits))
    try:   auc = roc_auc_score(labels, probs)
    except: auc = float('nan')
    try:   ll = log_loss(labels, np.clip(probs, 1e-6, 1 - 1e-6))
    except: ll = float('nan')
    return {
        'auc':       auc,
        'log_loss':  ll,
        'acc':       ((probs >= 0.5) == labels).mean(),
        'base_rate': labels.mean(),
        'n':         len(labels),
    }


def train(model: SequentialModel, train_ds, val_ds, cfg: TrainConfig,
          device='cpu') -> list[dict]:
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    tl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    vl = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(tl, desc=f'epoch {epoch+1}/{cfg.epochs}')
        run = None
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss(model, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            run = loss.item() if run is None else 0.9 * run + 0.1 * loss.item()
            pbar.set_postfix(loss=f'{run:.4f}')

        m = evaluate(model, vl, device=device)
        m.update({'epoch': epoch + 1, 'train_loss': run})
        history.append(m)
        print(f'  val  auc={m["auc"]:.4f}  ll={m["log_loss"]:.4f}  acc={m["acc"]:.3f}')

    return history


# ── Save / load ───────────────────────────────────────────────────────────────

def save_model(model: SequentialModel, path: Union[str, Path]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'config':     asdict(model.cfg),
        'vocab':      model.vocab,
        'state_dict': model.state_dict(),
        'version':    2,
    }, path)


def load_model(path: Union[str, Path], map_location: str = 'cpu') -> SequentialModel:
    blob = torch.load(Path(path), map_location=map_location, weights_only=False)
    cfg  = ModelConfig(**blob['config'])
    m    = build_model(cfg, blob['vocab'])
    m.load_state_dict(blob['state_dict'])
    return m.eval()


class hstu:
    """Convenience namespace: hstu.save(model, path) / model = hstu.load(path)."""
    save = staticmethod(save_model)
    load = staticmethod(load_model)


# ── Inference API ─────────────────────────────────────────────────────────────

@dataclass
class HistoryEvent:
    word: str
    context: List[str]
    action: bool
    timestamp: float = 0.0


class HSTUPredictor:
    """High-level inference wrapper — identical interface for all three architectures."""

    def __init__(self, model: SequentialModel):
        self.model = model.eval()
        self.cfg   = model.cfg
        self.t2i   = model.vocab['token2idx']

    def _build_batch(self, history: List[HistoryEvent],
                     candidates: List[Tuple], now_ts: float) -> dict:
        L, C = self.cfg.max_seq_len, self.cfg.max_context_tokens
        tail = history[-L:]
        hw = np.zeros(L, np.int64); hc = np.zeros((L, C), np.int64)
        ha = np.zeros(L, np.int64); hd = np.zeros(L, np.int64)
        prev_ts = None
        for i, ev in enumerate(tail):
            j = L - len(tail) + i
            hw[j] = self.t2i.get(ev.word, 1)
            hc[j] = encode_tokens(ev.context, self.t2i, C)
            ha[j] = 1 if ev.action else 0
            hd[j] = 0 if prev_ts is None else log_bucketize_delta(
                ev.timestamp - prev_ts, self.cfg.n_time_buckets)
            prev_ts = ev.timestamp
        last_ts = tail[-1].timestamp if tail else now_ts
        n = len(candidates)
        return {
            'hist_word':     torch.from_numpy(hw).unsqueeze(0).expand(n, -1),
            'hist_context':  torch.from_numpy(hc).unsqueeze(0).expand(n, -1, -1),
            'hist_action':   torch.from_numpy(ha).unsqueeze(0).expand(n, -1),
            'hist_delta':    torch.from_numpy(hd).unsqueeze(0).expand(n, -1),
            'target_word':   torch.tensor([self.t2i.get(w, 1) for w, _, _ in candidates],
                                          dtype=torch.long),
            'target_context': torch.from_numpy(np.array(
                [encode_tokens(ctx, self.t2i, C) for _, ctx, _ in candidates], np.int64)),
            'target_delta':  torch.tensor(
                [log_bucketize_delta(max(0., now_ts - last_ts + dt), self.cfg.n_time_buckets)
                 for _, _, dt in candidates], dtype=torch.long),
        }

    @torch.no_grad()
    def predict_action(self, history: List[HistoryEvent], new_context: List[str],
                       new_word: str, now_timestamp: float = 0.0) -> float:
        batch = self._build_batch(history, [(new_word, new_context, 0.0)], now_timestamp)
        return float(torch.sigmoid(self.model(batch)).item())

    @torch.no_grad()
    def predict_action_batch(self, history: List[HistoryEvent],
                             candidates: List[Tuple[str, List[str]]],
                             now_timestamp: float = 0.0) -> List[float]:
        if not candidates:
            return []
        batch = self._build_batch(history, [(w, c, 0.) for w, c in candidates], now_timestamp)
        return torch.sigmoid(self.model(batch)).tolist()
