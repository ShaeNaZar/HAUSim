"""
Sequence dataset utilities.

Preprocessing pipeline (vocab building, event enrichment, sequence slicing)
and SequenceWindowDataset that can merge SLAM + synthetic events transparently.

Public API
----------
build_dataset(slam_df, synthetic_df, data_cfg, model_cfg)
    → train_ds, val_ds, vocab

SequenceWindowDataset(sequences, max_seq_len, max_context_tokens, mode, ...)
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CANONICAL_COLS = ['user_id', 'word', 'context', 'action', 'timestamp']


# ── Preprocessing helpers ──────────────────────────────────────────────────────

def log_bucketize_delta(delta_sec: float, n_buckets: int,
                        max_val: float = 60 * 60 * 24 * 365) -> int:
    x = max(0.0, min(float(delta_sec), max_val))
    log_x = math.log1p(x)
    log_max = math.log1p(max_val)
    return max(0, min(n_buckets - 1, int(math.floor(log_x / log_max * (n_buckets - 1e-9)))))


def encode_tokens(tokens: list[str], token2idx: dict, max_len: int) -> list[int]:
    out = [token2idx.get(t, 1) for t in tokens[:max_len]]
    return out + [0] * (max_len - len(out))


def build_token_vocab(events: pd.DataFrame, min_freq: int = 1) -> dict:
    counter: Counter = Counter()
    for w in events['word']:
        counter[w] += 1
    for ctx in events['context']:
        for t in ctx:
            counter[t] += 1
    t2i = {'<pad>': 0, '<unk>': 1}
    for tok, cnt in counter.most_common():
        if cnt >= min_freq and tok not in t2i:
            t2i[tok] = len(t2i)
    return {'token2idx': t2i, 'n_tokens': len(t2i)}


def prepare_events(events: pd.DataFrame, data_cfg, model_cfg) -> tuple[pd.DataFrame, dict]:
    """Filter users by event count, subsample, build vocab, add derived columns."""
    df = events.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    uc = df.groupby('user_id').size()
    keep = uc[
        (uc >= data_cfg.min_events_per_user) &
        (uc <= data_cfg.max_events_per_user)
    ].index
    df = df[df['user_id'].isin(keep)]

    if data_cfg.subsample_users:
        rng = np.random.default_rng(0)
        s = rng.choice(
            df['user_id'].unique(),
            size=min(data_cfg.subsample_users, df['user_id'].nunique()),
            replace=False,
        )
        df = df[df['user_id'].isin(s)]

    vocab = build_token_vocab(df, min_freq=data_cfg.min_token_frequency)
    t2i = vocab['token2idx']
    C = model_cfg.max_context_tokens

    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True).copy()
    df['word_idx'] = df['word'].map(lambda w: t2i.get(w, 1)).astype(np.int64)
    df['context_idx'] = df['context'].map(lambda c: encode_tokens(c, t2i, C))
    df['delta_sec'] = df.groupby('user_id')['timestamp'].diff().fillna(0).clip(lower=0)
    df['delta_bucket'] = df['delta_sec'].map(
        lambda d: log_bucketize_delta(d, model_cfg.n_time_buckets)
    ).astype(np.int64)

    return df, vocab


def to_user_sequences(df: pd.DataFrame) -> list[dict]:
    seqs = []
    for uid, g in df.groupby('user_id', sort=False):
        g = g.sort_values('timestamp')
        seqs.append({
            'user_id':     uid,
            'word_idx':    g['word_idx'].values.astype(np.int64),
            'context_idx': np.array(list(g['context_idx'].values), dtype=np.int64),
            'action':      g['action'].values.astype(np.int64),
            'delta_bucket': g['delta_bucket'].values.astype(np.int64),
            'timestamp':   g['timestamp'].values.astype(np.int64),
        })
    return seqs


def split_sequences(sequences: list[dict], val_fraction: float) -> tuple[list[dict], list[dict]]:
    train, val = [], []
    for s in sequences:
        n = len(s['word_idx'])
        if n < 5:
            continue
        cut = max(1, int(n * (1 - val_fraction)))
        train.append({k: (v[:cut] if isinstance(v, np.ndarray) else v) for k, v in s.items()})
        v = dict(s)
        v['val_start'] = cut
        val.append(v)
    return train, val


# ── Dataset ───────────────────────────────────────────────────────────────────

class SequenceWindowDataset(Dataset):
    """
    Sliding-window dataset over user event sequences.

    Works identically for SLAM, synthetic, or merged event streams.
    In 'train' mode samples `targets_per_user` random positions per sequence.
    In 'val' mode uses every position from val_start to end.
    """

    def __init__(
        self,
        sequences: list[dict],
        max_seq_len: int,
        max_context_tokens: int,
        mode: str = 'train',
        targets_per_user: int = 20,
    ):
        self.seqs = sequences
        self.L = max_seq_len
        self.C = max_context_tokens
        self.mode = mode
        self.examples: list[tuple[int, int]] = []

        rng = np.random.default_rng(42 if mode == 'train' else 7)
        for i, s in enumerate(sequences):
            n = len(s['word_idx'])
            if mode == 'train':
                if n < 2:
                    continue
                for p in rng.integers(1, n, size=min(targets_per_user, n - 1)):
                    self.examples.append((i, int(p)))
            else:
                for p in range(s['val_start'], n):
                    self.examples.append((i, p))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        si, t = self.examples[idx]
        s = self.seqs[si]
        L, C = self.L, self.C
        start = max(0, t - L)

        words  = s['word_idx'][start:t]
        ctxs   = s['context_idx'][start:t]
        acts   = s['action'][start:t]
        deltas = s['delta_bucket'][start:t]

        pad = L - len(words)
        if pad > 0:
            words  = np.concatenate([np.zeros(pad, np.int64), words])
            acts   = np.concatenate([np.zeros(pad, np.int64), acts])
            deltas = np.concatenate([np.zeros(pad, np.int64), deltas])
            ctxs   = (np.concatenate([np.zeros((pad, C), np.int64), ctxs])
                      if len(ctxs) else np.zeros((L, C), np.int64))

        return {
            'hist_word':      torch.from_numpy(words),
            'hist_context':   torch.from_numpy(ctxs),
            'hist_action':    torch.from_numpy(acts),
            'hist_delta':     torch.from_numpy(deltas),
            'target_word':    torch.tensor(int(s['word_idx'][t]),      dtype=torch.long),
            'target_context': torch.from_numpy(s['context_idx'][t]),
            'target_delta':   torch.tensor(int(s['delta_bucket'][t]),  dtype=torch.long),
            'target_label':   torch.tensor(float(s['action'][t]),      dtype=torch.float32),
        }


# ── Top-level builder ─────────────────────────────────────────────────────────

def build_dataset(
    slam_df: Optional[pd.DataFrame],
    synthetic_df: Optional[pd.DataFrame],
    data_cfg,
    model_cfg,
) -> tuple[SequenceWindowDataset, SequenceWindowDataset, dict]:
    """
    Merge SLAM and synthetic event frames, run the full preprocessing
    pipeline, and return (train_ds, val_ds, vocab).

    Either argument may be None to use only one source.
    """
    frames = [df for df in (slam_df, synthetic_df) if df is not None and len(df) > 0]
    if not frames:
        raise ValueError("At least one of slam_df / synthetic_df must be non-empty")

    events = pd.concat(frames, ignore_index=True)

    enriched, vocab = prepare_events(events, data_cfg, model_cfg)
    sequences = to_user_sequences(enriched)
    train_seqs, val_seqs = split_sequences(sequences, data_cfg.val_fraction)

    train_ds = SequenceWindowDataset(
        train_seqs, model_cfg.max_seq_len, model_cfg.max_context_tokens,
        mode='train', targets_per_user=40,
    )
    val_ds = SequenceWindowDataset(
        val_seqs, model_cfg.max_seq_len, model_cfg.max_context_tokens,
        mode='val',
    )

    return train_ds, val_ds, vocab
