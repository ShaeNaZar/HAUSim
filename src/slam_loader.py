"""
Duolingo SLAM dataset loader.

Parses the raw SLAM format (en_es.slam.20190204.train) into the canonical event schema:
  user_id | word | context: list[str] | action: bool | timestamp: int

All SLAM tokens are treated as PICK_DEFINITION exercises.
Context is the full sentence token list (no English level appended).

The parquet cache always stores the FULL dataset. Pass max_exercises to
load_duolingo_slam() to slice out the first N exercises after loading —
the cache is never regenerated for partial reads.

Call convert_to_parquet() once to write the compact cache; subsequent
load_duolingo_slam() calls use it automatically and skip the slow text parse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

CANONICAL_COLS = ['user_id', 'word', 'context', 'action', 'timestamp']

_DEFAULT_REF_TS = 1_500_000_000


def validate_event_frame(df: pd.DataFrame):
    missing = set(CANONICAL_COLS) - set(df.columns)
    assert not missing, f'Missing columns: {missing}'
    assert df['action'].dtype == bool
    assert isinstance(df['context'].iloc[0], (list, tuple))


def _parse_slam_file(path: Path) -> pd.DataFrame:
    """
    Parse the full .train file. Returns a raw DataFrame with an integer
    `exercise_num` column (global 0-indexed counter) used for slicing later.
    """
    user_ids:      list[str]   = []
    exercise_ids:  list[str]   = []
    exercise_nums: list[int]   = []
    days_vals:     list[float] = []
    tokens:        list[str]   = []
    labels:        list[int]   = []

    current_user   = ''
    current_days   = 0.0
    current_ex_id  = ''
    exercise_count = 0
    pending_tokens: list[tuple[str, int]] = []

    def flush():
        nonlocal exercise_count
        if not current_ex_id or not pending_tokens:
            return
        ex_id = f"{current_user}_{exercise_count}"
        for tok, lbl in pending_tokens:
            user_ids.append(current_user)
            exercise_ids.append(ex_id)
            exercise_nums.append(exercise_count)
            days_vals.append(current_days)
            tokens.append(tok)
            labels.append(lbl)
        exercise_count += 1

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('# prompt'):
                continue
            if line.startswith('#'):
                flush()
                pending_tokens = []
                kv: dict[str, str] = {}
                for part in line[1:].strip().split():
                    if ':' in part:
                        k, v = part.split(':', 1)
                        kv[k] = v
                current_user  = kv.get('user', '')
                current_days  = float(kv.get('days', 0) or 0)
                current_ex_id = current_user
            elif line.strip() == '':
                flush()
                pending_tokens = []
                current_ex_id  = ''
            else:
                parts = line.split()
                if len(parts) < 6:
                    continue
                token = parts[1]
                label = int(parts[6]) if len(parts) > 6 else -1
                pending_tokens.append((token, label))

    flush()

    return pd.DataFrame({
        'user_id':      pd.array(user_ids,      dtype='category'),
        'exercise_id':  pd.array(exercise_ids,  dtype='category'),
        'exercise_num': pd.array(exercise_nums, dtype='int32'),
        'days':         pd.array(days_vals,     dtype='float32'),
        'token':        pd.array(tokens,        dtype='category'),
        'label':        pd.array(labels,        dtype='int8'),
    })


def _build_canonical(raw: pd.DataFrame, ref_ts: int) -> pd.DataFrame:
    raw = raw[raw['label'] >= 0].copy()

    ctx = raw.groupby('exercise_id', sort=False)['token'].apply(list)
    raw = raw.join(ctx.rename('context'), on='exercise_id')

    out = pd.DataFrame({
        'exercise_num': raw['exercise_num'],
        'user_id':      raw['user_id'].astype('category'),
        'word':         raw['token'].astype('category'),
        'context':      raw['context'],
        'action':       (raw['label'] == 0),
        'timestamp':    (ref_ts + raw['days'].astype('float64') * 86400).astype('int64'),
    })

    validate_event_frame(out.drop(columns=['exercise_num']))
    return out


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    import json
    df_save = df.copy()
    df_save['context'] = df_save['context'].apply(json.dumps)
    df_save.to_parquet(path, index=False, compression='zstd')


def _load_parquet(path: Path) -> pd.DataFrame:
    import json
    df = pd.read_parquet(path)
    df['context'] = df['context'].apply(json.loads)
    df['action']  = df['action'].astype(bool)
    return df


def _apply_max_exercises(df: pd.DataFrame, max_exercises: Optional[int]) -> pd.DataFrame:
    """Keep only the first max_exercises exercises, then drop the helper column."""
    if max_exercises is not None:
        df = df[df['exercise_num'] < max_exercises].copy()
    return df.drop(columns=['exercise_num'])


def convert_to_parquet(
    train_path: str | Path,
    out_path: Optional[str | Path] = None,
    ref_ts: int = _DEFAULT_REF_TS,
) -> Path:
    """
    One-time conversion: parse the FULL .train file and write a compact .parquet.

    The parquet file is written next to the .train file by default.
    Returns the path of the written file.
    """
    train_path = Path(train_path)
    if out_path is None:
        out_path = train_path.with_suffix('.parquet')
    out_path = Path(out_path)

    raw = _parse_slam_file(train_path)
    df  = _build_canonical(raw, ref_ts)
    _save_parquet(df, out_path)
    return out_path


def load_duolingo_slam(
    path: str | Path,
    max_exercises: Optional[int] = None,
    ref_ts: int = _DEFAULT_REF_TS,
    save_parquet: bool = True,
) -> pd.DataFrame:
    """
    Load a SLAM .train file and return a canonical event DataFrame.

    The parquet cache holds the FULL dataset. If it exists it is always used
    regardless of max_exercises, keeping re-parses out of the hot path.
    max_exercises slices the first N exercises from the already-loaded data.

    path          — path to en_es.slam.20190204.train (or .parquet)
    max_exercises — return only the first N exercises (for dev iteration)
    ref_ts        — base Unix timestamp; days-since-study-start are added to it
    save_parquet  — write .parquet on first parse so future loads are instant
    """
    path         = Path(path)
    parquet_path = path.with_suffix('.parquet')

    if parquet_path.exists():
        df = _load_parquet(parquet_path)
    else:
        raw = _parse_slam_file(path)
        df  = _build_canonical(raw, ref_ts)
        if save_parquet:
            _save_parquet(df, parquet_path)

    df = _apply_max_exercises(df, max_exercises)
    validate_event_frame(df)
    return df
