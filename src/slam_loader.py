"""
Duolingo SLAM dataset loader.

Parses the raw SLAM format (en_es.slam.20190204.train) into the canonical event schema:
  user_id | word | context: list[str] | action: bool | timestamp: int

All SLAM tokens are treated as PICK_DEFINITION exercises.
Context is the full sentence token list (no English level appended).

For repeated loads, call convert_to_parquet() once to write a compact .parquet
file next to the .train file; subsequent load_duolingo_slam() calls will use it
automatically and skip the slow text parse entirely.
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


def _parse_slam_file(path: Path, max_exercises: Optional[int]) -> pd.DataFrame:
    """
    Lean parse: only collects the four fields needed for the canonical output.
    Drops token_id, pos, dep_label, session, format, and token_position at
    source so they never enter memory.
    """
    user_ids:    list[str]   = []
    exercise_ids: list[str]  = []
    days_vals:   list[float] = []
    tokens:      list[str]   = []
    labels:      list[int]   = []

    current_user = ''
    current_days = 0.0
    current_ex_id = ''
    exercise_count = 0
    pending_tokens: list[tuple[str, int]] = []   # (token, label)

    def flush():
        nonlocal exercise_count
        if not current_ex_id or not pending_tokens:
            return
        ex_id = f"{current_user}_{exercise_count}"
        for tok, lbl in pending_tokens:
            user_ids.append(current_user)
            exercise_ids.append(ex_id)
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
                current_user = kv.get('user', '')
                current_days = float(kv.get('days', 0) or 0)
                current_ex_id = current_user  # non-empty signals active exercise
                if max_exercises and exercise_count >= max_exercises:
                    break
            elif line.strip() == '':
                flush()
                pending_tokens = []
                current_ex_id = ''
            else:
                parts = line.split()
                if len(parts) < 6:
                    continue
                token = parts[1]
                label = int(parts[6]) if len(parts) > 6 else -1
                pending_tokens.append((token, label))

    flush()

    return pd.DataFrame({
        'user_id':     pd.array(user_ids,     dtype='category'),
        'exercise_id': pd.array(exercise_ids, dtype='category'),
        'days':        pd.array(days_vals,    dtype='float32'),
        'token':       pd.array(tokens,       dtype='category'),
        'label':       pd.array(labels,       dtype='int8'),
    })


def _build_canonical(raw: pd.DataFrame, ref_ts: int) -> pd.DataFrame:
    raw = raw[raw['label'] >= 0].copy()

    # Build context lists — one list object per exercise, shared across rows.
    # Sort is disabled for speed; groupby preserves insertion order in pandas >= 2.
    ctx = raw.groupby('exercise_id', sort=False)['token'].apply(list)
    raw = raw.join(ctx.rename('context'), on='exercise_id')

    out = pd.DataFrame({
        'user_id':   raw['user_id'].astype('category'),
        'word':      raw['token'].astype('category'),
        'context':   raw['context'],
        'action':    (raw['label'] == 0),
        'timestamp': (ref_ts + raw['days'].astype('float64') * 86400).astype('int64'),
    })

    validate_event_frame(out)
    return out


def convert_to_parquet(
    train_path: str | Path,
    out_path: Optional[str | Path] = None,
    max_exercises: Optional[int] = None,
    ref_ts: int = _DEFAULT_REF_TS,
) -> Path:
    """
    One-time conversion: parse the .train file and write a compact .parquet file.

    The parquet file is written next to the .train file by default.
    Returns the path of the written file.
    """
    train_path = Path(train_path)
    if out_path is None:
        out_path = train_path.with_suffix('.parquet')
    out_path = Path(out_path)

    raw = _parse_slam_file(train_path, max_exercises)
    df  = _build_canonical(raw, ref_ts)

    # Store context as a JSON string column so parquet round-trips cleanly
    # without requiring pyarrow list-type support at read time.
    import json
    df_save = df.copy()
    df_save['context'] = df_save['context'].apply(json.dumps)
    df_save.to_parquet(out_path, index=False, compression='zstd')
    return out_path


def _load_parquet(path: Path) -> pd.DataFrame:
    import json
    df = pd.read_parquet(path)
    df['context'] = df['context'].apply(json.loads)
    df['action']  = df['action'].astype(bool)
    return df


def load_duolingo_slam(
    path: str | Path,
    max_exercises: Optional[int] = None,
    ref_ts: int = _DEFAULT_REF_TS,
    save_parquet: bool = True,
) -> pd.DataFrame:
    """
    Load a SLAM .train file and return a canonical event DataFrame.

    If a .parquet file with the same stem exists next to the .train file it is
    loaded directly (fast, low memory).  Otherwise the .train file is parsed;
    when save_parquet=True the result is written to parquet for future calls.

    path          — path to en_es.slam.20190204.train (or .parquet)
    max_exercises — cap for faster dev iteration
    ref_ts        — base Unix timestamp; days-since-study-start are added to it
    save_parquet  — write .parquet on first parse so future loads are instant
    """
    path = Path(path)
    parquet_path = path.with_suffix('.parquet')

    if parquet_path.exists() and max_exercises is None:
        return _load_parquet(parquet_path)

    raw = _parse_slam_file(path, max_exercises)
    df  = _build_canonical(raw, ref_ts)

    if save_parquet and max_exercises is None:
        import json
        df_save = df.copy()
        df_save['context'] = df_save['context'].apply(json.dumps)
        df_save.to_parquet(parquet_path, index=False, compression='zstd')

    return df
