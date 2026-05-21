"""
Synthetic session generator.

Simulates language-learning sessions for a population of synthetic users
by combining:
  - Human presets keyed by CEFR level (from algorithm.py)
  - FSRS-based recall simulation (from fsrs_simulator.py)
  - Three selector strategies: RandomSelector, QueueSelector, GreedyFSRSSelector

Output: canonical event DataFrame (same schema as slam_loader.py)
  user_id | word | context: list[str] | action: bool | timestamp: int

Context encoding:
  - A randomly chosen half of users receive context = [exercise_class, cefr_level]
  - The other half receive context = []
  (controlled by context_fraction, default 0.5)
"""

from __future__ import annotations

import copy
import math
import random
from typing import Optional

import pandas as pd

from . import fsrs_simulator as fsrs
from .algorithm import (
    CEFR_LEVELS,
    EXERCISE_TYPE_MAP,
    _HUMAN_PRESETS,
    GreedyFSRSSelector,
    QueueGenerator,
    QueueSelector,
    RandomSelector,
)

_SELECTOR_TYPES = [RandomSelector, QueueSelector, GreedyFSRSSelector]

# Seconds per day; FSRS uses days internally
_DAY = 86_400
# Base Unix timestamp so generated events look realistic
_BASE_TS = 1_600_000_000


def _make_sim_human(cefr_level: str, human_id: int) -> fsrs.Human:
    preset = _HUMAN_PRESETS.get(cefr_level.upper(), _HUMAN_PRESETS["B1"])
    return fsrs.Human(human_id=human_id, **preset)


def _lognormal_counts(n: int, total: int, rng: random.Random, sigma: float = 1.5) -> list[int]:
    """Sample n counts from log-normal, scale so sum ≈ total, minimum 1 each."""
    raw = [rng.lognormvariate(0.0, sigma) for _ in range(n)]
    scale = total / sum(raw)
    counts = [max(1, round(v * scale)) for v in raw]
    # Adjust last entry so sum is exactly total
    counts[-1] += total - sum(counts)
    counts[-1] = max(1, counts[-1])
    return counts


def generate_synthetic_events(
    n_humans: int,
    n_sessions: int,
    words_per_session: int = 10,
    max_steps_per_session: int = 100,
    session_interval_days: float = 1.0,
    context_fraction: float = 0.7,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic learning events.

    Parameters
    ----------
    n_humans            : total synthetic users (distributed equally across A1–C2)
    n_sessions          : total sessions summed across all users (log-normal distribution)
    words_per_session   : words sampled per session from the CERF word list
    max_steps_per_session: safety cap on steps per session (guards infinite_mode loops)
    session_interval_days: simulated time between sessions (days)
    context_fraction    : fraction of users whose events include [exercise_class, cefr_level]
    seed                : random seed

    Returns
    -------
    pd.DataFrame with columns: user_id, word, context, action, timestamp
    """
    rng = random.Random(seed)
    queue_gen = QueueGenerator()

    # ── Build human roster ──────────────────────────────────────
    n_per_level = n_humans // len(CEFR_LEVELS)
    remainder = n_humans - n_per_level * len(CEFR_LEVELS)

    user_levels: list[str] = []
    for level in CEFR_LEVELS:
        user_levels.extend([level] * n_per_level)
    for i in range(remainder):
        user_levels.append(CEFR_LEVELS[i % len(CEFR_LEVELS)])

    rng.shuffle(user_levels)

    has_context = [rng.random() < context_fraction for _ in range(n_humans)]

    # ── Distribute sessions via log-normal ──────────────────────
    session_counts = _lognormal_counts(n_humans, n_sessions, rng)

    # ── Simulate ────────────────────────────────────────────────
    all_rows: list[dict] = []

    for user_idx in range(n_humans):
        level = user_levels[user_idx]
        with_ctx = has_context[user_idx]
        user_id = f"syn_{user_idx}"

        sim_human = _make_sim_human(level, human_id=user_idx)
        fsrs_word_cache: dict[str, fsrs.Word] = {}

        # Stagger start times so users don't all begin at the same second
        current_ts_days = user_idx * 0.01  # ~15 min offset between users

        for sess_idx in range(session_counts[user_idx]):
            sel_type = _SELECTOR_TYPES[sess_idx % len(_SELECTOR_TYPES)]
            sel_kwargs: dict = {}
            if sel_type is GreedyFSRSSelector:
                sel_kwargs["english_level"] = level

            queue = queue_gen.construct_queue(words_per_session, sel_type, **sel_kwargs)
            sim_human.reset_session()

            step = 0
            while step < max_steps_per_session:
                if queue.is_finished():
                    break

                ex = queue.produce_next_excercise()

                # Get / create the FSRS Word for the sim human
                if ex.word not in fsrs_word_cache:
                    wid = len(fsrs_word_cache)
                    fsrs_word_cache[ex.word] = fsrs.Word(
                        word_id=wid, text=ex.word, translation=""
                    )
                fsrs_w = fsrs_word_cache[ex.word]

                ex_type = EXERCISE_TYPE_MAP.get(
                    ex.exercise_class, fsrs.ExerciseType.TRANSLATE_EN_RU
                )

                result = sim_human.attempt(fsrs_w, ex_type, current_ts_days)
                is_correct: bool = result["success"]

                queue.progress(ex, is_correct)

                context_tokens = [ex.exercise_class, level] if with_ctx else [ex.exercise_class]

                all_rows.append({
                    "user_id":   user_id,
                    "word":      ex.word,
                    "context":   context_tokens,
                    "action":    is_correct,
                    "timestamp": int(_BASE_TS + current_ts_days * _DAY),
                })

                current_ts_days += 1.0 / 1440  # 1 minute per exercise
                step += 1

            current_ts_days += session_interval_days

    df = pd.DataFrame(all_rows)
    df["action"] = df["action"].astype(bool)
    return df
