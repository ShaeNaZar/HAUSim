"""
FSRS-based human learner simulator for synthetic data generation.

Implements the Free Spaced Repetition Scheduler (FSRS) with extensions:
- individual user parameters (Human)
- exercise context (exercise type, session position)
- interference between similar words
- fatigue within a session
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
#  EXERCISE TYPES
# ============================================================

class ExerciseType(Enum):
    """Exercise type. The multiplier indicates difficulty relative to the baseline."""
    MULTIPLE_CHOICE   = ("multiple_choice",   0.70)  # recognition
    TRANSLATE_EN_RU   = ("translate_en_ru",   0.85)  # comprehension
    TRANSLATE_RU_EN   = ("translate_ru_en",   1.00)  # active recall (baseline)
    TYPING            = ("typing",            1.30)  # free input
    LISTENING         = ("listening",         1.40)  # audio

    def __init__(self, code: str, difficulty_multiplier: float):
        self.code = code
        self.difficulty_multiplier = difficulty_multiplier


# ============================================================
#  WORD
# ============================================================

@dataclass
class Word:
    """
    A vocabulary item with linguistic features and memory state.

    stability (S) — how long the word persists in memory (days).
    difficulty (D) — intrinsic difficulty for the user, [1, 10].
    last_review_ts — timestamp of the last review (days from simulation epoch).
    reps — number of consecutive successful reviews.
    lapses — total number of failures.
    """
    word_id: int
    text: str
    translation: str

    # Linguistic features (static)
    frequency_zipf: float = 4.0          # Zipf score, 1 (rare) — 7 (very common)
    length: int = 5                       # number of characters
    is_cognate: bool = False              # has a cognate in the native language
    semantic_cluster: int = 0             # cluster id for interference modelling

    # Memory state (updated during learning)
    stability: float = 0.0                # S, days
    difficulty: float = 5.0               # D, [1, 10]
    last_review_ts: Optional[float] = None
    reps: int = 0
    lapses: int = 0
    seen: bool = False


# ============================================================
#  FSRS PARAMETERS
# ============================================================

# Default FSRS-5 weights (adapted from the open-source implementation).
# Weights can be tuned per user.
DEFAULT_FSRS_WEIGHTS = (
    0.40,  # w0  — initial stability on first success
    0.90,  # w1  — same for almost-success
    2.30,  # w2  — same for hard
    10.9,  # w3  — same for easy
    4.93,  # w4  — initial difficulty
    0.94,  # w5  — difficulty change modifier
    0.86,  # w6  — difficulty stabilisation
    0.01,  # w7  — difficulty smoothing
    1.49,  # w8  — stability growth on success
    0.14,  # w9  — stability growth penalty for high stability
    0.94,  # w10 — stability growth bonus for low retrievability
    2.18,  # w11 — stability scale after failure
    0.05,  # w12 — difficulty influence on failure
    0.34,  # w13 — stability exponent on failure
    1.26,  # w14 — retrievability influence on failure
    0.29,  # w15 — modifier for hard success
    2.61,  # w16 — modifier for easy success
)


# ============================================================
#  HUMAN
# ============================================================

@dataclass
class Human:
    """
    A simulated learner.

    Internal parameters capture individual characteristics:
    - ability: overall language talent, [0.5, 1.5]. Affects stability growth rate.
    - base_error_rate: background error chance even at perfect recall ([0, 0.15]).
    - fatigue_rate: rate of fatigue accumulation within a session.
    - interference_sensitivity: how strongly similar words interfere.
    - typo_rate: probability of a typo in TYPING mode when recall is correct.
    - native_language_distance: 0 (close language) — 1 (distant). Affects cognate bonus.
    - fsrs_weights: 17 FSRS weights, can be slightly varied around the defaults.
    """
    human_id: int

    ability: float = 1.0
    base_error_rate: float = 0.03
    fatigue_rate: float = 0.002          # per exercise within a session
    interference_sensitivity: float = 0.15
    typo_rate: float = 0.05
    native_language_distance: float = 0.5
    fsrs_weights: tuple = field(default_factory=lambda: DEFAULT_FSRS_WEIGHTS)

    # Dynamic state within a session
    _session_position: int = 0
    _recent_clusters: list = field(default_factory=list)  # last N clusters (for interference)

    # ----- public API -----

    def reset_session(self):
        """Reset transient effects (fatigue, recent words)."""
        self._session_position = 0
        self._recent_clusters = []

    def success_probability(self, word: Word, exercise: ExerciseType,
                            current_ts: float) -> float:
        """
        Probability that the learner answers this word correctly now.

        Includes exercise difficulty, fatigue, interference, background error,
        and typing typos.
        """
        base_recall = self._base_recall_probability(word, exercise, current_ts)
        probability = self._contextual_recall_probability(base_recall, word, exercise)
        if exercise == ExerciseType.TYPING:
            probability *= (1 - self.typo_rate)
        return self._clamp(probability, 0.0, 1.0)

    def failure_probability(self, word: Word, exercise: ExerciseType,
                            current_ts: float) -> float:
        """Probability that the learner answers this word incorrectly now."""
        return 1.0 - self.success_probability(word, exercise, current_ts)

    def retrievability_after_success(self, word: Word, exercise: ExerciseType,
                                     current_ts: float,
                                     horizon: float = 1.0) -> float:
        """R_after_success: retrievability at current_ts + horizon after GOOD."""
        return self._retrievability_after_grade(
            word, exercise, current_ts, horizon, grade=3
        )

    def retrievability_after_failure(self, word: Word, exercise: ExerciseType,
                                     current_ts: float,
                                     horizon: float = 1.0) -> float:
        """R_after_failure: retrievability at current_ts + horizon after AGAIN."""
        return self._retrievability_after_grade(
            word, exercise, current_ts, horizon, grade=1
        )

    def attempt(self, word: Word, exercise: ExerciseType, current_ts: float) -> dict:
        """
        Simulate an attempt at an exercise.

        Returns a dict with attempt metadata and updates the word's memory state.
        """
        # 1. Compute recall probability
        base_recall = self._base_recall_probability(word, exercise, current_ts)

        # 2. Apply context modifiers
        recall_prob = self._contextual_recall_probability(base_recall, word, exercise)

        # 3. Flip the coin
        success = random.random() < recall_prob

        # 4. Account for typos in TYPING
        if success and exercise == ExerciseType.TYPING:
            if random.random() < self.typo_rate:
                success = False

        # 5. Determine grade (for FSRS)
        if not success:
            grade = 1  # AGAIN
        else:
            # Probability of easy/good/hard depends on how confidently the word was recalled
            margin = recall_prob - 0.5
            r = random.random()
            if margin > 0.35 and r < 0.6:
                grade = 4  # EASY
            elif margin < 0.05 and r < 0.5:
                grade = 2  # HARD
            else:
                grade = 3  # GOOD

        # 6. Update word memory state via FSRS
        self._update_word_state(word, grade, current_ts, base_recall)

        # 7. Track session state
        self._session_position += 1
        self._recent_clusters.append(word.semantic_cluster)
        if len(self._recent_clusters) > 5:
            self._recent_clusters.pop(0)

        return {
            "human_id":     self.human_id,
            "word_id":      word.word_id,
            "exercise":     exercise.code,
            "timestamp":    current_ts,
            "session_pos":  self._session_position,
            "recall_prob":  recall_prob,
            "success":      success,
            "grade":        grade,
            "stability":    word.stability,
            "difficulty":   word.difficulty,
            "reps":         word.reps,
            "lapses":       word.lapses,
        }

    # ----- internal methods -----

    def _base_recall_probability(self, word: Word, exercise: ExerciseType,
                                 current_ts: float) -> float:
        if not word.seen:
            # First exposure — word is unknown, chance is near zero
            # except for cognates and multiple choice where guessing is possible.
            return self._first_exposure_recall(word, exercise)
        return self._compute_retrievability(word, current_ts)

    def _contextual_recall_probability(self, base_recall: float, word: Word,
                                       exercise: ExerciseType) -> float:
        probability = self._apply_context_modifiers(base_recall, word, exercise)
        return self._clamp(probability, 0.0, 1.0)

    def _retrievability_after_grade(self, word: Word, exercise: ExerciseType,
                                    current_ts: float, horizon: float,
                                    grade: int) -> float:
        import copy

        r_base = self._base_recall_probability(word, exercise, current_ts)
        future_ts = current_ts + horizon
        word_after_attempt = copy.copy(word)
        self._update_word_state(word_after_attempt, grade, current_ts, r_base)
        return self._compute_retrievability(word_after_attempt, future_ts)

    def _first_exposure_recall(self, word: Word, exercise: ExerciseType) -> float:
        """Probability of guessing correctly on first exposure."""
        if exercise == ExerciseType.MULTIPLE_CHOICE:
            base = 0.25  # 4 choices
        else:
            base = 0.02
        # Cognates help significantly if the native language is close
        if word.is_cognate:
            base += 0.4 * (1 - self.native_language_distance)
        return min(0.95, base)

    def _compute_retrievability(self, word: Word, current_ts: float) -> float:
        """
        FSRS retrievability formula:
            R(t, S) = (1 + t / (9*S)) ^ -1
        """
        if word.last_review_ts is None or word.stability <= 0:
            return 0.0
        t = max(0.0, current_ts - word.last_review_ts)
        return (1 + t / (9 * word.stability)) ** -1

    def _apply_context_modifiers(self, recall: float, word: Word, exercise: ExerciseType) -> float:
        """Context effects: exercise type, fatigue, interference, word frequency."""
        # Exercise type: harder exercises reduce recall probability.
        # Transform recall via logit, apply penalty, convert back.
        if 0 < recall < 1:
            logit = math.log(recall / (1 - recall))
            logit -= (exercise.difficulty_multiplier - 1.0) * 1.5
            recall = 1 / (1 + math.exp(-logit))

        # Fatigue
        recall *= math.exp(-self.fatigue_rate * self._session_position)

        # Interference: if recent words were in the same semantic cluster
        cluster_hits = self._recent_clusters.count(word.semantic_cluster)
        if cluster_hits > 0:
            recall *= (1 - self.interference_sensitivity * cluster_hits / 5)

        # Word frequency: common words are slightly easier (general familiarity)
        freq_bonus = (word.frequency_zipf - 4.0) * 0.02
        recall = min(1.0, recall + freq_bonus)

        # Background error rate
        recall *= (1 - self.base_error_rate)

        return recall

    def _update_word_state(self, word: Word, grade: int, current_ts: float, retrievability: float):
        """Update S, D, reps, lapses according to FSRS."""
        w = self.fsrs_weights

        if not word.seen:
            # Initialise on first exposure
            word.stability = max(0.1, w[grade - 1] * self.ability)
            word.difficulty = self._clamp(w[4] - (grade - 3), 1.0, 10.0)
            word.seen = True
        else:
            # Update difficulty
            delta_d = -w[6] * (grade - 3)
            new_d = word.difficulty + w[5] * delta_d
            # Stabilise toward mean
            target = w[4] - w[5] * 2  # "easy" anchor
            word.difficulty = self._clamp(
                w[7] * target + (1 - w[7]) * new_d, 1.0, 10.0
            )

            # Update stability
            if grade == 1:  # failure
                new_s = (
                    w[11]
                    * (word.difficulty ** -w[12])
                    * (((word.stability + 1) ** w[13]) - 1)
                    * math.exp(w[14] * (1 - retrievability))
                )
                word.lapses += 1
                word.reps = 0
            else:  # success
                hard_penalty = w[15] if grade == 2 else 1.0
                easy_bonus = w[16] if grade == 4 else 1.0
                new_s = word.stability * (
                    math.exp(w[8])
                    * (11 - word.difficulty)
                    * (word.stability ** -w[9])
                    * (math.exp(w[10] * (1 - retrievability)) - 1)
                    * hard_penalty
                    * easy_bonus
                    + 1
                )
                word.reps += 1

            # Apply user talent
            new_s *= self.ability
            word.stability = max(0.1, min(new_s, 36500.0))  # cap at 100 years

        word.last_review_ts = current_ts

    def estimate_mean_recognition(self, words: list[Word], current_ts: float) -> float:
        """
        Mean retrievability across a set of words at current_ts.
        Unseen words contribute 0.
        """
        if not words:
            return 0.0
        return sum(self._compute_retrievability(w, current_ts) for w in words) / len(words)

    def estimate_erg(self, word: Word, exercise: ExerciseType, current_ts: float,
                     horizon: float = 1.0) -> float:
        """
        Expected Recognition Growth: how much performing this exercise improves
        expected retrievability at (current_ts + horizon).

        ERG = p_success * R(horizon | success) + p_fail * R(horizon | fail) - R(horizon | no review)
        """
        # Success probability including exercise difficulty, fatigue, interference
        p_success = self.success_probability(word, exercise, current_ts)

        future_ts = current_ts + horizon

        # Retrievability at horizon if we skip this exercise entirely
        r_no_review = self._compute_retrievability(word, future_ts) if word.seen else 0.0

        r_after_success = self.retrievability_after_success(
            word, exercise, current_ts, horizon
        )
        r_after_failure = self.retrievability_after_failure(
            word, exercise, current_ts, horizon
        )

        return p_success * r_after_success + (1 - p_success) * r_after_failure - r_no_review

    @staticmethod
    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))


# ============================================================
#  USER FACTORY
# ============================================================

class HumanFactory:
    """
    Factory for synthetic learners.

    Samples parameters from plausible distributions and optionally adds
    noise to FSRS weights to give each user an individualised forgetting curve.
    """

    def __init__(self, seed: Optional[int] = None, perturb_fsrs: bool = True,
                 perturb_scale: float = 0.10):
        self.rng = random.Random(seed)
        self.perturb_fsrs = perturb_fsrs
        self.perturb_scale = perturb_scale
        self._next_id = 0

    def sample(self) -> Human:
        """Create one random Human."""
        hid = self._next_id
        self._next_id += 1

        # Ability: log-normal around 1.0 → long right tail of talent
        ability = self._clamp(
            self.rng.lognormvariate(0.0, 0.20), 0.5, 1.8
        )

        # Background error rate: beta distribution (most users 2–5%)
        base_error_rate = self.rng.betavariate(2, 50)

        # Fatigue: gamma
        fatigue_rate = self.rng.gammavariate(2.0, 0.001)

        # Interference sensitivity
        interference_sensitivity = self._clamp(
            self.rng.gauss(0.15, 0.05), 0.0, 0.4
        )

        # Typo rate
        typo_rate = self._clamp(
            self.rng.betavariate(2, 30), 0.0, 0.3
        )

        # Language distance: moderate distance for most users
        native_language_distance = self._clamp(
            self.rng.gauss(0.5, 0.15), 0.0, 1.0
        )

        # Perturb FSRS weights
        if self.perturb_fsrs:
            weights = tuple(
                max(0.001, w * (1 + self.rng.gauss(0, self.perturb_scale)))
                for w in DEFAULT_FSRS_WEIGHTS
            )
        else:
            weights = DEFAULT_FSRS_WEIGHTS

        return Human(
            human_id=hid,
            ability=ability,
            base_error_rate=base_error_rate,
            fatigue_rate=fatigue_rate,
            interference_sensitivity=interference_sensitivity,
            typo_rate=typo_rate,
            native_language_distance=native_language_distance,
            fsrs_weights=weights,
        )

    def sample_many(self, n: int) -> list[Human]:
        return [self.sample() for _ in range(n)]

    @staticmethod
    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))


# ============================================================
#  VOCABULARY FACTORY (for testing convenience)
# ============================================================

class VocabularyFactory:
    """Simple test vocabulary generator."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self._next_id = 0

    def sample(self, n: int, n_clusters: int = 20) -> list[Word]:
        words = []
        for i in range(n):
            wid = self._next_id
            self._next_id += 1

            # Zipf frequency: from 2 (rare) to 6 (common)
            freq = self._clamp(self.rng.gauss(4.0, 1.0), 2.0, 6.5)

            words.append(Word(
                word_id=wid,
                text=f"word_{wid}",
                translation=f"translation_{wid}",
                frequency_zipf=freq,
                length=self.rng.randint(3, 12),
                is_cognate=self.rng.random() < 0.15,
                semantic_cluster=self.rng.randint(0, n_clusters - 1),
            ))
        return words

    @staticmethod
    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))


# ============================================================
#  DEMO: multi-session simulation
# ============================================================

def simulate_learning(human: Human, vocabulary: list[Word],
                      n_sessions: int = 30,
                      words_per_session: int = 20,
                      session_interval_days: float = 1.0,
                      exercise_mix: Optional[dict] = None) -> list[dict]:
    """
    Run a user through n_sessions learning sessions and return the full attempt log.

    exercise_mix — dict of ExerciseType -> probability. Defaults to uniform.
    """
    if exercise_mix is None:
        exercise_mix = {et: 1.0 for et in ExerciseType}

    types, weights = zip(*exercise_mix.items())
    log = []
    current_ts = 0.0

    for s in range(n_sessions):
        human.reset_session()
        batch = random.sample(vocabulary, min(words_per_session, len(vocabulary)))
        for word in batch:
            exercise = random.choices(types, weights=weights, k=1)[0]
            record = human.attempt(word, exercise, current_ts)
            record["session"] = s
            log.append(record)
            current_ts += 1 / 1440  # 1 minute per exercise
        current_ts += session_interval_days

    return log


if __name__ == "__main__":
    random.seed(42)

    factory = HumanFactory(seed=42)
    humans = factory.sample_many(3)

    vocab_factory = VocabularyFactory(seed=42)
    vocab = vocab_factory.sample(100)

    print("=" * 60)
    print("Generated users:")
    print("=" * 60)
    for h in humans:
        print(f"#{h.human_id}: ability={h.ability:.3f}, "
              f"error={h.base_error_rate:.3f}, "
              f"fatigue={h.fatigue_rate:.5f}, "
              f"interference={h.interference_sensitivity:.3f}, "
              f"typo={h.typo_rate:.3f}")

    print()
    print("=" * 60)
    print(f"Learning simulation for user #{humans[0].human_id}")
    print("=" * 60)

    import copy
    personal_vocab = copy.deepcopy(vocab)
    log = simulate_learning(humans[0], personal_vocab,
                            n_sessions=20, words_per_session=15)

    total = len(log)
    success_rate = sum(1 for r in log if r["success"]) / total
    avg_stability = sum(w.stability for w in personal_vocab if w.seen) / \
                    max(1, sum(1 for w in personal_vocab if w.seen))
    seen = sum(1 for w in personal_vocab if w.seen)

    print(f"Total attempts:        {total}")
    print(f"Unique words seen:     {seen} / {len(personal_vocab)}")
    print(f"Success rate:          {success_rate:.1%}")
    print(f"Mean stability:        {avg_stability:.2f} days")

    print()
    print("First 10 attempts:")
    print(f"{'sess':>4} {'word':>4} {'ex':>16} {'p':>5} {'ok':>3} {'g':>2} {'S':>6} {'D':>4}")
    for r in log[:10]:
        print(f"{r['session']:>4} {r['word_id']:>4} {r['exercise']:>16} "
              f"{r['recall_prob']:>5.2f} {str(r['success']):>3} {r['grade']:>2} "
              f"{r['stability']:>6.2f} {r['difficulty']:>4.1f}")
