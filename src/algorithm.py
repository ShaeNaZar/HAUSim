import copy
import csv
import os
import random

from . import fsrs_simulator as fsrs
from .models import HSTUPredictor, HistoryEvent

# hardness of every exercise
EXERCESES_COMPLEXITY = {
    "PICK_A_WORD": 0.8,
    "PICK_DEFINITION": 1.0,
    "TYPE_A WORD": 1.2
}

# Familiarity growth with each correct answer
EXERCESES_PROGRESS = {
    "PICK_A_WORD": 0.3,
    "PICK_DEFINITION": 0.4,
    "TYPE_A WORD": 0.5
}

# Maps exercise class names to fsrs ExerciseType for FSRS simulation
EXERCISE_TYPE_MAP: dict[str, fsrs.ExerciseType] = {
    "PICK_A_WORD":    fsrs.ExerciseType.MULTIPLE_CHOICE,
    "PICK_DEFINITION": fsrs.ExerciseType.TRANSLATE_EN_RU,
    "TYPE_A WORD":    fsrs.ExerciseType.TYPING,
}

_EXERCISE_KEYS = list(EXERCESES_COMPLEXITY.keys())


class Exercise:
    word: str
    word_class: str  # A1, A2, B1...
    exercise_class: str


class Selector:
    def __init__(self, words: dict[str, str], infinite_mode: bool = True):
        # words: word text -> CEFR level
        # infinite_mode: when True, words are never removed on familiarity completion (default)
        self._word_levels: dict[str, str] = dict(words)
        self.words: dict[str, float] = {w: 0.0 for w in words}
        self.history: list[tuple[Exercise, bool]] = []
        self.infinite_mode = infinite_mode

    def word_class(self, word: str) -> str:
        return self._word_levels.get(word, "")

    def record_attempt(self, exercise: Exercise, is_correct: bool):
        self.history.append((exercise, is_correct))
        if is_correct:
            gain = EXERCESES_PROGRESS.get(exercise.exercise_class, 0.0)
            self.words[exercise.word] = self.words.get(exercise.word, 0.0) + gain
        if not self.infinite_mode and self.words.get(exercise.word, 0.0) >= 1.0:
            self._remove_word(exercise.word)

    def _remove_word(self, word: str):
        self.words.pop(word, None)

    def produce_next_excercise(self) -> Exercise:
        raise NotImplementedError


class QueueGenerator:
    # Loads words from words/ENGLISH_CERF_WORDS.csv
    def __init__(self):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "words", "ENGLISH_CERF_WORDS.csv")
        self._words: dict[str, str] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row["headword"].strip()
                if word not in self._words:  # keep first CEFR entry when duplicates exist
                    self._words[word] = row["CEFR"].strip()

    """
    Sample session_size random words, build a selector, return a Queue.
    Extra kwargs are forwarded to the selector constructor.
    """
    def construct_queue(self, session_size: int, selector_type: type, **kwargs) -> "Queue":
        sampled = dict(random.sample(list(self._words.items()), min(session_size, len(self._words))))
        selector = selector_type(sampled, **kwargs)
        return Queue(selector)


class Queue:
    def __init__(self, selector: Selector):
        self.selector = selector

    def produce_next_excercise(self) -> Exercise:
        return self.selector.produce_next_excercise()

    def get_history(self) -> list[tuple[Exercise, bool]]:
        return self.selector.history

    def progress(self, exercise: Exercise, is_correct_answer: bool):
        self.selector.record_attempt(exercise, is_correct_answer)

    def is_finished(self) -> bool:
        return len(self.selector.words) == 0


class RandomSelector(Selector):
    # randomly picks a word and a random exercise type
    def produce_next_excercise(self) -> Exercise:
        ex = Exercise()
        ex.word = random.choice(list(self.words.keys()))
        ex.word_class = self.word_class(ex.word)
        ex.exercise_class = random.choice(_EXERCISE_KEYS)
        return ex


class QueueSelector(Selector):
    # picks words in circular order; skips words already mastered mid-cycle
    def __init__(self, words: dict[str, str], **kwargs):
        super().__init__(words, **kwargs)
        self._word_list: list[str] = list(words.keys())
        self._index: int = 0

    def produce_next_excercise(self) -> Exercise:
        n = len(self._word_list)
        for _ in range(n):
            candidate = self._word_list[self._index % n]
            self._index += 1
            if candidate in self.words:
                ex = Exercise()
                ex.word = candidate
                ex.word_class = self.word_class(candidate)
                ex.exercise_class = random.choice(_EXERCISE_KEYS)
                return ex
        # fallback: pick any remaining active word
        ex = Exercise()
        ex.word = next(iter(self.words))
        ex.word_class = self.word_class(ex.word)
        ex.exercise_class = random.choice(_EXERCISE_KEYS)
        return ex


"""
Recognition growth explanation:

At the start of the exercise, we have recognition probability for every word

We can estimate current recognition as mean(recognition(w)) with respect to all words in the dataset

Expected Recognition Growth: P(correct | exercise) [Recognition | exercise, correct] + P(uncorrect | exercise) * [Recognition | uncorrect, exercise] - Recognition

So, for every word and possible exercise, we want estimate Expected Probability Growth, and pick a new word with maximum EPG
"""

# Human parameter presets per CEFR level.
# Higher levels → greater ability, lower error/fatigue/interference.
_HUMAN_PRESETS: dict[str, dict] = {
    "A1": dict(ability=0.60, base_error_rate=0.12, fatigue_rate=0.004,  interference_sensitivity=0.25),
    "A2": dict(ability=0.75, base_error_rate=0.08, fatigue_rate=0.003,  interference_sensitivity=0.20),
    "B1": dict(ability=0.90, base_error_rate=0.05, fatigue_rate=0.002,  interference_sensitivity=0.15),
    "B2": dict(ability=1.05, base_error_rate=0.04, fatigue_rate=0.0015, interference_sensitivity=0.12),
    "C1": dict(ability=1.20, base_error_rate=0.025, fatigue_rate=0.001, interference_sensitivity=0.08),
    "C2": dict(ability=1.40, base_error_rate=0.015, fatigue_rate=0.0008,interference_sensitivity=0.05),
}

CEFR_LEVELS = list(_HUMAN_PRESETS.keys())


class GreedyFSRSSelector(Selector):
    def __init__(self, words: dict[str, str], english_level: str = "B1",
                 step_size_days: float = 1 / 1440, **kwargs):
        super().__init__(words, **kwargs)
        self.english_level = english_level
        self._step_size_days = step_size_days
        preset = _HUMAN_PRESETS.get(english_level.upper(), _HUMAN_PRESETS["B1"])
        self.human = fsrs.Human(human_id=0, **preset)
        self._word_states: dict[str, fsrs.Word] = {}
        self._current_ts: float = 0.0

    def _get_or_create_word(self, word_text: str) -> fsrs.Word:
        if word_text not in self._word_states:
            wid = len(self._word_states)
            self._word_states[word_text] = fsrs.Word(word_id=wid, text=word_text, translation="")
        return self._word_states[word_text]

    def record_attempt(self, exercise: Exercise, is_correct: bool):
        super().record_attempt(exercise, is_correct)
        fsrs_word = self._get_or_create_word(exercise.word)
        ex_type = EXERCISE_TYPE_MAP.get(exercise.exercise_class, fsrs.ExerciseType.TRANSLATE_EN_RU)
        if not fsrs_word.seen:
            base_recall = self.human._first_exposure_recall(fsrs_word, ex_type)
        else:
            base_recall = self.human._compute_retrievability(fsrs_word, self._current_ts)
        grade = 3 if is_correct else 1
        self.human._update_word_state(fsrs_word, grade, self._current_ts, base_recall)
        self._current_ts += self._step_size_days

    # pick the word+exercise pair that maximises expected recognition growth
    def produce_next_excercise(self) -> Exercise:
        best_word_text = None
        best_exercise_type = None
        best_erg = -float("inf")

        for word_text in self.words:
            fsrs_word = self._get_or_create_word(word_text)
            for exercise_type in fsrs.ExerciseType:
                erg = self.human.estimate_erg(fsrs_word, exercise_type, self._current_ts,
                                              horizon=self._step_size_days)
                if erg > best_erg:
                    best_erg = erg
                    best_word_text = word_text
                    best_exercise_type = exercise_type

        ex = Exercise()
        ex.word = best_word_text
        ex.word_class = self.word_class(best_word_text) if best_word_text else ""
        ex.exercise_class = best_exercise_type.code if best_exercise_type else ""
        return ex

class BeamSelector(Selector):
    """
    2-step lookahead FSRS selector.

    For each candidate (word_1, exercise_1) computes:
      V = ERG_step1 + p_success * best_ERG_step2(state_after_success)
                    + p_fail    * best_ERG_step2(state_after_failure)

    Picks the candidate with maximum V. Same ERG metric as GreedyFSRSSelector,
    but looks one step further ahead.
    """

    def __init__(self, words: dict[str, str], english_level: str = "B1",
                 step_size_days: float = 1 / 1440, **kwargs):
        super().__init__(words, **kwargs)
        self.english_level = english_level
        self._step_size_days = step_size_days
        preset = _HUMAN_PRESETS.get(english_level.upper(), _HUMAN_PRESETS["B1"])
        self.human = fsrs.Human(human_id=0, **preset)
        self._word_states: dict[str, fsrs.Word] = {}
        self._current_ts: float = 0.0

    def _get_or_create_word(self, word_text: str) -> fsrs.Word:
        if word_text not in self._word_states:
            wid = len(self._word_states)
            self._word_states[word_text] = fsrs.Word(word_id=wid, text=word_text, translation="")
        return self._word_states[word_text]

    def record_attempt(self, exercise: Exercise, is_correct: bool):
        super().record_attempt(exercise, is_correct)
        fsrs_word = self._get_or_create_word(exercise.word)
        ex_type = EXERCISE_TYPE_MAP.get(exercise.exercise_class, fsrs.ExerciseType.TRANSLATE_EN_RU)
        base_recall = (self.human._first_exposure_recall(fsrs_word, ex_type)
                       if not fsrs_word.seen
                       else self.human._compute_retrievability(fsrs_word, self._current_ts))
        grade = 3 if is_correct else 1
        self.human._update_word_state(fsrs_word, grade, self._current_ts, base_recall)
        self._current_ts += self._step_size_days

    def _word_after_attempt(self, word: fsrs.Word, ex_type: fsrs.ExerciseType,
                             ts: float, grade: int) -> fsrs.Word:
        w = copy.copy(word)
        base_recall = (self.human._first_exposure_recall(w, ex_type)
                       if not w.seen
                       else self.human._compute_retrievability(w, ts))
        self.human._update_word_state(w, grade, ts, base_recall)
        return w

    def _best_erg(self, word_states: dict[str, fsrs.Word], ts: float) -> float:
        best = -float("inf")
        for word in word_states.values():
            for ex_type in fsrs.ExerciseType:
                erg = self.human.estimate_erg(word, ex_type, ts,
                                              horizon=self._step_size_days)
                if erg > best:
                    best = erg
        return best if best > -float("inf") else 0.0

    def produce_next_excercise(self) -> Exercise:
        for word_text in self.words:
            self._get_or_create_word(word_text)

        active = {w: self._word_states[w] for w in self.words}
        ts1 = self._current_ts
        ts2 = ts1 + self._step_size_days

        best_score = -float("inf")
        best_word = None
        best_ex_type = None

        for word_text, word_state in active.items():
            for ex_type in fsrs.ExerciseType:
                erg1 = self.human.estimate_erg(word_state, ex_type, ts1,
                                               horizon=self._step_size_days)
                p = self.human.success_probability(word_state, ex_type, ts1)

                w_ok  = self._word_after_attempt(word_state, ex_type, ts1, grade=3)
                w_bad = self._word_after_attempt(word_state, ex_type, ts1, grade=1)

                erg2 = (p       * self._best_erg({**active, word_text: w_ok},  ts2) +
                        (1 - p) * self._best_erg({**active, word_text: w_bad}, ts2))

                score = erg1 + erg2
                if score > best_score:
                    best_score = score
                    best_word = word_text
                    best_ex_type = ex_type

        ex = Exercise()
        ex.word = best_word or next(iter(self.words))
        ex.word_class = self.word_class(ex.word)
        ex.exercise_class = best_ex_type.code if best_ex_type else ""
        return ex


class GreedyHSTUSelector(Selector):
    """
    Greedy ERG selector that splits responsibilities between two models:

    - HSTU predictor  → p_correct  (success probability for the candidate exercise)
    - FSRS Human      → r_no_review, r_after_success, r_after_failure  (memory state)

    mode="erg"  — raw retrievability values:
        ERG = p_correct * r_after_success + (1 - p_correct) * r_after_failure - r_no_review
    mode="sigm" — shifted-sigmoid applied to each r term (same as GreedySigmoidSelector):
        ERG = p_correct * σ(r_after_success) + (1 - p_correct) * σ(r_after_failure) - σ(r_no_review)
    """

    def __init__(self, words: dict[str, str], predictor: HSTUPredictor,
                 english_level: str = "B1", step_size_days: float = 1 / 1440,
                 mode: str = "erg", **kwargs):
        super().__init__(words, **kwargs)
        if mode not in ("erg", "sigm"):
            raise ValueError(f"mode must be 'erg' or 'sigm', got {mode!r}")
        self.mode = mode
        self.predictor = predictor
        self._step_size_days = step_size_days
        preset = _HUMAN_PRESETS.get(english_level.upper(), _HUMAN_PRESETS["B1"])
        self.human = fsrs.Human(human_id=0, **preset)
        self._word_states: dict[str, fsrs.Word] = {}
        self._history: list[HistoryEvent] = []
        self._current_ts: float = 0.0

    def _get_or_create_word(self, word_text: str) -> fsrs.Word:
        if word_text not in self._word_states:
            wid = len(self._word_states)
            self._word_states[word_text] = fsrs.Word(word_id=wid, text=word_text, translation="")
        return self._word_states[word_text]

    def _context(self, word_text: str, exercise_class: str) -> list[str]:
        return [exercise_class, self.word_class(word_text)]

    def record_attempt(self, exercise: Exercise, is_correct: bool):
        super().record_attempt(exercise, is_correct)
        fsrs_word = self._get_or_create_word(exercise.word)
        ex_type = EXERCISE_TYPE_MAP.get(exercise.exercise_class, fsrs.ExerciseType.TRANSLATE_EN_RU)
        base_recall = (self.human._first_exposure_recall(fsrs_word, ex_type)
                       if not fsrs_word.seen
                       else self.human._compute_retrievability(fsrs_word, self._current_ts))
        grade = 3 if is_correct else 1
        self.human._update_word_state(fsrs_word, grade, self._current_ts, base_recall)
        self._history.append(HistoryEvent(
            word=exercise.word,
            context=self._context(exercise.word, exercise.exercise_class),
            action=is_correct,
            timestamp=self._current_ts,
        ))
        self._current_ts += self._step_size_days

    def produce_next_excercise(self) -> Exercise:
        best_word = None
        best_exercise_class = None
        best_erg = -float("inf")

        ts = self._current_ts

        for word_text in self.words:
            fsrs_word = self._get_or_create_word(word_text)
            for exercise_class in _EXERCISE_KEYS:
                ex_type = EXERCISE_TYPE_MAP.get(exercise_class, fsrs.ExerciseType.TRANSLATE_EN_RU)
                ctx = self._context(word_text, exercise_class)

                # Success probability from HSTU
                p_correct = self.predictor.predict_action(
                    self._history, ctx, word_text, ts
                )

                # Recognition estimates (r_no_review, r_after_success, r_after_failure) from FSRS
                r_no_review, r_after_success, r_after_failure = self.human.recognition_estimates(
                    fsrs_word, ex_type, ts, self._step_size_days
                )

                if self.mode == "sigm":
                    r_no_review    = fsrs.Human._shifted_sigmoid(r_no_review)
                    r_after_success = fsrs.Human._shifted_sigmoid(r_after_success)
                    r_after_failure = fsrs.Human._shifted_sigmoid(r_after_failure)

                erg = (p_correct * r_after_success
                       + (1 - p_correct) * r_after_failure
                       - r_no_review)

                if erg > best_erg:
                    best_erg = erg
                    best_word = word_text
                    best_exercise_class = exercise_class

        ex = Exercise()
        ex.word = best_word or next(iter(self.words))
        ex.word_class = self.word_class(ex.word)
        ex.exercise_class = best_exercise_class or _EXERCISE_KEYS[0]
        return ex


class GreedySigmoidSelector(Selector):
    def __init__(self, words: dict[str, str], english_level: str = "B1",
                 step_size_days: float = 1 / 1440, **kwargs):
        super().__init__(words, **kwargs)
        self.english_level = english_level
        self._step_size_days = step_size_days
        preset = _HUMAN_PRESETS.get(english_level.upper(), _HUMAN_PRESETS["B1"])
        self.human = fsrs.Human(human_id=0, **preset)
        self._word_states: dict[str, fsrs.Word] = {}
        self._current_ts: float = 0.0

    def _get_or_create_word(self, word_text: str) -> fsrs.Word:
        if word_text not in self._word_states:
            wid = len(self._word_states)
            self._word_states[word_text] = fsrs.Word(word_id=wid, text=word_text, translation="")
        return self._word_states[word_text]

    def record_attempt(self, exercise: Exercise, is_correct: bool):
        super().record_attempt(exercise, is_correct)
        fsrs_word = self._get_or_create_word(exercise.word)
        ex_type = EXERCISE_TYPE_MAP.get(exercise.exercise_class, fsrs.ExerciseType.TRANSLATE_EN_RU)
        if not fsrs_word.seen:
            base_recall = self.human._first_exposure_recall(fsrs_word, ex_type)
        else:
            base_recall = self.human._compute_retrievability(fsrs_word, self._current_ts)
        grade = 3 if is_correct else 1
        self.human._update_word_state(fsrs_word, grade, self._current_ts, base_recall)
        self._current_ts += self._step_size_days

    # pick the word+exercise pair that maximises expected recognition growth
    def produce_next_excercise(self) -> Exercise:
        best_word_text = None
        best_exercise_type = None
        best_erg = -float("inf")

        for word_text in self.words:
            fsrs_word = self._get_or_create_word(word_text)
            for exercise_type in fsrs.ExerciseType:
                erg = self.human.estimate_sigmoid(fsrs_word, exercise_type, self._current_ts,
                                              horizon=self._step_size_days)
                if erg > best_erg:
                    best_erg = erg
                    best_word_text = word_text
                    best_exercise_type = exercise_type

        ex = Exercise()
        ex.word = best_word_text
        ex.word_class = self.word_class(best_word_text) if best_word_text else ""
        ex.exercise_class = best_exercise_type.code if best_exercise_type else ""
        return ex
