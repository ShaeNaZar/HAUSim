import csv
import os
import random

from . import fsrs_simulator as fsrs

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

class KtoToSeleclot(Selector):
    def __init__(self, words: dict[str, str], infinite_mode: bool = True,
                 cnt_words_in_fucher: int = 2,
                 k: int = 5,
                 cnt_word_candidates: int | None = None):
        super().__init__(words, infinite_mode=infinite_mode)
        self.cnt_words_in_fucher = cnt_words_in_fucher
        self.k = max(1, k)
        self.cnt_word_candidates = (
            max(1, cnt_word_candidates)
            if cnt_word_candidates is not None
            else self.k
        )
        self.word_candidates: list[str] = []
        self.word_candidate_scores: list[tuple[str, float]] = []
        self._candidate_exercise_classes: dict[str, str] = {}

    def generate_words_after_correct_choice(
        self,
        word: str,
        exercise_class: str,
        words_state: dict[str, float] | None = None,
    ) -> dict[str, float]:
        return self._generate_words_after_choice(
            word=word,
            exercise_class=exercise_class,
            is_correct=True,
            words_state=words_state,
        )

    def generate_words_after_incorrect_choice(
        self,
        word: str,
        exercise_class: str,
        words_state: dict[str, float] | None = None,
    ) -> dict[str, float]:
        return self._generate_words_after_choice(
            word=word,
            exercise_class=exercise_class,
            is_correct=False,
            words_state=words_state,
        )

    def _generate_words_after_choice(
        self,
        word: str,
        exercise_class: str,
        is_correct: bool,
        words_state: dict[str, float] | None = None,
    ) -> dict[str, float]:
        current_words = self.words if words_state is None else words_state
        if word not in current_words:
            raise ValueError(f"Unknown active word: {word}")

        next_words = dict(current_words)
        if is_correct:
            gain = EXERCESES_PROGRESS.get(exercise_class, 0.0)
            next_words[word] = next_words.get(word, 0.0) + gain

        if not self.infinite_mode and next_words.get(word, 0.0) >= 1.0:
            next_words.pop(word, None)

        return next_words

    def update_word_candidates(self) -> list[str]:
        beam = self._beam_search(self.words, self.cnt_words_in_fucher)
        self.word_candidate_scores = [
            (first_word, score)
            for score, _, first_word, _ in beam[:self.cnt_word_candidates]
            if first_word is not None
        ]
        self.word_candidates = [word for word, _ in self.word_candidate_scores]
        self._candidate_exercise_classes = {
            first_word: first_exercise
            for _, _, first_word, first_exercise in beam
            if first_word is not None and first_exercise is not None
        }
        return self.word_candidates

    def _beam_search(
        self,
        words_state: dict[str, float],
        depth: int,
    ) -> list[tuple[float, dict[str, float], str | None, str | None]]:
        if depth <= 0 or not words_state:
            return []

        beam: list[tuple[float, dict[str, float], str | None, str | None]] = [
            (0.0, dict(words_state), None, None)
        ]

        for _ in range(depth):
            expanded: list[tuple[float, dict[str, float], str | None, str | None]] = []
            for score, state, first_word, first_exercise in beam:
                if not state:
                    expanded.append((score, state, first_word, first_exercise))
                    continue

                for word in state:
                    for exercise_class in _EXERCISE_KEYS:
                        gain, next_state = self._expected_gain_and_next_state(
                            word, exercise_class, state
                        )
                        expanded.append((
                            score + gain,
                            next_state,
                            first_word or word,
                            first_exercise or exercise_class,
                        ))

            if not expanded:
                break
            beam = self._prune_beam(expanded)

        return beam

    def _prune_beam(
        self,
        beam: list[tuple[float, dict[str, float], str | None, str | None]],
    ) -> list[tuple[float, dict[str, float], str | None, str | None]]:
        beam.sort(key=lambda item: item[0], reverse=True)

        pruned = []
        seen_first_words = set()
        for item in beam:
            first_word = item[2]
            if first_word in seen_first_words:
                continue
            pruned.append(item)
            seen_first_words.add(first_word)
            if len(pruned) >= self.k:
                break

        return pruned

    def _expected_gain_and_next_state(
        self,
        word: str,
        exercise_class: str,
        words_state: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        recognition_before = self._mean_recognition(words_state)
        p_correct = self._success_probability(word, exercise_class, words_state)

        words_after_correct = self.generate_words_after_correct_choice(
            word, exercise_class, words_state
        )
        words_after_incorrect = self.generate_words_after_incorrect_choice(
            word, exercise_class, words_state
        )

        recognition_after_correct = self._mean_recognition(words_after_correct)
        recognition_after_incorrect = self._mean_recognition(words_after_incorrect)
        expected_gain = (
            p_correct * (recognition_after_correct - recognition_before)
            + (1 - p_correct) * (recognition_after_incorrect - recognition_before)
        )

        next_state = self._expected_words_after_choice(
            current_words=words_state,
            words_after_correct=words_after_correct,
            words_after_incorrect=words_after_incorrect,
            p_correct=p_correct,
        )
        return expected_gain, next_state

    def _expected_words_after_choice(
        self,
        current_words: dict[str, float],
        words_after_correct: dict[str, float],
        words_after_incorrect: dict[str, float],
        p_correct: float,
    ) -> dict[str, float]:
        next_words = {}
        for word in current_words:
            correct_value = words_after_correct.get(word, 1.0)
            incorrect_value = words_after_incorrect.get(word, 1.0)
            expected_value = (
                p_correct * correct_value
                + (1 - p_correct) * incorrect_value
            )

            if self.infinite_mode or expected_value < 1.0:
                next_words[word] = expected_value

        return next_words

    def _success_probability(
        self,
        word: str,
        exercise_class: str,
        words_state: dict[str, float],
    ) -> float:
        familiarity = self._clamp(words_state.get(word, 0.0), 0.0, 1.0)
        complexity = EXERCESES_COMPLEXITY.get(exercise_class, 1.0)
        guess_floor = 0.25 if exercise_class == "PICK_A_WORD" else 0.02
        learned_probability = self._clamp(familiarity / complexity, 0.0, 1.0)
        probability = guess_floor + (1 - guess_floor) * learned_probability
        return self._clamp(probability, 0.0, 1.0)

    def _mean_recognition(self, words_state: dict[str, float]) -> float:
        if not self._word_levels:
            return 0.0

        total = 0.0
        for word in self._word_levels:
            if word in words_state:
                total += self._clamp(words_state[word], 0.0, 1.0)
            else:
                total += 1.0
        return total / len(self._word_levels)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def produce_next_excercise(self) -> Exercise:
        candidates = self.update_word_candidates()
        word = candidates[0] if candidates else min(self.words, key=self.words.get)

        ex = Exercise()
        ex.word = word
        ex.word_class = self.word_class(ex.word)
        ex.exercise_class = self._candidate_exercise_classes.get(
            ex.word,
            max(
                _EXERCISE_KEYS,
                key=lambda exercise_class: self._expected_gain_and_next_state(
                    ex.word, exercise_class, self.words
                )[0],
            ),
        )
        return ex
