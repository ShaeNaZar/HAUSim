"""
FSRS-based human learner simulator for synthetic data generation.

Реализует Free Spaced Repetition Scheduler (FSRS) с расширениями:
- индивидуальные параметры пользователя (Human)
- контекст задания (тип упражнения, позиция в сессии)
- интерференция между похожими словами
- усталость в течение сессии
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
#  ТИПЫ ЗАДАНИЙ
# ============================================================

class ExerciseType(Enum):
    """Тип упражнения. Множитель — насколько он сложнее базового перевода."""
    MULTIPLE_CHOICE   = ("multiple_choice",   0.70)  # узнавание
    TRANSLATE_EN_RU   = ("translate_en_ru",   0.85)  # понимание
    TRANSLATE_RU_EN   = ("translate_ru_en",   1.00)  # активное вспоминание (baseline)
    TYPING            = ("typing",            1.30)  # свободный ввод
    LISTENING         = ("listening",         1.40)  # на слух

    def __init__(self, code: str, difficulty_multiplier: float):
        self.code = code
        self.difficulty_multiplier = difficulty_multiplier


# ============================================================
#  СЛОВО
# ============================================================

@dataclass
class Word:
    """
    Изучаемое слово с лингвистическими признаками и состоянием памяти.

    stability (S) — насколько долго слово держится в памяти (в днях).
    difficulty (D) — врожденная сложность для пользователя, [1, 10].
    last_review_ts — момент последнего повторения (в днях от эпохи симуляции).
    reps — число успешных повторений подряд.
    lapses — общее число провалов.
    """
    word_id: int
    text: str
    translation: str

    # Лингвистические признаки (статичные)
    frequency_zipf: float = 4.0          # Zipf-score, 1 (редкое) — 7 (очень частое)
    length: int = 5                       # количество символов
    is_cognate: bool = False              # есть когнат в родном языке
    semantic_cluster: int = 0             # id кластера для моделирования интерференции

    # Состояние памяти (меняется по ходу обучения)
    stability: float = 0.0                # S, дни
    difficulty: float = 5.0               # D, [1, 10]
    last_review_ts: Optional[float] = None
    reps: int = 0
    lapses: int = 0
    seen: bool = False


# ============================================================
#  FSRS-ПАРАМЕТРЫ
# ============================================================

# Дефолтные веса FSRS-5 (адаптировано из open-source реализации).
# Веса можно тюнить под конкретного пользователя.
DEFAULT_FSRS_WEIGHTS = (
    0.40,  # w0  — начальная стабильность при первом успехе
    0.90,  # w1  — то же при almost-success
    2.30,  # w2  — то же при hard
    10.9,  # w3  — то же при easy
    4.93,  # w4  — начальная сложность
    0.94,  # w5  — модификатор изменения сложности
    0.86,  # w6  — стабилизация сложности
    0.01,  # w7  — гладкость сложности
    1.49,  # w8  — рост стабильности при успехе
    0.14,  # w9  — штраф к росту от высокой стабильности
    0.94,  # w10 — бонус к росту при низкой retrievability
    2.18,  # w11 — масштаб стабильности после провала
    0.05,  # w12 — влияние сложности на провал
    0.34,  # w13 — степень стабильности при провале
    1.26,  # w14 — влияние retrievability на провал
    0.29,  # w15 — модификатор для hard-успеха
    2.61,  # w16 — модификатор для easy-успеха
)


# ============================================================
#  HUMAN
# ============================================================

@dataclass
class Human:
    """
    Симулируемый пользователь-ученик.

    Внутренние параметры задают индивидуальные особенности:
    - ability: общий «талант» к языкам, [0.5, 1.5]. Влияет на скорость роста стабильности.
    - base_error_rate: фоновый шанс ошибки даже при идеальном припоминании ([0, 0.15]).
    - fatigue_rate: скорость накопления усталости в сессии.
    - interference_sensitivity: насколько сильно мешают похожие слова.
    - typo_rate: вероятность опечатки в TYPING-режиме при правильном припоминании.
    - native_language_distance: 0 (близкий язык) — 1 (далекий). Влияет на бонус от когнатов.
    - fsrs_weights: 17 весов FSRS, можно слегка варьировать вокруг дефолтных.
    """
    human_id: int

    ability: float = 1.0
    base_error_rate: float = 0.03
    fatigue_rate: float = 0.002          # на каждое слово в сессии
    interference_sensitivity: float = 0.15
    typo_rate: float = 0.05
    native_language_distance: float = 0.5
    fsrs_weights: tuple = field(default_factory=lambda: DEFAULT_FSRS_WEIGHTS)

    # Динамическое состояние в течение сессии
    _session_position: int = 0
    _recent_clusters: list = field(default_factory=list)  # последние N кластеров (для интерференции)

    # ----- основное API -----

    def reset_session(self):
        """Сбросить временные эффекты (усталость, недавние слова)."""
        self._session_position = 0
        self._recent_clusters = []

    def attempt(self, word: Word, exercise: ExerciseType, current_ts: float) -> dict:
        """
        Симулировать попытку ответа на упражнение.

        Возвращает словарь с метаданными попытки и обновляет состояние слова.
        """
        # 1. Считаем вероятность вспомнить
        if not word.seen:
            # Первое предъявление — слово неизвестно, шанс почти нулевой
            # (кроме когнатов и multiple choice, где можно угадать)
            base_recall = self._first_exposure_recall(word, exercise)
        else:
            base_recall = self._compute_retrievability(word, current_ts)

        # 2. Применяем модификаторы контекста
        recall_prob = self._apply_context_modifiers(base_recall, word, exercise)
        recall_prob = max(0.0, min(1.0, recall_prob))

        # 3. Бросаем монетку
        success = random.random() < recall_prob

        # 4. Учитываем typo в TYPING
        if success and exercise == ExerciseType.TYPING:
            if random.random() < self.typo_rate:
                success = False

        # 5. Определяем grade (для FSRS)
        if not success:
            grade = 1  # AGAIN
        else:
            # Вероятность easy/good/hard зависит от того, насколько уверенно вспомнили
            margin = recall_prob - 0.5
            r = random.random()
            if margin > 0.35 and r < 0.6:
                grade = 4  # EASY
            elif margin < 0.05 and r < 0.5:
                grade = 2  # HARD
            else:
                grade = 3  # GOOD

        # 6. Обновляем состояние слова через FSRS
        self._update_word_state(word, grade, current_ts, base_recall)

        # 7. Трекаем сессию
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

    # ----- внутренние методы -----

    def _first_exposure_recall(self, word: Word, exercise: ExerciseType) -> float:
        """Шанс «угадать» на первом предъявлении."""
        if exercise == ExerciseType.MULTIPLE_CHOICE:
            base = 0.25  # 4 варианта
        else:
            base = 0.02
        # Когнаты сильно помогают, если родной язык близок
        if word.is_cognate:
            base += 0.4 * (1 - self.native_language_distance)
        return min(0.95, base)

    def _compute_retrievability(self, word: Word, current_ts: float) -> float:
        """
        FSRS-формула retrievability:
            R(t, S) = (1 + t / (9*S)) ^ -1
        """
        if word.last_review_ts is None or word.stability <= 0:
            return 0.0
        t = max(0.0, current_ts - word.last_review_ts)
        return (1 + t / (9 * word.stability)) ** -1

    def _apply_context_modifiers(self, recall: float, word: Word, exercise: ExerciseType) -> float:
        """Контекстные эффекты: тип задания, усталость, интерференция, частотность."""
        # Тип задания: чем сложнее, тем ниже шанс вспомнить
        # Преобразуем recall через логит, применим штраф, вернем обратно
        if 0 < recall < 1:
            logit = math.log(recall / (1 - recall))
            logit -= (exercise.difficulty_multiplier - 1.0) * 1.5
            recall = 1 / (1 + math.exp(-logit))

        # Усталость
        recall *= math.exp(-self.fatigue_rate * self._session_position)

        # Интерференция: если в последних словах был тот же семантический кластер
        cluster_hits = self._recent_clusters.count(word.semantic_cluster)
        if cluster_hits > 0:
            recall *= (1 - self.interference_sensitivity * cluster_hits / 5)

        # Частотность: частые слова чуть легче (знакомость общая)
        freq_bonus = (word.frequency_zipf - 4.0) * 0.02
        recall = min(1.0, recall + freq_bonus)

        # Фоновая ошибка
        recall *= (1 - self.base_error_rate)

        return recall

    def _update_word_state(self, word: Word, grade: int, current_ts: float, retrievability: float):
        """Обновить S, D, reps, lapses согласно FSRS."""
        w = self.fsrs_weights

        if not word.seen:
            # Инициализация при первом предъявлении
            word.stability = max(0.1, w[grade - 1] * self.ability)
            word.difficulty = self._clamp(w[4] - (grade - 3), 1.0, 10.0)
            word.seen = True
        else:
            # Обновление сложности
            delta_d = -w[6] * (grade - 3)
            new_d = word.difficulty + w[5] * delta_d
            # Стабилизация к среднему
            target = w[4] - w[5] * 2  # «легкий» якорь
            word.difficulty = self._clamp(
                w[7] * target + (1 - w[7]) * new_d, 1.0, 10.0
            )

            # Обновление стабильности
            if grade == 1:  # провал
                new_s = (
                    w[11]
                    * (word.difficulty ** -w[12])
                    * (((word.stability + 1) ** w[13]) - 1)
                    * math.exp(w[14] * (1 - retrievability))
                )
                word.lapses += 1
                word.reps = 0
            else:  # успех
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

            # Применяем «талант» пользователя
            new_s *= self.ability
            word.stability = max(0.1, min(new_s, 36500.0))  # клипим до 100 лет

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
        import copy

        # Base recall probability before context modifiers
        if not word.seen:
            r_base = self._first_exposure_recall(word, exercise)
        else:
            r_base = self._compute_retrievability(word, current_ts)

        # Success probability including exercise difficulty, fatigue, interference
        p_success = self._apply_context_modifiers(r_base, word, exercise)
        p_success = max(0.0, min(1.0, p_success))

        future_ts = current_ts + horizon

        # Retrievability at horizon if we skip this exercise entirely
        r_no_review = self._compute_retrievability(word, future_ts) if word.seen else 0.0

        # Simulate success (grade=3, GOOD)
        word_ok = copy.copy(word)
        self._update_word_state(word_ok, 3, current_ts, r_base)
        r_after_success = self._compute_retrievability(word_ok, future_ts)

        # Simulate failure (grade=1, AGAIN)
        word_fail = copy.copy(word)
        self._update_word_state(word_fail, 1, current_ts, r_base)
        r_after_failure = self._compute_retrievability(word_fail, future_ts)

        return p_success * r_after_success + (1 - p_success) * r_after_failure - r_no_review

    @staticmethod
    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))


# ============================================================
#  ГЕНЕРАТОР ПОЛЬЗОВАТЕЛЕЙ
# ============================================================

class HumanFactory:
    """
    Фабрика синтетических пользователей.

    Сэмплирует параметры из правдоподобных распределений и при желании
    добавляет шум к FSRS-весам, чтобы получить «индивидуальный» алгоритм
    забывания для каждого пользователя.
    """

    def __init__(self, seed: Optional[int] = None, perturb_fsrs: bool = True,
                 perturb_scale: float = 0.10):
        self.rng = random.Random(seed)
        self.perturb_fsrs = perturb_fsrs
        self.perturb_scale = perturb_scale
        self._next_id = 0

    def sample(self) -> Human:
        """Создать одного случайного Human."""
        hid = self._next_id
        self._next_id += 1

        # Ability: логнормальное вокруг 1.0 → справа длинный хвост талантов
        ability = self._clamp(
            self.rng.lognormvariate(0.0, 0.20), 0.5, 1.8
        )

        # Базовая ошибка: бета-распределение (большинство — 2-5%)
        base_error_rate = self.rng.betavariate(2, 50)

        # Усталость: гамма
        fatigue_rate = self.rng.gammavariate(2.0, 0.001)

        # Чувствительность к интерференции
        interference_sensitivity = self._clamp(
            self.rng.gauss(0.15, 0.05), 0.0, 0.4
        )

        # Опечатки
        typo_rate = self._clamp(
            self.rng.betavariate(2, 30), 0.0, 0.3
        )

        # Языковая дистанция: для большинства задаем умеренное расстояние
        native_language_distance = self._clamp(
            self.rng.gauss(0.5, 0.15), 0.0, 1.0
        )

        # Возмущение FSRS-весов
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
#  ГЕНЕРАТОР СЛОВАРЯ (для удобства тестирования)
# ============================================================

class VocabularyFactory:
    """Простой генератор тестового словаря."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self._next_id = 0

    def sample(self, n: int, n_clusters: int = 20) -> list[Word]:
        words = []
        for i in range(n):
            wid = self._next_id
            self._next_id += 1

            # Zipf-частотность: от 2 (редкие) до 6 (частые)
            freq = self._clamp(self.rng.gauss(4.0, 1.0), 2.0, 6.5)

            words.append(Word(
                word_id=wid,
                text=f"word_{wid}",
                translation=f"слово_{wid}",
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
#  ДЕМО: симуляция нескольких сессий
# ============================================================

def simulate_learning(human: Human, vocabulary: list[Word],
                      n_sessions: int = 30,
                      words_per_session: int = 20,
                      session_interval_days: float = 1.0,
                      exercise_mix: Optional[dict] = None) -> list[dict]:
    """
    Прогнать пользователя через n_sessions учебных сессий и вернуть лог всех попыток.

    exercise_mix — словарь ExerciseType -> вероятность. По умолчанию равномерно.
    """
    if exercise_mix is None:
        exercise_mix = {et: 1.0 for et in ExerciseType}

    types, weights = zip(*exercise_mix.items())
    log = []
    current_ts = 0.0

    for s in range(n_sessions):
        human.reset_session()
        # На каждой сессии берем случайные слова (можно усложнить и брать «к повторению»)
        batch = random.sample(vocabulary, min(words_per_session, len(vocabulary)))
        for word in batch:
            exercise = random.choices(types, weights=weights, k=1)[0]
            record = human.attempt(word, exercise, current_ts)
            record["session"] = s
            log.append(record)
            # Небольшой шаг времени внутри сессии (минуты переводим в дни)
            current_ts += 1 / 1440  # 1 минута на упражнение
        # Пауза до следующей сессии
        current_ts += session_interval_days

    return log


if __name__ == "__main__":
    random.seed(42)

    # Сгенерируем 3 пользователей и 100 слов
    factory = HumanFactory(seed=42)
    humans = factory.sample_many(3)

    vocab_factory = VocabularyFactory(seed=42)
    vocab = vocab_factory.sample(100)

    print("=" * 60)
    print("Сгенерированные пользователи:")
    print("=" * 60)
    for h in humans:
        print(f"#{h.human_id}: ability={h.ability:.3f}, "
              f"error={h.base_error_rate:.3f}, "
              f"fatigue={h.fatigue_rate:.5f}, "
              f"interference={h.interference_sensitivity:.3f}, "
              f"typo={h.typo_rate:.3f}")

    print()
    print("=" * 60)
    print(f"Симуляция обучения для пользователя #{humans[0].human_id}")
    print("=" * 60)

    # У каждого пользователя — своя копия словаря (состояние памяти у всех своё)
    import copy
    personal_vocab = copy.deepcopy(vocab)
    log = simulate_learning(humans[0], personal_vocab,
                            n_sessions=20, words_per_session=15)

    # Сводка
    total = len(log)
    success_rate = sum(1 for r in log if r["success"]) / total
    avg_stability = sum(w.stability for w in personal_vocab if w.seen) / \
                    max(1, sum(1 for w in personal_vocab if w.seen))
    seen = sum(1 for w in personal_vocab if w.seen)

    print(f"Всего попыток:         {total}")
    print(f"Уникальных слов:       {seen} / {len(personal_vocab)}")
    print(f"Доля успехов:          {success_rate:.1%}")
    print(f"Средняя стабильность:  {avg_stability:.2f} дней")

    # Покажем 10 первых строк лога
    print()
    print("Первые 10 попыток:")
    print(f"{'sess':>4} {'word':>4} {'ex':>16} {'p':>5} {'ok':>3} {'g':>2} {'S':>6} {'D':>4}")
    for r in log[:10]:
        print(f"{r['session']:>4} {r['word_id']:>4} {r['exercise']:>16} "
              f"{r['recall_prob']:>5.2f} {str(r['success']):>3} {r['grade']:>2} "
              f"{r['stability']:>6.2f} {r['difficulty']:>4.1f}")
