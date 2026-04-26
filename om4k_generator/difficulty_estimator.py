import math
from dataclasses import dataclass
from typing import List, Optional

from .models import NoteObject


@dataclass
class _ManiaDifficultyHitObject:
    start_time: float
    end_time: float
    column: int
    delta_time: float
    column_strain_time: float
    previous_hit_objects: List[Optional["_ManiaDifficultyHitObject"]]
    index: int
    previous: Optional["_ManiaDifficultyHitObject"]


class DifficultyEstimator:
    difficulty_multiplier = 0.018
    section_length = 400
    decay_weight = 0.9
    individual_decay_base = 0.125
    overall_decay_base = 0.30
    release_threshold = 30

    @staticmethod
    def estimate_sr(notes: List[NoteObject], duration_ms: int = 0) -> float:
        """Port of osu!lazer's osu!mania strain star rating calculation."""
        difficulty_objects = DifficultyEstimator._create_difficulty_objects(notes)
        if not difficulty_objects:
            return 0.0

        difficulty_value = DifficultyEstimator._strain_difficulty_value(difficulty_objects)
        return round(difficulty_value * DifficultyEstimator.difficulty_multiplier, 2)

    @staticmethod
    def _create_difficulty_objects(notes: List[NoteObject]) -> List[_ManiaDifficultyHitObject]:
        sorted_notes = sorted(notes, key=lambda n: (round(n.time_ms), n.lane))
        if len(sorted_notes) < 2:
            return []

        objects: List[_ManiaDifficultyHitObject] = []
        per_column_objects: List[List[_ManiaDifficultyHitObject]] = [[], [], [], []]

        for i in range(1, len(sorted_notes)):
            current_note = sorted_notes[i]
            last_note = sorted_notes[i - 1]
            column = current_note.lane

            previous_in_column = per_column_objects[column][-1] if per_column_objects[column] else None
            previous_object = objects[-1] if objects else None
            previous_hit_objects: List[Optional[_ManiaDifficultyHitObject]] = [None, None, None, None]

            if previous_object is not None:
                previous_hit_objects = list(previous_object.previous_hit_objects)
                previous_hit_objects[previous_object.column] = previous_object

            start_time = float(current_note.time_ms)
            end_time = float(current_note.end_time_ms if current_note.end_time_ms is not None else current_note.time_ms)

            difficulty_object = _ManiaDifficultyHitObject(
                start_time=start_time,
                end_time=end_time,
                column=column,
                delta_time=float(current_note.time_ms - last_note.time_ms),
                column_strain_time=start_time - previous_in_column.start_time if previous_in_column else start_time,
                previous_hit_objects=previous_hit_objects,
                index=len(objects),
                previous=previous_object,
            )

            objects.append(difficulty_object)
            per_column_objects[column].append(difficulty_object)

        return objects

    @staticmethod
    def _strain_difficulty_value(objects: List[_ManiaDifficultyHitObject]) -> float:
        individual_strains = [0.0, 0.0, 0.0, 0.0]
        highest_individual_strain = 0.0
        overall_strain = 1.0
        current_strain = 0.0

        current_section_peak = 0.0
        current_section_end = 0.0
        strain_peaks: List[float] = []

        for current in objects:
            if current.index == 0:
                current_section_end = math.ceil(current.start_time / DifficultyEstimator.section_length) * DifficultyEstimator.section_length

            while current.start_time > current_section_end:
                strain_peaks.append(current_section_peak)
                current_section_peak = DifficultyEstimator._calculate_initial_strain(
                    current_section_end,
                    current,
                    highest_individual_strain,
                    overall_strain,
                )
                current_section_end += DifficultyEstimator.section_length

            individual_strains[current.column] = DifficultyEstimator._apply_decay(
                individual_strains[current.column],
                current.column_strain_time,
                DifficultyEstimator.individual_decay_base,
            )
            individual_strains[current.column] += DifficultyEstimator._individual_strain(current)

            if current.delta_time <= 1:
                highest_individual_strain = max(highest_individual_strain, individual_strains[current.column])
            else:
                highest_individual_strain = individual_strains[current.column]

            overall_strain = DifficultyEstimator._apply_decay(
                overall_strain,
                current.delta_time,
                DifficultyEstimator.overall_decay_base,
            )
            overall_strain += DifficultyEstimator._overall_strain(current)

            # osu!mania Strain uses StrainDecayBase=1 and returns the delta to the base CurrentStrain.
            current_strain += highest_individual_strain + overall_strain - current_strain
            current_section_peak = max(current_strain, current_section_peak)

        strain_peaks.append(current_section_peak)
        return DifficultyEstimator._weighted_strain_sum(strain_peaks)

    @staticmethod
    def _calculate_initial_strain(
        offset: float,
        current: _ManiaDifficultyHitObject,
        highest_individual_strain: float,
        overall_strain: float,
    ) -> float:
        previous_time = current.previous.start_time if current.previous else current.start_time
        delta = offset - previous_time
        return (
            DifficultyEstimator._apply_decay(highest_individual_strain, delta, DifficultyEstimator.individual_decay_base)
            + DifficultyEstimator._apply_decay(overall_strain, delta, DifficultyEstimator.overall_decay_base)
        )

    @staticmethod
    def _individual_strain(current: _ManiaDifficultyHitObject) -> float:
        hold_factor = 1.0

        for previous in current.previous_hit_objects:
            if previous is None:
                continue
            if DifficultyEstimator._definitely_bigger(previous.end_time, current.end_time) and DifficultyEstimator._definitely_bigger(current.start_time, previous.start_time):
                hold_factor = 1.25
                break

        return 2.0 * hold_factor

    @staticmethod
    def _overall_strain(current: _ManiaDifficultyHitObject) -> float:
        is_overlapping = False
        closest_end_time = abs(current.end_time - current.start_time)
        hold_factor = 1.0
        hold_addition = 0.0

        for previous in current.previous_hit_objects:
            if previous is None:
                continue

            is_overlapping = is_overlapping or (
                DifficultyEstimator._definitely_bigger(previous.end_time, current.start_time)
                and DifficultyEstimator._definitely_bigger(current.end_time, previous.end_time)
                and DifficultyEstimator._definitely_bigger(current.start_time, previous.start_time)
            )

            if DifficultyEstimator._definitely_bigger(previous.end_time, current.end_time) and DifficultyEstimator._definitely_bigger(current.start_time, previous.start_time):
                hold_factor = 1.25

            closest_end_time = min(closest_end_time, abs(current.end_time - previous.end_time))

        if is_overlapping:
            hold_addition = DifficultyEstimator._logistic(
                closest_end_time,
                midpoint_offset=DifficultyEstimator.release_threshold,
                multiplier=0.27,
            )

        return (1.0 + hold_addition) * hold_factor

    @staticmethod
    def _weighted_strain_sum(strain_peaks: List[float]) -> float:
        difficulty = 0.0
        weight = 1.0

        for strain in sorted((p for p in strain_peaks if p > 0), reverse=True):
            difficulty += strain * weight
            weight *= DifficultyEstimator.decay_weight

        return difficulty

    @staticmethod
    def _apply_decay(value: float, delta_time: float, decay_base: float) -> float:
        return value * math.pow(decay_base, delta_time / 1000.0)

    @staticmethod
    def _definitely_bigger(value1: float, value2: float, acceptable_difference: float = 1.0) -> bool:
        return value1 - value2 > acceptable_difference

    @staticmethod
    def _logistic(x: float, midpoint_offset: float, multiplier: float, max_value: float = 1.0) -> float:
        return max_value / (1.0 + math.exp(multiplier * (midpoint_offset - x)))
