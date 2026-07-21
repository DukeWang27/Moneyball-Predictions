"""Leakage-safe champion/challenger tournament for MLB pregame probabilities.

The tournament deliberately keeps the feature pipeline fixed and changes only the
model family.  A regularized logistic model remains the champion baseline.  A small
histogram gradient-boosted tree, a Poisson expected-runs model, and a conservative
ensemble compete on chronological validation seasons.  A challenger is promoted only
when it improves mean Brier score across folds without materially damaging any fold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .model import clip_probability, logit
from .optimized import (
    FeatureRow,
    ModelVariant,
    OptimizedModelArtifact,
    ProbabilityCalibrator,
    SeasonSummary,
    apply_multifold_selection,
    build_season_rows,
    train_optimized_artifact,
)

TOURNAMENT_FEATURE_INDICES = (0, 3, 4, 8, 9, 10, 11, 12, 13, 14)
PROMOTION_MEAN_BRIER_GAIN = 0.0005
MAX_SINGLE_FOLD_DAMAGE = 0.0005
MAX_MEAN_LOG_LOSS_DAMAGE = 0.0003


class ProbabilityPredictor(Protocol):
    def predict(self, features: tuple[float, ...]) -> float: ...


@dataclass
class RawLinearPredictor:
    variant: ModelVariant

    def predict(self, features: tuple[float, ...]) -> float:
        return self.variant.model.predict(features)


@dataclass
class BoostedTreePredictor:
    feature_indices: tuple[int, ...]
    model: HistGradientBoostingClassifier | None
    fallback_probability: float

    @classmethod
    def fit(
        cls,
        rows: list[FeatureRow],
        *,
        feature_indices: tuple[int, ...] = TOURNAMENT_FEATURE_INDICES,
    ) -> "BoostedTreePredictor":
        fallback = float(np.mean([row.home_win for row in rows])) if rows else 0.54
        if len(rows) < 80 or len({row.home_win for row in rows}) < 2:
            return cls(feature_indices, None, clip_probability(fallback, 0.20, 0.80))
        x = np.asarray(
            [[row.features[index] for index in feature_indices] for row in rows], dtype=float
        )
        y = np.asarray([row.home_win for row in rows], dtype=int)
        min_leaf = max(20, min(60, len(rows) // 30))
        model = HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=160,
            max_leaf_nodes=7,
            min_samples_leaf=min_leaf,
            l2_regularization=4.0,
            max_bins=64,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=27,
        )
        model.fit(x, y)
        return cls(feature_indices, model, clip_probability(fallback, 0.20, 0.80))

    def predict(self, features: tuple[float, ...]) -> float:
        if self.model is None:
            return self.fallback_probability
        x = np.asarray([[features[index] for index in self.feature_indices]], dtype=float)
        return clip_probability(float(self.model.predict_proba(x)[0, 1]), 0.08, 0.92)


@dataclass
class ExpectedRunsPredictor:
    feature_indices: tuple[int, ...]
    away_model: HistGradientBoostingRegressor | None
    home_model: HistGradientBoostingRegressor | None
    fallback_away_runs: float
    fallback_home_runs: float

    @classmethod
    def fit(
        cls,
        rows: list[FeatureRow],
        *,
        feature_indices: tuple[int, ...] = TOURNAMENT_FEATURE_INDICES,
    ) -> "ExpectedRunsPredictor":
        away_values = [float(row.game.away_score or 0) for row in rows]
        home_values = [float(row.game.home_score or 0) for row in rows]
        fallback_away = float(np.mean(away_values)) if away_values else 4.35
        fallback_home = float(np.mean(home_values)) if home_values else 4.55
        if len(rows) < 80:
            return cls(feature_indices, None, None, fallback_away, fallback_home)
        x = np.asarray(
            [[row.features[index] for index in feature_indices] for row in rows], dtype=float
        )
        away_y = np.asarray(away_values, dtype=float)
        home_y = np.asarray(home_values, dtype=float)
        min_leaf = max(20, min(60, len(rows) // 30))

        def fit_one(target: np.ndarray) -> HistGradientBoostingRegressor:
            model = HistGradientBoostingRegressor(
                loss="poisson",
                learning_rate=0.035,
                max_iter=160,
                max_leaf_nodes=7,
                min_samples_leaf=min_leaf,
                l2_regularization=4.0,
                max_bins=64,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=27,
            )
            model.fit(x, target)
            return model

        return cls(
            feature_indices,
            fit_one(away_y),
            fit_one(home_y),
            fallback_away,
            fallback_home,
        )

    def expected_runs(self, features: tuple[float, ...]) -> tuple[float, float]:
        if self.away_model is None or self.home_model is None:
            return self.fallback_away_runs, self.fallback_home_runs
        x = np.asarray([[features[index] for index in self.feature_indices]], dtype=float)
        away = max(float(self.away_model.predict(x)[0]), 0.25)
        home = max(float(self.home_model.predict(x)[0]), 0.25)
        return away, home

    def predict(self, features: tuple[float, ...]) -> float:
        away_runs, home_runs = self.expected_runs(features)
        return _poisson_home_win_probability(away_runs, home_runs)


@dataclass
class EnsemblePredictor:
    predictors: tuple[ProbabilityPredictor, ...]
    weights: tuple[float, ...]

    def predict(self, features: tuple[float, ...]) -> float:
        probability = sum(
            weight * predictor.predict(features)
            for weight, predictor in zip(self.weights, self.predictors, strict=True)
        )
        return clip_probability(probability, 0.08, 0.92)


@dataclass
class CalibratedPredictor:
    raw_predictor: ProbabilityPredictor
    calibrator: ProbabilityCalibrator

    def predict(self, features: tuple[float, ...]) -> float:
        return self.calibrator.transform(self.raw_predictor.predict(features))


@dataclass(frozen=True)
class TournamentCandidate:
    key: str
    label: str
    family: str
    predictor: ProbabilityPredictor
    validation_brier: float | None
    validation_log_loss: float | None
    calibration_method: str
    blend_weights: tuple[float, ...] = ()

    def predict(self, features: tuple[float, ...]) -> float:
        return self.predictor.predict(features)


@dataclass(frozen=True)
class ModelTournamentArtifact:
    candidates: dict[str, TournamentCandidate]
    champion_key: str
    champion_reason: str
    promoted_challenger: bool
    fold_briers: dict[str, tuple[float, ...]] = field(default_factory=dict)
    fold_log_losses: dict[str, tuple[float, ...]] = field(default_factory=dict)
    selection_folds: tuple[int, ...] = ()
    training_rows: int = 0
    validation_rows: int = 0
    linear_artifact: OptimizedModelArtifact | None = None

    @property
    def champion(self) -> TournamentCandidate:
        return self.candidates[self.champion_key]


@dataclass(frozen=True)
class TournamentPreparation:
    artifact: ModelTournamentArtifact
    prior_summary: SeasonSummary
    prior_elo: dict[str, float]


def _poisson_pmf(mean: float, maximum: int = 24) -> list[float]:
    mean = max(mean, 0.01)
    values = [math.exp(-mean)]
    for runs in range(1, maximum + 1):
        values.append(values[-1] * mean / runs)
    total = sum(values)
    if total <= 0:
        return [1.0] + [0.0] * maximum
    return [value / total for value in values]


def _poisson_home_win_probability(away_mean: float, home_mean: float) -> float:
    away = _poisson_pmf(away_mean)
    home = _poisson_pmf(home_mean)
    probability = 0.0
    tie = 0.0
    away_cumulative = 0.0
    for runs, home_probability in enumerate(home):
        if runs > 0:
            away_cumulative += away[runs - 1]
        probability += home_probability * away_cumulative
        tie += home_probability * away[runs]
    # Tied regulation games continue; use a modest home advantage rather than treating a tie as a draw.
    probability += 0.54 * tie
    return clip_probability(probability, 0.08, 0.92)


def _fit_platt(raw: list[float], rows: list[FeatureRow]) -> ProbabilityCalibrator:
    if len(rows) < 80:
        return ProbabilityCalibrator()
    x = np.asarray([logit(clip_probability(value)) for value in raw], dtype=float)
    y = np.asarray([row.home_win for row in rows], dtype=float)
    design = np.column_stack((np.ones(len(x)), x))
    coefficients = np.asarray([0.0, 1.0], dtype=float)
    penalty = np.diag([0.2, 1.5])
    for _ in range(60):
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weight = np.clip(probability * (1.0 - probability), 1e-7, None)
        gradient = design.T @ (probability - y) + penalty @ (
            coefficients - np.asarray([0.0, 1.0])
        )
        hessian = design.T @ (design * weight[:, None]) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return ProbabilityCalibrator(
        method="platt",
        intercept=float(coefficients[0]),
        slope=float(coefficients[1]),
    )


def _score(probabilities: list[float], rows: list[FeatureRow]) -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    brier = 0.0
    log_loss = 0.0
    for probability, row in zip(probabilities, rows, strict=True):
        clipped = clip_probability(probability)
        brier += (probability - row.home_win) ** 2
        log_loss += -(
            row.home_win * math.log(clipped)
            + (1.0 - row.home_win) * math.log(1.0 - clipped)
        )
    return brier / len(rows), log_loss / len(rows)


def _calibrate(
    raw_predictor: ProbabilityPredictor,
    validation_rows: list[FeatureRow],
) -> tuple[CalibratedPredictor, float | None, float | None, str]:
    raw = [raw_predictor.predict(row.features) for row in validation_rows]
    candidates = {
        "identity": ProbabilityCalibrator(),
        "shrink10": ProbabilityCalibrator(method="shrink", shrinkage=0.10),
        "shrink20": ProbabilityCalibrator(method="shrink", shrinkage=0.20),
        "shrink30": ProbabilityCalibrator(method="shrink", shrinkage=0.30),
        "shrink40": ProbabilityCalibrator(method="shrink", shrinkage=0.40),
        "platt": _fit_platt(raw, validation_rows),
    }
    scored: list[tuple[float, float, str, ProbabilityCalibrator]] = []
    for name, calibrator in candidates.items():
        probabilities = [calibrator.transform(value) for value in raw]
        brier, log_loss = _score(probabilities, validation_rows)
        if brier is not None and log_loss is not None:
            scored.append((brier, log_loss, name, calibrator))
    if not scored:
        calibrator = ProbabilityCalibrator(method="shrink", shrinkage=0.20)
        return CalibratedPredictor(raw_predictor, calibrator), None, None, "shrink20"
    # Use Brier first, then log loss, then prefer simpler calibration in near-ties.
    scored.sort(key=lambda item: (item[0], item[1], item[2] != "identity"))
    best_brier = scored[0][0]
    near = [item for item in scored if item[0] <= best_brier + 0.00015]
    near.sort(key=lambda item: (item[1], item[2] != "identity"))
    brier, log_loss, name, calibrator = near[0]
    return CalibratedPredictor(raw_predictor, calibrator), brier, log_loss, name


def _fit_single_fold(
    training_rows: list[FeatureRow],
    validation_rows: list[FeatureRow],
    linear_variant: ModelVariant,
) -> dict[str, TournamentCandidate]:
    linear_raw = RawLinearPredictor(linear_variant)
    tree_raw = BoostedTreePredictor.fit(training_rows)
    runs_raw = ExpectedRunsPredictor.fit(training_rows)

    linear, linear_brier, linear_log, linear_cal = _calibrate(linear_raw, validation_rows)
    tree, tree_brier, tree_log, tree_cal = _calibrate(tree_raw, validation_rows)
    runs, runs_brier, runs_log, runs_cal = _calibrate(runs_raw, validation_rows)

    candidates: dict[str, TournamentCandidate] = {
        "logistic": TournamentCandidate(
            key="logistic",
            label="Regularized logistic",
            family="Linear",
            predictor=linear,
            validation_brier=linear_brier,
            validation_log_loss=linear_log,
            calibration_method=linear_cal,
        ),
        "boosted_tree": TournamentCandidate(
            key="boosted_tree",
            label="Boosted decision trees",
            family="HistGradientBoosting",
            predictor=tree,
            validation_brier=tree_brier,
            validation_log_loss=tree_log,
            calibration_method=tree_cal,
        ),
        "expected_runs": TournamentCandidate(
            key="expected_runs",
            label="Expected runs (Poisson)",
            family="Poisson run model",
            predictor=runs,
            validation_brier=runs_brier,
            validation_log_loss=runs_log,
            calibration_method=runs_cal,
        ),
    }

    blends = {
        "ensemble_70_20_10": (0.70, 0.20, 0.10),
        "ensemble_60_25_15": (0.60, 0.25, 0.15),
        "ensemble_50_30_20": (0.50, 0.30, 0.20),
        "ensemble_60_40_00": (0.60, 0.40, 0.00),
    }
    ensemble_scores: list[tuple[float, float, str, tuple[float, ...], EnsemblePredictor]] = []
    for key, weights in blends.items():
        predictor = EnsemblePredictor((linear, tree, runs), weights)
        probabilities = [predictor.predict(row.features) for row in validation_rows]
        brier, log_loss = _score(probabilities, validation_rows)
        if brier is not None and log_loss is not None:
            ensemble_scores.append((brier, log_loss, key, weights, predictor))
    if ensemble_scores:
        ensemble_scores.sort(key=lambda item: (item[0], item[1]))
        brier, log_loss, _, weights, predictor = ensemble_scores[0]
        candidates["ensemble"] = TournamentCandidate(
            key="ensemble",
            label="Validation-selected ensemble",
            family="Blend",
            predictor=predictor,
            validation_brier=brier,
            validation_log_loss=log_loss,
            calibration_method="component-calibrated blend",
            blend_weights=weights,
        )
    return candidates


def _fold_rows(
    *,
    seed_games: list,
    training_games: list,
    validation_games: list,
    min_games: int,
) -> tuple[
    list[FeatureRow],
    list[FeatureRow],
    SeasonSummary,
    OptimizedModelArtifact,
]:
    _, seed_engine = build_season_rows(
        seed_games, prior_summary=None, prior_elo=None, min_games=min_games
    )
    training_rows, training_engine = build_season_rows(
        training_games,
        prior_summary=seed_engine.summary(),
        prior_elo=None,
        min_games=min_games,
    )
    validation_rows, validation_engine = build_season_rows(
        validation_games,
        prior_summary=training_engine.summary(),
        prior_elo=None,
        min_games=min_games,
    )
    linear_artifact = train_optimized_artifact(training_rows, validation_rows)
    return training_rows, validation_rows, validation_engine.summary(), linear_artifact


def _choose_architecture(
    fold_candidates: list[dict[str, TournamentCandidate]],
) -> tuple[str, str, bool, dict[str, tuple[float, ...]], dict[str, tuple[float, ...]]]:
    common = set(fold_candidates[0])
    for candidates in fold_candidates[1:]:
        common &= set(candidates)
    fold_briers: dict[str, tuple[float, ...]] = {}
    fold_logs: dict[str, tuple[float, ...]] = {}
    for key in sorted(common):
        briers = tuple(
            float(candidates[key].validation_brier)
            for candidates in fold_candidates
            if candidates[key].validation_brier is not None
        )
        logs = tuple(
            float(candidates[key].validation_log_loss)
            for candidates in fold_candidates
            if candidates[key].validation_log_loss is not None
        )
        if len(briers) == len(fold_candidates):
            fold_briers[key] = briers
        if len(logs) == len(fold_candidates):
            fold_logs[key] = logs

    if "logistic" not in fold_briers:
        return (
            "logistic",
            "Insufficient validation history; retained the interpretable logistic champion.",
            False,
            fold_briers,
            fold_logs,
        )
    logistic_mean = float(np.mean(fold_briers["logistic"]))
    logistic_log_mean = float(np.mean(fold_logs.get("logistic", (1.0,))))
    eligible: list[tuple[float, str]] = []
    for key, scores in fold_briers.items():
        if key == "logistic":
            continue
        mean_score = float(np.mean(scores))
        mean_gain = logistic_mean - mean_score
        worst_damage = max(
            score - baseline
            for score, baseline in zip(scores, fold_briers["logistic"], strict=True)
        )
        log_mean = float(np.mean(fold_logs.get(key, (1.0,))))
        log_damage = log_mean - logistic_log_mean
        if (
            mean_gain >= PROMOTION_MEAN_BRIER_GAIN
            and worst_damage <= MAX_SINGLE_FOLD_DAMAGE
            and log_damage <= MAX_MEAN_LOG_LOSS_DAMAGE
        ):
            eligible.append((mean_score, key))
    if not eligible:
        return (
            "logistic",
            "No challenger cleared the multi-fold Brier, log-loss, and stability gates; retained logistic.",
            False,
            fold_briers,
            fold_logs,
        )
    eligible.sort()
    champion_key = eligible[0][1]
    gain = logistic_mean - eligible[0][0]
    return (
        champion_key,
        f"Promoted {champion_key}: mean validation Brier improved by {gain:.4f} across all selection folds without a material fold regression.",
        True,
        fold_briers,
        fold_logs,
    )


def prepare_model_tournament_multifold(
    *,
    earlier_seed_games: list,
    earlier_training_games: list,
    earlier_validation_games: list,
    seed_games: list,
    training_games: list,
    validation_games: list,
    min_games: int,
    fold_years: tuple[int, ...],
) -> TournamentPreparation:
    earlier_train, earlier_validation, _, earlier_linear = _fold_rows(
        seed_games=earlier_seed_games,
        training_games=earlier_training_games,
        validation_games=earlier_validation_games,
        min_games=min_games,
    )
    final_train, final_validation, prior_summary, final_linear = _fold_rows(
        seed_games=seed_games,
        training_games=training_games,
        validation_games=validation_games,
        min_games=min_games,
    )
    selected_linear = apply_multifold_selection(
        final_linear, [earlier_linear], fold_years=fold_years
    )
    earlier_candidates = _fit_single_fold(
        earlier_train, earlier_validation, earlier_linear.champion
    )
    final_candidates = _fit_single_fold(
        final_train, final_validation, selected_linear.champion
    )
    champion_key, reason, promoted, fold_briers, fold_logs = _choose_architecture(
        [earlier_candidates, final_candidates]
    )
    artifact = ModelTournamentArtifact(
        candidates=final_candidates,
        champion_key=champion_key,
        champion_reason=reason,
        promoted_challenger=promoted,
        fold_briers=fold_briers,
        fold_log_losses=fold_logs,
        selection_folds=fold_years,
        training_rows=len(final_train),
        validation_rows=len(final_validation),
        linear_artifact=selected_linear,
    )
    return TournamentPreparation(artifact=artifact, prior_summary=prior_summary, prior_elo={})


def prepare_model_tournament(
    *,
    seed_games: list,
    training_games: list,
    validation_games: list,
    min_games: int,
) -> TournamentPreparation:
    training_rows, validation_rows, prior_summary, linear_artifact = _fold_rows(
        seed_games=seed_games,
        training_games=training_games,
        validation_games=validation_games,
        min_games=min_games,
    )
    candidates = _fit_single_fold(training_rows, validation_rows, linear_artifact.champion)
    artifact = ModelTournamentArtifact(
        candidates=candidates,
        champion_key="logistic",
        champion_reason=(
            "Only one validation fold was available, so challengers are reported but not auto-promoted."
        ),
        promoted_challenger=False,
        fold_briers={
            key: (float(candidate.validation_brier),)
            for key, candidate in candidates.items()
            if candidate.validation_brier is not None
        },
        fold_log_losses={
            key: (float(candidate.validation_log_loss),)
            for key, candidate in candidates.items()
            if candidate.validation_log_loss is not None
        },
        training_rows=len(training_rows),
        validation_rows=len(validation_rows),
        linear_artifact=linear_artifact,
    )
    return TournamentPreparation(artifact=artifact, prior_summary=prior_summary, prior_elo={})
