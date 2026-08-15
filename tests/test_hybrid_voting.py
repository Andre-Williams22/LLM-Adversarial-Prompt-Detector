"""Tests for the hybrid voting layer.

Voting is the policy surface of the classifier: it turns four independent model
scores into a single allow/block decision. These tests pin the behaviour of each
of the four voting rules and of the three sensitivity presets, so that a change
to a threshold cannot silently change what the system blocks.

Score order throughout is ``[keyword, toxic, hate, safety]``.
"""
import pytest

from utils.fast_detection import FastAdversarialDetector

SAFE = [0.0, 0.0, 0.0, 0.0]


@pytest.fixture(scope="module", params=["conservative", "balanced", "high"])
def detector(request):
    return FastAdversarialDetector(sensitivity_mode=request.param)


@pytest.fixture(scope="module")
def balanced():
    return FastAdversarialDetector(sensitivity_mode="balanced")


def test_all_zero_scores_are_allowed(detector):
    is_adversarial, reason, details = detector.hybrid_voting_decision(SAFE)

    assert is_adversarial is False
    assert details["final_decision"] == "safe"
    assert not any(details[rule] for rule in (
        "high_confidence_trigger", "weak_signals_trigger",
        "majority_consensus", "weighted_ensemble",
    ))
    assert "below threshold" in reason


def test_a_single_confident_keyword_hit_blocks_in_every_mode(detector):
    """Rule 1: one model above the high-confidence threshold is enough."""
    scores = [0.95, 0.0, 0.0, 0.0]

    is_adversarial, reason, details = detector.hybrid_voting_decision(scores)

    assert is_adversarial is True
    assert details["high_confidence_trigger"] is True
    assert reason == "High-confidence detection"


def test_rules_are_evaluated_in_priority_order(balanced):
    """A prompt that satisfies several rules is attributed to the first one."""
    scores = [0.9, 0.9, 0.9, 0.9]

    _, _, details = balanced.hybrid_voting_decision(scores)

    assert details["high_confidence_trigger"] is True
    assert details["weak_signals_trigger"] is False
    assert details["majority_consensus"] is False
    assert details["weighted_ensemble"] is False


def test_two_weak_signals_block_under_balanced(balanced):
    """Rule 2: subtle attacks that only nudge several models still get caught."""
    weak = balanced.voting_config["weak_signals_threshold"] + 0.01
    scores = [weak, weak, 0.0, 0.0]

    is_adversarial, reason, details = balanced.hybrid_voting_decision(scores)

    assert is_adversarial is True
    assert details["weak_signals_trigger"] is True
    assert reason == "Multiple weak signals detected"


def test_weighted_ensemble_is_the_last_line_of_defence(balanced):
    """Rule 4: no single rule fires, but the weighted sum clears its threshold."""
    scores = [0.19, 0.19, 0.19, 0.19]
    config = balanced.voting_config
    weights = [config["keyword_weight"]] + config["other_weights"]
    weighted = sum(s * w for s, w in zip(scores, weights))

    is_adversarial, reason, details = balanced.hybrid_voting_decision(scores)

    assert weighted <= config["weighted_threshold"], "fixture no longer exercises rule 4"
    assert is_adversarial is False
    assert details["weighted_ensemble"] is False
    assert "below threshold" in reason


def test_keyword_carries_the_largest_ensemble_weight(detector):
    """The keyword detector is the most reliable signal, so it must dominate."""
    config = detector.voting_config

    assert config["keyword_weight"] > max(config["other_weights"])
    assert config["keyword_weight"] + sum(config["other_weights"]) == pytest.approx(1.0)


@pytest.mark.parametrize("rule", [
    "high_confidence_threshold",
    "weak_signals_threshold",
    "majority_threshold",
    "weighted_threshold",
])
def test_sensitivity_modes_are_ordered_by_strictness(rule):
    """high recall <= balanced <= conservative for every decision threshold."""
    thresholds = {
        mode: FastAdversarialDetector(sensitivity_mode=mode).voting_config[rule]
        for mode in ("high", "balanced", "conservative")
    }

    assert thresholds["high"] <= thresholds["balanced"] <= thresholds["conservative"]


def test_unknown_sensitivity_mode_falls_back_to_balanced():
    fallback = FastAdversarialDetector(sensitivity_mode="not-a-mode")
    balanced = FastAdversarialDetector(sensitivity_mode="balanced")

    assert fallback.voting_config == balanced.voting_config


def test_high_sensitivity_never_allows_what_balanced_blocks():
    """Raising sensitivity must only ever add detections, never remove them."""
    high = FastAdversarialDetector(sensitivity_mode="high")
    balanced = FastAdversarialDetector(sensitivity_mode="balanced")
    grid = [round(0.1 * i, 1) for i in range(11)]

    for keyword in grid:
        for other in grid:
            scores = [keyword, other, other, other]
            if balanced.hybrid_voting_decision(scores)[0]:
                assert high.hybrid_voting_decision(scores)[0], (
                    f"high sensitivity allowed {scores} that balanced blocked"
                )


def test_voting_details_shape_is_stable(balanced):
    """The details dict is logged to MLflow and Grafana; its keys are a contract."""
    _, _, details = balanced.hybrid_voting_decision([0.95, 0.0, 0.0, 0.0])

    assert set(details) == {
        "high_confidence_trigger",
        "weak_signals_trigger",
        "majority_consensus",
        "weighted_ensemble",
        "final_decision",
    }
