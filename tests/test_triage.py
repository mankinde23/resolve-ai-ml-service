
import pytest

from app.triage import decide_triage


THRESHOLD = 0.75


def make_severity(value, needs_more_information=False):
    return {
        "severity": value,
        "needs_more_information": needs_more_information,
    }


def make_prediction(category="Bug", confidence=0.90):
    return {
        "predicted_category": category,
        "confidence": confidence,
    }


def test_critical_severity_escalates():
    result = decide_triage(
        make_prediction(confidence=0.30),
        THRESHOLD,
        make_severity("critical"),
    )

    assert result["action"] == "escalate"
    assert result["priority"] == "high"
    assert result["requires_human_review"] is True


def test_high_severity_requires_review():
    result = decide_triage(
        make_prediction(confidence=0.95),
        THRESHOLD,
        make_severity("high"),
    )

    assert result["action"] == "human_review"
    assert result["priority"] == "high"


def test_undetermined_severity_requires_review():
    result = decide_triage(
        make_prediction(confidence=0.95),
        THRESHOLD,
        make_severity("undetermined", True),
    )

    assert result["action"] == "human_review"
    assert result["reason"] == "impact_information_required"


def test_low_severity_and_high_confidence_auto_route():
    result = decide_triage(
        make_prediction(confidence=0.95),
        THRESHOLD,
        make_severity("low"),
    )

    assert result["action"] == "auto_route"
    assert result["priority"] == "normal"


def test_low_severity_and_low_confidence_requires_review():
    result = decide_triage(
        make_prediction(confidence=0.20),
        THRESHOLD,
        make_severity("low"),
    )

    assert result["action"] == "human_review"
    assert result["reason"] == "confidence_below_threshold"


def test_security_category_escalates():
    result = decide_triage(
        make_prediction(category="Security", confidence=0.30),
        THRESHOLD,
        make_severity("low"),
    )

    assert result["action"] == "escalate"
    assert result["priority"] == "high"


def test_missing_confidence_requires_review():
    result = decide_triage(
        make_prediction(confidence=None),
        THRESHOLD,
        make_severity("medium"),
    )

    assert result["action"] == "human_review"
    assert result["reason"] == "confidence_unavailable"


def test_invalid_threshold_raises_error():
    with pytest.raises(ValueError):
        decide_triage(
            make_prediction(),
            1.5,
            make_severity("low"),
        )


def test_invalid_confidence_raises_error():
    with pytest.raises(ValueError):
        decide_triage(
            make_prediction(confidence=1.5),
            THRESHOLD,
            make_severity("low"),
        )
