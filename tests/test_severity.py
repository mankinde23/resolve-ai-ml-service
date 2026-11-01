
import pytest

from app.severity import assess_severity


def test_detects_critical_security_incident():
    result = assess_severity(
        "We discovered a data breach affecting customer records."
    )

    assert result["severity"] == "critical"
    assert result["needs_more_information"] is False


def test_detects_high_business_impact():
    result = assess_severity(
        "Payment processing is down and customers cannot complete purchases."
    )

    assert result["severity"] == "high"


def test_detects_medium_workflow_disruption():
    result = assess_severity(
        "The approval workflow is blocked and operations are delayed."
    )

    assert result["severity"] == "medium"


def test_detects_low_cosmetic_issue():
    result = assess_severity(
        "There is a minor typo on the dashboard."
    )

    assert result["severity"] == "low"


def test_returns_undetermined_without_impact_evidence():
    result = assess_severity(
        "The procurement dashboard displays an unexpected message."
    )

    assert result["severity"] == "undetermined"
    assert result["needs_more_information"] is True


def test_rejects_empty_ticket():
    with pytest.raises(ValueError):
        assess_severity("   ")
