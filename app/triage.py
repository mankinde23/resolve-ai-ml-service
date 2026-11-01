
from typing import Any


def decide_triage(
    prediction: dict[str, Any],
    confidence_threshold: float,
    severity_result: dict[str, Any],
) -> dict[str, Any]:

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")

    category = str(prediction["predicted_category"]).strip()
    confidence = prediction.get("confidence")
    severity = severity_result["severity"]

    if not category:
        raise ValueError("Predicted category cannot be empty.")

    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1.")

    if severity not in {"critical", "high", "medium", "low", "undetermined"}:
        raise ValueError("Invalid severity value.")

    # Critical incidents always require escalation.
    if severity == "critical":
        action = "escalate"
        priority = "high"
        reason = "critical_severity_requires_escalation"

    # Preserve the existing security-category safeguard.
    elif category.casefold() == "security":
        action = "escalate"
        priority = "high"
        reason = "security_category_requires_review"

    # High-severity incidents require human attention.
    elif severity == "high":
        action = "human_review"
        priority = "high"
        reason = "high_severity_requires_review"

    # Insufficient impact information requires investigation.
    elif (
        severity == "undetermined"
        or severity_result.get("needs_more_information", False)
    ):
        action = "human_review"
        priority = "normal"
        reason = "impact_information_required"

    # Classification uncertainty requires human review.
    elif confidence is None:
        action = "human_review"
        priority = "normal"
        reason = "confidence_unavailable"

    elif confidence < confidence_threshold:
        action = "human_review"
        priority = "normal"
        reason = "confidence_below_threshold"

    else:
        action = "auto_route"
        priority = "normal"
        reason = "prediction_meets_provisional_threshold"

    return {
        "predicted_category": category,
        "confidence": confidence,
        "action": action,
        "priority": priority,
        "requires_human_review": action != "auto_route",
        "reason": reason,
    }
