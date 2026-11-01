
import re
from typing import Any


SEVERITY_RULES = {
    "critical": [
        "data breach",
        "ransomware",
        "confirmed unauthorized access",
        "data loss",
        "complete outage",
        "entire production system is down",
        "all users are unable to access",
    ],
    "high": [
        "payment processing is down",
        "orders cannot be processed",
        "business operations have stopped",
        "multiple departments are affected",
        "no workaround is available",
        "all staff are affected",
    ],
    "medium": [
        "workflow is blocked",
        "cannot approve",
        "operations are delayed",
        "repeatedly fails",
        "intermittent failure",
        "team cannot access",
    ],
    "low": [
        "cosmetic issue",
        "minor typo",
        "visual alignment issue",
        "minor formatting issue",
    ],
}


def assess_severity(text: str) -> dict[str, Any]:
    normalized_text = re.sub(r"\s+", " ", text.casefold()).strip()

    if not normalized_text:
        raise ValueError("Ticket text cannot be empty.")

    for severity in ("critical", "high", "medium", "low"):
        for phrase in SEVERITY_RULES[severity]:
            if phrase in normalized_text:
                return {
                    "severity": severity,
                    "reason": f"impact_signal:{phrase.replace(' ', '_')}",
                    "needs_more_information": False,
                }

    return {
        "severity": "undetermined",
        "reason": "insufficient_impact_details",
        "needs_more_information": True,
    }
