"""Rule-based learning recommendations tied to weak academic/engagement signals."""

from __future__ import annotations

from typing import Any

RESOURCE_MAP = {
    "failures": {
        "gap": "Prior course failures",
        "action": "Join a small-group remedial session and complete a weekly concept checklist with a tutor.",
        "resource": "Remedial tutoring + spaced-repetition practice set",
    },
    "absences": {
        "gap": "High absence count",
        "action": "Set an attendance plan with a mentor and catch up missed lessons through recorded walkthroughs.",
        "resource": "Attendance mentoring + recorded lesson pack",
    },
    "attendance_proxy": {
        "gap": "Low attendance proxy",
        "action": "Protect two fixed study-and-class blocks each weekday.",
        "resource": "Weekly attendance contract",
    },
    "studytime": {
        "gap": "Limited weekly study time",
        "action": "Add three 40-minute focused study blocks and log what was practised.",
        "resource": "Study-skills workshop + Pomodoro planner",
    },
    "engagement_score": {
        "gap": "Low academic engagement",
        "action": "Pick one extra activity or peer study group this term and keep a short weekly reflection.",
        "resource": "Peer study circle",
    },
    "higher": {
        "gap": "Unclear higher-education goal",
        "action": "Book a career/academic counselling slot to map courses to a concrete next step.",
        "resource": "Career counselling session",
    },
    "internet": {
        "gap": "Limited home internet access",
        "action": "Reserve school lab or library hours twice a week for online practice.",
        "resource": "On-campus digital access slot",
    },
    "parent_edu_avg": {
        "gap": "Limited at-home academic support background",
        "action": "Increase school-based academic support rather than assuming homework help at home.",
        "resource": "After-school supervised study hall",
    },
    "alcohol_index": {
        "gap": "Higher weekday/weekend alcohol use",
        "action": "Share wellbeing options; academic plans should include sleep and health support.",
        "resource": "Student wellbeing / counselling referral",
    },
    "goout": {
        "gap": "High social time relative to study",
        "action": "Keep social plans, but move them after a completed study block.",
        "resource": "Time-blocking template",
    },
    "traveltime": {
        "gap": "Long commute",
        "action": "Use commute audio reviews and keep a compact flash-card pack.",
        "resource": "Mobile revision pack",
    },
    "schoolsup": {
        "gap": "Already flagged for extra school support",
        "action": "Keep extra support but add subject-specific practice, not only general help.",
        "resource": "Subject specialist hours",
    },
    "subject": {
        "gap": "Subject-specific difficulty",
        "action": "Use topic diagnostics: algebra/functions for math, reading/writing drills for Portuguese.",
        "resource": "Subject diagnostic quiz + targeted worksheet",
    },
    "health": {
        "gap": "Lower self-reported health",
        "action": "Coordinate academic load with health support; do not treat this as a motivation issue.",
        "resource": "Health/wellbeing referral",
    },
}


def recommend_for_student(payload: dict[str, Any], shap_or_deltas: list[dict] | None = None) -> list[dict]:
    recs: list[dict] = []

    def add(key: str, evidence: str) -> None:
        item = RESOURCE_MAP[key].copy()
        item["evidence"] = evidence
        recs.append(item)

    if int(payload.get("failures", 0)) >= 1:
        add("failures", f"Past failures = {payload['failures']}")
    if int(payload.get("absences", 0)) >= 8:
        add("absences", f"Absences = {payload['absences']}")
    if int(payload.get("studytime", 2)) <= 1:
        add("studytime", "Weekly study time is in the lowest band (<2 hours).")
    if str(payload.get("higher", "yes")).lower() == "no":
        add("higher", "Student does not currently plan higher education.")
    if str(payload.get("internet", "yes")).lower() == "no":
        add("internet", "No home internet access recorded.")
    if int(payload.get("goout", 3)) >= 4 and int(payload.get("studytime", 2)) <= 2:
        add("goout", "Going-out score is high while study time is modest.")
    if (int(payload.get("Dalc", 1)) + int(payload.get("Walc", 1))) / 2 >= 3:
        add("alcohol_index", "Alcohol index is at or above the mid scale.")
    if int(payload.get("traveltime", 1)) >= 3:
        add("traveltime", "Commute is 30+ minutes.")
    if str(payload.get("schoolsup", "no")).lower() == "yes":
        add("schoolsup", "Already receiving extra educational support.")
    if int(payload.get("health", 3)) <= 2:
        add("health", "Self-reported health is low.")
    if payload.get("subject") == "math":
        add("subject", "Current course is mathematics.")
    elif payload.get("subject") == "portuguese":
        add("subject", "Current course is Portuguese.")

    if shap_or_deltas:
        top_neg = [d for d in shap_or_deltas if d.get("delta_if_typical", d.get("shap", 0)) < 0][:3]
        for item in top_neg:
            recs.append(
                {
                    "gap": f"Model-sensitive factor: {item.get('feature')}",
                    "action": "Review this factor with a teacher; it pulled the predicted score down versus a typical student.",
                    "resource": "Teacher conference using the explanation panel",
                    "evidence": str(item),
                }
            )

    if not recs:
        recs.append(
            {
                "gap": "No strong risk flags from rules",
                "action": "Keep current study habits and use stretch resources in the weaker subject topics.",
                "resource": "Optional enrichment pack",
                "evidence": "Rule engine found no high-priority gaps.",
            }
        )
    # de-duplicate by gap
    seen = set()
    unique = []
    for rec in recs:
        if rec["gap"] in seen:
            continue
        seen.add(rec["gap"])
        unique.append(rec)
    return unique[:6]
