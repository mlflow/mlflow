_DATABRICKS_DEFAULT_JUDGE_MODEL = "databricks"
_DATABRICKS_AGENTIC_JUDGE_MODEL = "gpt-oss-120b"

# Use case constants for chat completions
USE_CASE_BUILTIN_JUDGE = "builtin_judge"
USE_CASE_AGENTIC_JUDGE = "agentic_judge"
USE_CASE_CUSTOM_PROMPT_JUDGE = "custom_prompt_judge"
USE_CASE_JUDGE_ALIGNMENT = "judge_alignment"

# Common affirmative values that should map to YES
_AFFIRMATIVE_VALUES = frozenset(
    [
        "true",
        "pass",
        "passed",
        "correct",
        "success",
        "1",
        "1.0",
        "yes",
        "y",
        "yea",
        "yeah",
        "affirmative",
        "absolutely",
        "certainly",
        "indeed",
        "sure",
        "ok",
        "okay",
        "agree",
        "accepted",
        "right",
        "positive",
        "accurate",
        "valid",
        "validity",
        "confirmed",
        "approved",
        "complete",
        "completed",
        "good",
        "great",
        "excellent",
        "active",
        "enabled",
        "on",
        "present",
        "found",
        "match",
        "matched",
        "validated",
        "approve",
        "accept",
        "pos",
    ]
)

# Common negative values that should map to NO
_NEGATIVE_VALUES = frozenset(
    [
        "false",
        "fail",
        "failed",
        "incorrect",
        "failure",
        "0",
        "0.0",
        "no",
        "n",
        "nah",
        "nope",
        "negative",
        "reject",
        "rejected",
        "disagree",
        "not approved",
        "invalid",
        "inaccurate",
        "wrong",
        "declined",
        "denied",
        "incomplete",
        "bad",
        "poor",
        "inactive",
        "disabled",
        "off",
        "missing",
        "absent",
        "notfound",
        "mismatch",
        "mismatched",
        "none",
        "null",
        "nil",
        "deny",
        "disapprove",
        "disapproved",
        "neg",
    ]
)

_RESULT_FIELD_DESCRIPTION = "The evaluation rating/result"
_RATIONALE_FIELD_DESCRIPTION = "Detailed explanation for the evaluation"
