"""
Matching a run's findings against the issues earlier runs already reported.

Within-run deduplication merges findings against each other, in memory, before anything is
stored. This is the other half: a finding is matched against the issues this experiment already
carries, so a repeat detection updates what is there instead of filing it again. Both use the
same test -- ``SAME_ISSUE_RULE`` -- because a finding merged into a sibling this run has to be
the same finding that matches a stored Issue next run.

Three rules the within-run pass never has to consider, because it never meets a human decision
or a previous outcome:

- A read that fails is not an empty candidate list. Publishing against "no existing issues"
  would file every finding as new, which is the failure this is most worth guarding.
- A user's triage survives. An Issue someone rejected stays rejected; detection may refresh
  when it was last seen, not overturn the judgment.
- A resolved Issue is not reopened. The same problem after a fix is a regression, which is new
  information and gets its own Issue and its own first-seen time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pydantic

from mlflow.entities.issue import Issue, IssueStatus
from mlflow.genai.discovery.constants import (
    CROSS_RUN_MATCH_PROMPT_TEMPLATE,
    DEFAULT_MODEL,
    MAX_MATCH_CANDIDATES,
    MAX_MATCH_NAME_CHARS,
    MAX_MATCH_TEXT_CHARS,
)
from mlflow.genai.discovery.entities import _IdentifiedIssue

_logger = logging.getLogger(__name__)

# Statuses a finding may be matched onto. A resolved Issue is deliberately absent: see module
# docstring. `SUPPRESSED` is not a status this port introduces.
MATCHABLE_STATUSES = (IssueStatus.PENDING.value, IssueStatus.REJECTED.value)


class CandidateReadError(Exception):
    """
    Raised when the already-reported issues could not be read.

    Publication is expected to stop rather than continue against an empty candidate list, which
    would report every finding as new and file a duplicate of everything already known.
    """


class _Match(pydantic.BaseModel):
    finding_index: int = pydantic.Field(description="Index of the new finding being decided.")
    # Required but nullable. A pydantic default keeps the field out of the schema's ``required``
    # list, and OpenAI's strict structured-output mode requires every property to be listed
    # there -- so a default here makes the request invalid rather than the field optional, the
    # matcher falls through to its error path, and every finding is published as new.
    issue_id: str | None = pydantic.Field(
        description="ID of the already-reported issue this finding is, or null when it is new.",
    )


class _Matches(pydantic.BaseModel):
    matches: list[_Match] = pydantic.Field(
        description="One decision per finding. Empty when none of them is already reported.",
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _tolerate_omitted_matches(cls, data):
        """
        Accept a response that leaves ``matches`` out, without giving it a schema default.

        Unreachable under strict mode, which obliges the model to send every property. It
        matters for a provider that ignores strict mode: "nothing matched" is the answer most
        likely to be expressed by omission, and reading that as a failure would publish a
        duplicate of every finding.
        """
        if isinstance(data, dict) and "matches" not in data:
            return {**data, "matches": []}
        return data


@dataclass
class MatchOutcome:
    """
    Which findings are already reported and which are new.

    Args:
        matched: Finding index to the ID of the issue it was matched onto.
        candidates_considered: How many stored issues the matcher was shown, for the run log.
    """

    matched: dict[int, str] = field(default_factory=dict)
    candidates_considered: int = 0

    def issue_id_for(self, finding_index: int) -> str | None:
        return self.matched.get(finding_index)

    @property
    def already_reported(self) -> int:
        return len(self.matched)


def read_match_candidates(
    experiment_id: str, max_candidates: int = MAX_MATCH_CANDIDATES
) -> list[Issue]:
    """
    The already-reported issues a finding may be matched onto, newest first.

    Raises:
        CandidateReadError: If the issues could not be read. Deliberately not an empty list --
            see the module docstring.
    """
    from mlflow.tracing.client import TracingClient

    try:
        found = TracingClient()._search_issues(
            experiment_id=experiment_id,
            statuses=list(MATCHABLE_STATUSES),
            max_results=max_candidates,
            order_by=["last_seen_timestamp DESC"],
        )
    except Exception as e:
        raise CandidateReadError(
            f"Could not read the issues already reported for experiment {experiment_id}: {e}"
        ) from e
    return list(found)


def _clip(text: str, limit: int) -> str:
    """``text`` bounded to ``limit`` characters, marked when it was cut."""
    text = (text or "").strip()
    return text if len(text) <= limit else f"{text[:limit]}..."


def _format_existing(candidates: list[Issue]) -> str:
    lines = []
    for issue in candidates:
        causes = "; ".join(issue.root_causes or []) or "unknown"
        lines.append(
            f"- id={issue.issue_id} name={_clip(issue.name, MAX_MATCH_NAME_CHARS)!r} "
            f"description={_clip(issue.description, MAX_MATCH_TEXT_CHARS)!r} "
            f"root_cause={_clip(causes, MAX_MATCH_TEXT_CHARS)!r}"
        )
    return "\n".join(lines)


def _format_findings(findings: list[_IdentifiedIssue]) -> str:
    # Bounded for the same reason as the candidates: these come from the summarization stage,
    # which has no length ceiling of its own.
    return "\n".join(
        f"[{index}] {_clip(finding.name, MAX_MATCH_NAME_CHARS)}: "
        f"{_clip(finding.description, MAX_MATCH_TEXT_CHARS)} "
        f"(root cause: {_clip(finding.root_cause, MAX_MATCH_TEXT_CHARS)})"
        for index, finding in enumerate(findings)
    )


def match_findings_to_existing_issues(
    findings: list[_IdentifiedIssue],
    candidates: list[Issue],
    model: str = DEFAULT_MODEL,
    token_counter=None,
) -> MatchOutcome:
    """
    Decide which findings are issues this experiment already carries.

    An unmatched finding, a matcher failure and an empty candidate list all mean the same thing
    to the caller -- the finding is new -- but only the last is a fact. A matcher that errors is
    logged and leaves everything unmatched, which files duplicates rather than silently folding
    a finding into the wrong issue.

    Args:
        findings: This run's identified issues, after within-run deduplication.
        candidates: Already-reported issues, from :func:`read_match_candidates`.
        model: Model URI for the matcher.
        token_counter: Optional counter, so matching's cost lands in the run's total.

    Returns:
        A :class:`MatchOutcome` mapping finding index to an existing issue ID.
    """
    outcome = MatchOutcome(candidates_considered=len(candidates))
    if not findings or not candidates:
        return outcome

    from mlflow.genai.discovery.pipeline import _call_llm

    prompt = CROSS_RUN_MATCH_PROMPT_TEMPLATE.format(
        existing_list=_format_existing(candidates),
        finding_list=_format_findings(findings),
    )
    try:
        response = _call_llm(
            model,
            [{"role": "user", "content": prompt}],
            response_format=_Matches,
            token_counter=token_counter,
        )
        decided = _Matches.model_validate_json(response.choices[0].message.content)
    except Exception:
        _logger.warning(
            "Could not match findings against already-reported issues; reporting them as new",
            exc_info=True,
        )
        return outcome

    by_id = {issue.issue_id: issue for issue in candidates}
    # Collected before any is applied, because the schema lets the matcher return several
    # decisions for one finding. Taking them in order would let the last one win, so a model that
    # contradicted itself would route the finding's evidence to whichever issue it happened to
    # name last -- and a later "no match" would not undo an earlier match at all.
    decisions: dict[int, set[str | None]] = {}
    for match in decided.matches:
        if not 0 <= match.finding_index < len(findings):
            # The matcher invented an index; the finding it meant is unknowable, so nothing is
            # matched rather than guessing which one.
            _logger.debug("Ignoring match for unknown finding index %s", match.finding_index)
            continue
        decisions.setdefault(match.finding_index, set()).add(match.issue_id)

    for finding_index, issue_ids in decisions.items():
        if len(issue_ids) > 1:
            # Disagreement is not a tie to break: the finding is reported as new, which costs a
            # duplicate issue at worst, where guessing costs this finding's evidence landing on
            # an issue it does not describe.
            _logger.info(
                "Reporting finding %s as new: the matcher returned conflicting decisions for it",
                finding_index,
            )
            continue
        (issue_id,) = issue_ids
        if issue_id is None:
            continue
        if issue_id not in by_id:
            # Only the issues it was shown are matchable: a stale or foreign ID would attach
            # this run's evidence to an issue in another experiment, or to nothing at all.
            _logger.debug("Ignoring match to unoffered issue %s", issue_id)
            continue
        outcome.matched[finding_index] = issue_id
    return outcome
