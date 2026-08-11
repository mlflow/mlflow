import {
  INTERNAL_ASSESSMENT_ISSUE_DISCOVERY_JUDGE,
  isSessionLevelAssessment,
  type Assessment,
  type IssueReferenceAssessment,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
// Not re-exported from the OSS barrel — import from its module.
import { NOTES_ASSESSMENT_NAME } from '@databricks/web-shared/model-trace-explorer/assessments-pane/AssessmentsPaneNotesSection';
import type { Issue } from '@databricks/web-shared/genai-traces-table/cellRenderers/IssuesCell';

// Extra-column ids are namespaced so an assessment named e.g. `state` can't collide with a standard
// TraceColumnId in the shared sizing/selection stores.
export const ASSESSMENT_COLUMN_ID_PREFIX = 'assessment:';
export const assessmentColumnId = (name: string): string => `${ASSESSMENT_COLUMN_ID_PREFIX}${name}`;
export const isAssessmentColumnId = (id: string): boolean => id.startsWith(ASSESSMENT_COLUMN_ID_PREFIX);
export const assessmentNameFromColumnId = (id: string): string => id.slice(ASSESSMENT_COLUMN_ID_PREFIX.length);

// An issue-reference assessment is a detected issue, not a feedback/expectation. The prior tab
// surfaces these in a dedicated Issues column (see `extractTraceIssues`), so they must be kept out
// of the assessment columns and the assessment filter suggestions.
const isIssueReferenceAssessment = (assessment: Assessment): assessment is IssueReferenceAssessment =>
  Boolean('issue' in assessment && assessment.issue);

// A trace-level assessment worth showing as an assessment column: valid, not session-level, not an
// internal name the prior tab also hides, and not an issue reference (those get their own column).
const isDisplayableTraceAssessment = (assessment: Assessment): boolean =>
  assessment.valid !== false &&
  !isSessionLevelAssessment(assessment) &&
  !isIssueReferenceAssessment(assessment) &&
  assessment.assessment_name !== NOTES_ASSESSMENT_NAME &&
  assessment.assessment_name !== INTERNAL_ASSESSMENT_ISSUE_DISCOVERY_JUDGE;

/**
 * Extract a trace's detected issues for the dedicated Issues column, mirroring the prior tab
 * (`TraceUtils.convertTraceInfoV3ToRunEvalEntry`): the assessment name is the issue id, and
 * `issue.issue_name` is the display label (falling back to the id).
 */
export const extractTraceIssues = (trace: ModelTraceInfoV3): Issue[] =>
  (trace.assessments ?? [])
    // Skip invalid assessments before collecting issues, matching the prior tab's conversion path
    // (`TraceUtils.convertTraceInfoV3ToRunEvalEntry`), which drops `valid === false` upfront.
    .filter(
      (assessment): assessment is IssueReferenceAssessment =>
        assessment.valid !== false && isIssueReferenceAssessment(assessment),
    )
    .map((assessment) => ({
      id: assessment.assessment_name,
      name: assessment.issue.issue_name || assessment.assessment_name,
    }));

export interface AssessmentColumnSelection {
  /** Names offered in the column selector (on-page ∪ opted-in), sorted for stable order. */
  candidateNames: string[];
  /** Names whose columns render, sorted; a subset of `candidateNames`. */
  visibleNames: string[];
}

/**
 * Derive the assessment columns for the current page. A column is a candidate when its assessment
 * appears on the page OR the user opted into it (so a never-before-seen assessment surfaces the
 * first page it lands on, and an opted-in one stays visible even on pages that lack it). Default
 * visibility is on; an explicit `false` override hides it, an explicit `true` keeps it.
 */
export const computeAssessmentColumns = (
  traces: ModelTraceInfoV3[],
  overrides: Record<string, boolean>,
): AssessmentColumnSelection => {
  const names = new Set<string>();
  for (const trace of traces) {
    for (const assessment of trace.assessments ?? []) {
      if (isDisplayableTraceAssessment(assessment)) {
        names.add(assessment.assessment_name);
      }
    }
  }
  for (const [name, on] of Object.entries(overrides)) {
    if (on) {
      names.add(name);
    }
  }
  const candidateNames = [...names].sort();
  const visibleNames = candidateNames.filter((name) => overrides[name] ?? true);
  return { candidateNames, visibleNames };
};

/** The assessment shown in a cell for `name`: the most recent displayable one, or none. */
export const pickCellAssessment = (trace: ModelTraceInfoV3, name: string): Assessment | undefined =>
  // Single pass keeping the max by `create_time` (first-encountered wins a tie, matching a
  // stable descending sort + head); avoids sorting the whole list per cell per render.
  (trace.assessments ?? []).reduce<Assessment | undefined>((best, assessment) => {
    if (assessment.assessment_name !== name || !isDisplayableTraceAssessment(assessment)) {
      return best;
    }
    return best && (best.create_time ?? '') >= (assessment.create_time ?? '') ? best : assessment;
  }, undefined);
