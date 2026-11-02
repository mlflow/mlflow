import {
  INTERNAL_ASSESSMENT_ISSUE_DISCOVERY_JUDGE,
  isSessionLevelAssessment,
  type Assessment,
  type IssueReferenceAssessment,
  type ModelTraceInfoV3,
  getAssessmentValue,
  isExpectationAssessment,
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

// A session-level assessment worth a column for tables that show session feedback (the evaluation run
// view): the same rules as a trace-level one, but attached to the session through its metadata.
const isDisplayableSessionAssessment = (assessment: Assessment): boolean =>
  assessment.valid !== false &&
  isSessionLevelAssessment(assessment) &&
  !isIssueReferenceAssessment(assessment) &&
  assessment.assessment_name !== NOTES_ASSESSMENT_NAME &&
  assessment.assessment_name !== INTERNAL_ASSESSMENT_ISSUE_DISCOVERY_JUDGE;

/** Whether a session-level assessment is worth a column; `isAssessmentColumnEligible` is its trace-level twin. */
export const isSessionAssessmentColumnEligible = (assessment: Assessment): boolean =>
  isDisplayableSessionAssessment(assessment);

/** Names of the displayable session-level assessments on these traces. */
export const getSessionAssessmentNames = (traces: ModelTraceInfoV3[]): Set<string> => {
  const names = new Set<string>();
  for (const trace of traces) {
    for (const assessment of trace.assessments ?? []) {
      if (isDisplayableSessionAssessment(assessment)) {
        names.add(assessment.assessment_name);
      }
    }
  }
  return names;
};

/**
 * The session-level assessment shown for `name` across `traces` (a session's turns, or the one trace
 * it is attached to in a flat table): the most recent displayable one, or none.
 */
export const pickSessionCellAssessment = (traces: ModelTraceInfoV3[], name: string): Assessment | undefined =>
  traces
    .flatMap((trace) => trace.assessments ?? [])
    .reduce<Assessment | undefined>((best, assessment) => {
      if (assessment.assessment_name !== name || !isDisplayableSessionAssessment(assessment)) {
        return best;
      }
      return best && (best.create_time ?? '') >= (assessment.create_time ?? '') ? best : assessment;
    }, undefined);

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
  {
    seededNames = [],
    includeSessionAssessments = false,
  }: { seededNames?: readonly string[]; includeSessionAssessments?: boolean } = {},
): AssessmentColumnSelection => {
  // Seeded names are candidates even when no trace on the page carries them.
  const names = new Set<string>(seededNames);
  for (const trace of traces) {
    for (const assessment of trace.assessments ?? []) {
      if (isDisplayableTraceAssessment(assessment)) {
        names.add(assessment.assessment_name);
      }
    }
  }
  if (includeSessionAssessments) {
    getSessionAssessmentNames(traces).forEach((name) => names.add(name));
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

export type AssessmentColumnType = 'numeric' | 'categorical';

/**
 * Determine whether an assessment column should be rendered as numeric (a score with a bar)
 * or categorical (a simple tag value). Numeric if any value is not an integer; categorical otherwise.
 */
export const getAssessmentColumnType = (
  traces: ModelTraceInfoV3[],
  name: string,
  { includeSessionAssessments = false }: { includeSessionAssessments?: boolean } = {},
): AssessmentColumnType => {
  for (const trace of traces) {
    const assessment =
      pickCellAssessment(trace, name) ??
      (includeSessionAssessments ? pickSessionCellAssessment([trace], name) : undefined);
    if (assessment) {
      const value = getAssessmentValue(assessment);
      // If any value is not an integer, treat as numeric (e.g., 0.75 score).
      if (typeof value === 'number' && !Number.isInteger(value)) {
        return 'numeric';
      }
    }
  }
  return 'categorical';
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

/**
 * The source type shared by every assessment a column's cells show (judge, human or code), or undefined when
 * the sources differ or the column has no source. Session columns pick the session's assessment, like their cells.
 */
export const getAssessmentColumnSourceType = (
  traces: ModelTraceInfoV3[],
  name: string,
  { isSessionAssessment = false }: { isSessionAssessment?: boolean } = {},
): string | undefined => {
  let sourceType: string | undefined;
  for (const trace of traces) {
    const assessment = isSessionAssessment ? pickSessionCellAssessment([trace], name) : pickCellAssessment(trace, name);
    if (!assessment) {
      continue;
    }
    const current = assessment.source?.source_type;
    if (current === undefined || (sourceType !== undefined && current !== sourceType)) {
      return undefined;
    }
    sourceType = current;
  }
  return sourceType;
};

/**
 * Feedback that filters and filter suggestions can match: trace- and session-level, like the columns of the
 * evaluation run view. Its callers only run there or with the new eval runs UI on; trace cells still use
 * `isDisplayableTraceAssessment`.
 */
export const isAssessmentColumnEligible = (assessment: Assessment): boolean =>
  isDisplayableTraceAssessment(assessment) || isDisplayableSessionAssessment(assessment);

/** How an assessment's loaded values compare: as numbers (every value is a number) or as text. */
export type AssessmentValueKind = 'numeric' | 'other';

/** Key for an assessment's value kind; feedback and expectations with the same name are kept apart. */
export const assessmentValueKindKey = (field: 'feedback' | 'expectation', name: string): string => `${field}:${name}`;

/**
 * Classify the displayable assessments on `traces` by the type of their values: `numeric` when every
 * loaded value is a number, `other` as soon as one isn't (a string or a boolean). Errored or empty
 * values are skipped: they're stored as null and match no filter, so they say nothing about the type.
 */
export const classifyAssessmentValueKinds = (traces: ModelTraceInfoV3[]): Map<string, AssessmentValueKind> => {
  const kinds = new Map<string, AssessmentValueKind>();
  for (const trace of traces) {
    for (const assessment of trace.assessments ?? []) {
      if (!isDisplayableTraceAssessment(assessment)) {
        continue;
      }
      const key = assessmentValueKindKey(
        isExpectationAssessment(assessment) ? 'expectation' : 'feedback',
        assessment.assessment_name,
      );
      const value = getAssessmentValue(assessment);
      if (value === null || value === undefined) {
        continue;
      }
      const kind: AssessmentValueKind = typeof value === 'number' ? 'numeric' : 'other';
      if (kinds.get(key) !== 'other') {
        kinds.set(key, kind);
      }
    }
  }
  return kinds;
};
