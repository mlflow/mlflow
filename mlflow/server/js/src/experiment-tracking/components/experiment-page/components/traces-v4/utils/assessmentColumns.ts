import {
  INTERNAL_ASSESSMENT_ISSUE_DISCOVERY_JUDGE,
  isSessionLevelAssessment,
  type Assessment,
  type ModelTraceInfoV3,
  getAssessmentValue,
} from '@databricks/web-shared/model-trace-explorer';
// Not re-exported from the OSS barrel — import from its module.
import { NOTES_ASSESSMENT_NAME } from '@databricks/web-shared/model-trace-explorer/assessments-pane/AssessmentsPaneNotesSection';

// Extra-column ids are namespaced so an assessment named e.g. `state` can't collide with a standard
// TraceColumnId in the shared sizing/selection stores.
export const ASSESSMENT_COLUMN_ID_PREFIX = 'assessment:';
export const assessmentColumnId = (name: string): string => `${ASSESSMENT_COLUMN_ID_PREFIX}${name}`;
export const isAssessmentColumnId = (id: string): boolean => id.startsWith(ASSESSMENT_COLUMN_ID_PREFIX);
export const assessmentNameFromColumnId = (id: string): string => id.slice(ASSESSMENT_COLUMN_ID_PREFIX.length);

// A trace-level assessment worth showing: valid, not session-level, and not an internal name the
// prior tab also hides.
const isDisplayableTraceAssessment = (assessment: Assessment): boolean =>
  assessment.valid !== false &&
  !isSessionLevelAssessment(assessment) &&
  assessment.assessment_name !== NOTES_ASSESSMENT_NAME &&
  assessment.assessment_name !== INTERNAL_ASSESSMENT_ISSUE_DISCOVERY_JUDGE;

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

export type AssessmentColumnType = 'numeric' | 'categorical';

/**
 * Determine whether an assessment column should be rendered as numeric (a score with a bar)
 * or categorical (a simple tag value). Numeric if any value is not an integer; categorical otherwise.
 */
export const getAssessmentColumnType = (traces: ModelTraceInfoV3[], name: string): AssessmentColumnType => {
  for (const trace of traces) {
    const assessment = pickCellAssessment(trace, name);
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
