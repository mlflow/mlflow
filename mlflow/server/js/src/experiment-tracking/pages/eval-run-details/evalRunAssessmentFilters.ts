import {
  getAssessmentValue,
  isExpectationAssessment,
  type Assessment,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import {
  FilterOp,
  isClauseComplete,
  type FilterClause,
  type TraceFilterModel,
} from '@databricks/web-shared/traces-table';
import { isAssessmentColumnEligible } from '../../components/experiment-page/components/traces-v4/utils/assessmentColumns';

const NUMERIC_VALUE = /^-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i;

const isAssessmentClause = (clause: FilterClause): boolean =>
  (clause.field === 'assessment' || clause.field === 'expectation') && isClauseComplete(clause);

/**
 * Splits the run page's filters into the ones the server can apply and the feedback and expectation
 * clauses. The server matches those against any assessment on a trace, including ones other runs
 * logged on a linked trace, so the page matches them itself after scoping the traces to the run.
 */
export const splitEvalRunAssessmentFilters = (
  filterModel: TraceFilterModel,
): { serverFilterModel: TraceFilterModel; assessmentFilters: TraceFilterModel } => ({
  serverFilterModel: filterModel.filter((clause) => !isAssessmentClause(clause)),
  assessmentFilters: filterModel.filter(isAssessmentClause),
});

const valueMatches = (value: ReturnType<typeof getAssessmentValue>, filterValue: string): boolean => {
  const trimmed = filterValue.trim();
  const numericValue = typeof value === 'number' ? value : typeof value === 'string' ? value.trim() : undefined;
  if (numericValue !== undefined && NUMERIC_VALUE.test(String(numericValue)) && NUMERIC_VALUE.test(trimmed)) {
    // Whole numbers come back as `4.0` from the server, so compare numbers as numbers.
    return Number(numericValue) === Number(trimmed);
  }
  return value?.toString() === filterValue;
};

const latestAssessment = (trace: ModelTraceInfoV3, clause: FilterClause): Assessment | undefined =>
  (trace.assessments ?? [])
    .filter(
      (assessment) =>
        isAssessmentColumnEligible(assessment) &&
        assessment.assessment_name === clause.key &&
        isExpectationAssessment(assessment) === (clause.field === 'expectation'),
    )
    .reduce<
      Assessment | undefined
    >((latest, assessment) => (latest && (latest.create_time ?? '') >= (assessment.create_time ?? '') ? latest : assessment), undefined);

/** Whether a run-scoped trace matches one feedback or expectation clause, on its latest such assessment. */
export const matchesEvalRunAssessmentFilter = (trace: ModelTraceInfoV3, clause: FilterClause): boolean => {
  const assessment = latestAssessment(trace, clause);
  const value = assessment ? getAssessmentValue(assessment) : undefined;
  switch (clause.operator) {
    case FilterOp.EQUALS:
      return valueMatches(value, clause.value);
    case FilterOp.NOT_EQUALS:
      return value !== null && value !== undefined && !valueMatches(value, clause.value);
    default:
      return true;
  }
};
