import { isNil } from 'lodash';

import { ERROR_KEY } from './AggregationUtils';
import {
  getEvaluationResultAssessmentValue,
  KnownEvaluationResultAssessmentName,
} from '../components/GenAiEvaluationTracesReview.utils';
import type { AssessmentFilter, AssessmentValueType, EvalTraceComparisonEntry } from '../types';
import { FilterOperator } from '../types';

/**
 * Compares an assessment value against a filter value using the specified operator.
 * For numeric comparison operators (>, <, >=, <=), both values must be numbers.
 * For equality operators (=, !=), uses strict equality.
 */
function compareValues(
  assessmentValue: AssessmentValueType,
  filterValue: AssessmentValueType,
  operator?: FilterOperator,
): boolean {
  const stringAssessmentValue = String(assessmentValue);
  const stringFilterValue = String(filterValue);
  // Default to equality if no operator specified
  if (!operator || operator === FilterOperator.EQUALS) {
    return stringAssessmentValue === stringFilterValue;
  }

  if (operator === FilterOperator.NOT_EQUALS) {
    return stringAssessmentValue !== stringFilterValue;
  }

  // For numeric comparison operators, both values must be numbers
  const numAssessmentValue = Number(assessmentValue);
  const numFilterValue = Number(filterValue);
  if (isNaN(numAssessmentValue) || isNaN(numFilterValue)) {
    return false;
  }

  switch (operator) {
    case FilterOperator.GREATER_THAN:
      return numAssessmentValue > numFilterValue;
    case FilterOperator.LESS_THAN:
      return numAssessmentValue < numFilterValue;
    case FilterOperator.GREATER_THAN_OR_EQUALS:
      return numAssessmentValue >= numFilterValue;
    case FilterOperator.LESS_THAN_OR_EQUALS:
      return numAssessmentValue <= numFilterValue;
    default:
      return stringAssessmentValue === stringFilterValue;
  }
}

function filterEval(
  comparisonEntry: EvalTraceComparisonEntry,
  filters: AssessmentFilter[],
  currentRunDisplayName?: string,
  otherRunDisplayName?: string,
): boolean {
  // Currently only filters on the current run value.
  const currentRunValue = comparisonEntry?.currentRunValue;
  const otherRunValue = comparisonEntry?.otherRunValue;

  let includeEval = true;

  for (const filter of filters) {
    const assessmentName = filter.assessmentName;
    const filterValue = filter.filterValue;
    const run = filter.run;
    const runValue =
      run === currentRunDisplayName ? currentRunValue : run === otherRunDisplayName ? otherRunValue : undefined;
    // TODO(nsthorat): Fix this logic, not clear that this is the right way to filter.
    if (runValue === undefined) {
      continue;
    }

    const assessments =
      assessmentName === KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT
        ? (runValue.overallAssessments ?? [])
        : (runValue.responseAssessmentsByName[assessmentName] ?? []);

    if (filter.filterType === 'rca') {
      const currentIsAssessmentRootCause =
        runValue?.overallAssessments[0]?.rootCauseAssessment?.assessmentName === assessmentName;
      includeEval = includeEval && currentIsAssessmentRootCause;
    } else {
      // Filtering for undefined means we want traces with NO assessments (= undefined)
      // or WITH assessments (!= undefined) for this name
      // Filtering for ERROR_KEY means we want traces with assessments that have an errorMessage
      const matchesFilter =
        filterValue === undefined
          ? filter.filterOperator === FilterOperator.NOT_EQUALS
            ? assessments.length > 0
            : assessments.length === 0
          : filterValue === ERROR_KEY
            ? assessments.some((assessment) => Boolean(assessment.errorMessage))
            : assessments.some((assessment) =>
                compareValues(getEvaluationResultAssessmentValue(assessment), filterValue, filter.filterOperator),
              );
      includeEval = includeEval && matchesFilter;
    }
  }
  return includeEval;
}

export function filterEvaluationResults(
  evaluationResults: EvalTraceComparisonEntry[],
  assessmentFilters: AssessmentFilter[],
  searchQuery?: string,
  currentRunDisplayName?: string,
  otherRunDisplayName?: string,
): EvalTraceComparisonEntry[] {
  // Filter results by the assessment filters.
  return (
    evaluationResults
      .filter((entry) => {
        return filterEval(entry, assessmentFilters, currentRunDisplayName, otherRunDisplayName);
      })
      // Filter results by the text search box.
      .filter((entry) => {
        if (isNil(searchQuery) || searchQuery === '') {
          return true;
        }
        const searchQueryLower = searchQuery.toLowerCase();
        const currentInputsContainSearchQuery = JSON.stringify(entry.currentRunValue?.inputs)
          .toLowerCase()
          .includes(searchQueryLower);
        const inputsIdEqualsToSearchQuery = entry.currentRunValue?.inputsId.toLowerCase() === searchQueryLower;
        // Also match against trace IDs: both the short backend trace_id (evaluationId)
        // and the full V4 trace ID (fullTraceId) like trace:/catalog.schema/abc123...
        const evaluationIdContainsSearchQuery =
          entry.currentRunValue?.evaluationId?.toLowerCase().includes(searchQueryLower) ?? false;
        const fullTraceIdContainsSearchQuery =
          entry.currentRunValue?.fullTraceId?.toLowerCase().includes(searchQueryLower) ?? false;
        return (
          currentInputsContainSearchQuery ||
          inputsIdEqualsToSearchQuery ||
          evaluationIdContainsSearchQuery ||
          fullTraceIdContainsSearchQuery
        );
      })
  );
}
