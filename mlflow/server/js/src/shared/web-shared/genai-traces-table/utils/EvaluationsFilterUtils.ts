import { isNil } from 'lodash';

import {
  getEvaluationResultAssessmentValue,
  KnownEvaluationResultAssessmentName,
} from '../components/GenAiEvaluationTracesReview.utils';
import type { AssessmentFilter, EvalTraceComparisonEntry } from '../types';
import { ERROR_KEY } from './AggregationUtils';

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
        ? runValue.overallAssessments ?? []
        : runValue.responseAssessmentsByName[assessmentName] ?? [];

    if (filter.filterType === 'rca') {
      const currentIsAssessmentRootCause =
        runValue?.overallAssessments[0]?.rootCauseAssessment?.assessmentName === assessmentName;
      includeEval = includeEval && currentIsAssessmentRootCause;
    } else {
      // Filtering for undefined means we want traces with NO assessments for this name
      // Filtering for ERROR_KEY means we want traces with assessments that have an errorMessage
      const matchesFilter =
        filterValue === undefined
          ? assessments.length === 0
          : filterValue === ERROR_KEY
          ? assessments.some((assessment) => Boolean(assessment.errorMessage))
          : assessments.some(
              (assessment) => (getEvaluationResultAssessmentValue(assessment) ?? undefined) === filterValue,
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
        return currentInputsContainSearchQuery || inputsIdEqualsToSearchQuery;
      })
  );
}
