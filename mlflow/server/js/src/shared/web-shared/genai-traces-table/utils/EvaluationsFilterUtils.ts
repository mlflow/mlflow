import { isNil } from 'lodash';

import {
  getEvaluationResultAssessmentValue,
  KnownEvaluationResultAssessmentName,
} from '../components/GenAiEvaluationTracesReview.utils';
import type { AssessmentFilter, EvalTraceComparisonEntry } from '../types';

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

    const assessment =
      assessmentName === KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT
        ? runValue.overallAssessments?.[0]
        : runValue.responseAssessmentsByName[assessmentName]?.[0];

    if (filter.filterType === 'rca') {
      const currentIsAssessmentRootCause =
        runValue?.overallAssessments[0]?.rootCauseAssessment?.assessmentName === assessmentName;
      includeEval = includeEval && currentIsAssessmentRootCause;
    } else {
      let assessmentValue = assessment ? getEvaluationResultAssessmentValue(assessment) : undefined;
      if (isNil(assessmentValue)) {
        assessmentValue = undefined;
      }

      includeEval = includeEval && assessmentValue === filterValue;
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
        const currentInputsContainSearchQuery = Object.values(entry.currentRunValue?.inputs || {}).some(
          (inputValue) => {
            return JSON.stringify(inputValue).toLowerCase().includes(searchQueryLower);
          },
        );
        const inputsIdEqualsToSearchQuery = entry.currentRunValue?.inputsId.toLowerCase() === searchQueryLower;
        return currentInputsContainSearchQuery || inputsIdEqualsToSearchQuery;
      })
  );
}
