import { first, isNil } from 'lodash';

import type { ThemeType } from '@databricks/design-system';
import type { IntlShape } from '@databricks/i18n';

import { getAssessmentValueBarBackgroundColor } from './Colors';
import { isAssessmentPassing } from '../components/EvaluationsReviewAssessmentTag';
import {
  ASSESSMENTS_DOC_LINKS,
  DEFAULT_ASSESSMENTS_SORT_ORDER,
  getEvaluationResultAssessmentValue,
  getJudgeMetricsLink,
  KnownEvaluationResultAssessmentName,
  KnownEvaluationResultAssessmentStringValue,
  KnownEvaluationResultAssessmentValueDescription,
  KnownEvaluationResultAssessmentValueLabel,
  KnownEvaluationResultAssessmentValueMissingTooltip,
} from '../components/GenAiEvaluationTracesReview.utils';
import type {
  AssessmentAggregates,
  AssessmentRunCounts,
  AssessmentInfo,
  EvalTraceComparisonEntry,
  RunEvaluationResultAssessment,
  RunEvaluationTracesDataEntry,
  AssessmentDType,
  AssessmentFilter,
  AssessmentValueType,
  NumericAggregateCount,
  NumericAggregate,
} from '../types';

export interface StackedRunBarchartItem {
  value: number;
  fraction: number;
  isSelected: boolean;
  toggleFilter?: () => void;
  tooltip: string;
}
export interface StackedBarchartItem {
  name: string;
  current: StackedRunBarchartItem;
  other?: StackedRunBarchartItem;
  backgroundColor: string;
  scoreChange?: number;
}

export const ERROR_KEY = 'Error';

export function doesAssessmentContainErrors(assessment?: RunEvaluationResultAssessment): boolean {
  return Boolean(assessment?.errorCode || assessment?.errorMessage);
}

function getCustomMetricNameAndAssessment(assessmentPath: string): { metricName: string; assessmentName: string } {
  // metric/all_guidelines/guideline_adherence
  // gets parsed to {metricName: 'all_guidelines', assessmentName: 'guideline_adherence'}
  const splits = assessmentPath.split('/');
  if (splits.length === 1) {
    return { metricName: assessmentPath, assessmentName: assessmentPath };
  } else if (splits.length === 2) {
    return { metricName: splits[0], assessmentName: splits[1] };
  } else {
    return { metricName: splits[1], assessmentName: splits.slice(2).join('/') };
  }
}

const PASS_FAIL_VALUES: string[] = [
  KnownEvaluationResultAssessmentStringValue.YES,
  KnownEvaluationResultAssessmentStringValue.NO,
];
/**
 * Computes global metadata for each of the assessments.
 */
export function getAssessmentInfos(
  intl: IntlShape,
  currentEvaluationResults: RunEvaluationTracesDataEntry[],
  otherEvaluationResults: RunEvaluationTracesDataEntry[] | undefined,
): AssessmentInfo[] {
  const assessmentInfos: Record<string, AssessmentInfo> = {};
  // Compute dtypes in the first pass.
  const assessmentDtypes: Record<string, AssessmentDType | undefined> = {
    [KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT]: 'pass-fail',
  };
  // Set of all assessment names. Will be filled after the first pass when computing dtypes.
  const assessmentNames = new Set<string>();

  [...currentEvaluationResults, ...(otherEvaluationResults || [])].forEach((result) => {
    const responseAssessmentsByName: [string, RunEvaluationResultAssessment[]][] = Object.entries(
      result.responseAssessmentsByName || {},
    );

    const overallAssessmentsByName: [string, RunEvaluationResultAssessment[]][] = result.overallAssessments.map(
      (assessment) => [KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT as string, [assessment]],
    );
    const retrievalAssessmentsByName: [string, RunEvaluationResultAssessment[]][] = [];
    result.retrievalChunks?.forEach((chunk) => {
      // Iterate chunk.retrievalAssessmentsByName
      for (const [assessmentName, assessments] of Object.entries(chunk.retrievalAssessmentsByName || {})) {
        retrievalAssessmentsByName.push([assessmentName, assessments]);
      }
    });

    for (const [assessmentName, assessments] of [
      ...responseAssessmentsByName,
      ...overallAssessmentsByName,
      ...retrievalAssessmentsByName,
    ]) {
      assessmentNames.add(assessmentName);
      const assessment = assessments[0];
      // For string values, if we see a value that is not "yes" or "no", we treat it as a string.
      // This is not a great approach, we should probably actually pass the pass-fail dtype information back somehow.
      let dtype: AssessmentDType | undefined = !isNil(assessment.stringValue)
        ? 'pass-fail'
        : !isNil(assessment.numericValue)
        ? 'numeric'
        : !isNil(assessment.booleanValue)
        ? 'boolean'
        : undefined;

      if (doesAssessmentContainErrors(assessment)) {
        dtype = undefined;
      }

      if (!assessmentDtypes[assessmentName]) {
        if (assessmentName in KnownEvaluationResultAssessmentValueLabel) {
          dtype = 'pass-fail';
        }
        assessmentDtypes[assessmentName] = dtype;
      }

      // Treat non-"yes"|"no" as string values.
      if (
        dtype === 'pass-fail' &&
        !isNil(assessment.stringValue) &&
        !PASS_FAIL_VALUES.includes(assessment.stringValue)
      ) {
        assessmentDtypes[assessmentName] = 'string';
      }

      // If the dtype is not the same as the current dtype (meaning there's mixed data types),
      // treat it as a string.
      if (dtype !== undefined && dtype !== assessmentDtypes[assessmentName]) {
        assessmentDtypes[assessmentName] = 'string';
      }
    }
  });

  // if any assessment does not have a dtype, give it 'unknown' type. this can happen if all evaluations for that assessment are errors
  for (const assessmentName of assessmentNames) {
    if (!assessmentDtypes[assessmentName]) {
      assessmentDtypes[assessmentName] = 'unknown';
    }
  }

  [...currentEvaluationResults, ...(otherEvaluationResults || [])].forEach((result) => {
    const responseAssessmentsByName: [string, RunEvaluationResultAssessment[]][] = Object.entries(
      result.responseAssessmentsByName || {},
    );

    const overallAssessmentsByName: [string, RunEvaluationResultAssessment[]][] = result.overallAssessments.map(
      (assessment) => [KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT as string, [assessment]],
    );
    const retrievalAssessmentsByName: [string, RunEvaluationResultAssessment[]][] = [];
    result.retrievalChunks?.forEach((chunk) => {
      // Iterate chunk.retrievalAssessmentsByName
      for (const [assessmentName, assessments] of Object.entries(chunk.retrievalAssessmentsByName || {})) {
        retrievalAssessmentsByName.push([assessmentName, assessments]);
      }
    });

    const assessmentNames = Object.keys(assessmentDtypes);
    for (const assessmentName of assessmentNames) {
      const assessmentsByName = [
        ...responseAssessmentsByName.filter(([name]) => name === assessmentName),
        ...overallAssessmentsByName.filter(([name]) => name === assessmentName),
        ...retrievalAssessmentsByName.filter(([name]) => name === assessmentName),
      ];
      // NOTE: We only take the first assessment as row-level judges produce a single assessment.
      const assessments = assessmentsByName.map(([_, assessments]) => assessments[0]);
      const assessment: RunEvaluationResultAssessment | undefined = assessments[0];

      const isError = doesAssessmentContainErrors(assessment);

      if (isNil(assessmentInfos[assessmentName])) {
        let displayName: string;
        let metricName: string;
        let isCustomMetric = false;

        const isKnown = KnownEvaluationResultAssessmentValueLabel[assessmentName] !== undefined;
        if (isKnown) {
          displayName = intl.formatMessage(KnownEvaluationResultAssessmentValueLabel[assessmentName]);
          metricName = assessmentName;
          isCustomMetric = false;
        } else {
          const { metricName: customMetricName, assessmentName: customAssessmentName } =
            getCustomMetricNameAndAssessment(assessmentName);
          displayName = customAssessmentName || '-';
          metricName = customMetricName;
          if (assessment?.source?.sourceType === 'CODE') {
            isCustomMetric = true;
          }
        }
        const dtype = assessmentDtypes[assessmentName] || 'string';

        const docsLink = getJudgeMetricsLink(ASSESSMENTS_DOC_LINKS[assessmentName]);
        const missingTooltip =
          assessmentName in KnownEvaluationResultAssessmentValueMissingTooltip
            ? intl.formatMessage(KnownEvaluationResultAssessmentValueMissingTooltip[assessmentName])
            : '';
        const description =
          assessmentName in KnownEvaluationResultAssessmentValueDescription
            ? intl.formatMessage(KnownEvaluationResultAssessmentValueDescription[assessmentName])
            : assessment?.source?.sourceType === 'HUMAN'
            ? intl.formatMessage({
                defaultMessage: 'This assessment is produced by a human judge.',
                description: 'Human judge assessment description',
              })
            : intl.formatMessage({
                defaultMessage: 'This assessment is produced by a custom metric.',
                description: 'Custom judge assessment description',
              });

        let assessmentValue = assessment ? getEvaluationResultAssessmentValue(assessment) : undefined;
        if (assessmentValue === null) assessmentValue = undefined;

        const uniqueValues = new Set<AssessmentValueType>();
        if (!isError) {
          uniqueValues.add(assessmentValue);
        }

        assessmentInfos[assessmentName] = {
          name: assessmentName,
          displayName: displayName,
          isKnown,
          metricName,
          isCustomMetric,
          source: assessment?.source,
          dtype,
          uniqueValues,
          isOverall: assessmentName === KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT,
          docsLink,
          missingTooltip,
          description,
          isEditable: assessment?.source?.sourceType === 'AI_JUDGE' || assessment?.source?.sourceType === 'HUMAN',
          isRetrievalAssessment: retrievalAssessmentsByName.some(([name]) => name === assessmentName),
          containsErrors: isError,
        };
      } else {
        const assessmentInfo = assessmentInfos[assessmentName];
        let value = assessment ? getEvaluationResultAssessmentValue(assessment) : undefined;
        if (isNil(value)) value = undefined;
        if (!isError) {
          assessmentInfo.uniqueValues.add(value);
        }

        // Update isEditable.
        if (!assessmentInfo.isEditable) {
          assessmentInfo.isEditable =
            assessment?.source?.sourceType === 'AI_JUDGE' || assessment?.source?.sourceType === 'HUMAN';
        }

        // isRetrievalAssessment should be true if any evaluation result has this assessment.
        assessmentInfo.isRetrievalAssessment =
          assessmentInfo.isRetrievalAssessment || retrievalAssessmentsByName.some(([name]) => name === assessmentName);

        assessmentInfo.containsErrors = assessmentInfo.containsErrors || isError;
      }
    }
  });

  // Remove the overall assessment if it does not have any non-null values.
  const seenOverallAssessmentValues =
    assessmentInfos[KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT]?.uniqueValues || new Set();
  const hasOverallValue =
    seenOverallAssessmentValues.has(KnownEvaluationResultAssessmentStringValue.YES) ||
    seenOverallAssessmentValues.has(KnownEvaluationResultAssessmentStringValue.NO);
  if (!hasOverallValue) {
    delete assessmentInfos[KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT];
  }

  return sortAssessmentInfos(Object.values(assessmentInfos));
}

export function sortAssessmentInfos(assessmentInfos: AssessmentInfo[]): AssessmentInfo[] {
  // Sort by DEFAULT_ASSESSMENTS_SORT_ORDER, fall back to alphabetical order of (metricName, name) which come after.
  return assessmentInfos.sort((a, b) => {
    const orderA = DEFAULT_ASSESSMENTS_SORT_ORDER.indexOf(a.name);
    const orderB = DEFAULT_ASSESSMENTS_SORT_ORDER.indexOf(b.name);

    // If both are in the sort order, compare their indices
    if (orderA !== -1 && orderB !== -1) {
      return orderA - orderB;
    }

    // If only one is in the sort order, prioritize it
    if (orderA !== -1) return -1;
    if (orderB !== -1) return 1;

    // Otherwise, sort by name alphabetically
    return a.name.localeCompare(b.name);
  });
}

export function getNumericAggregate(numericValues: number[]): NumericAggregate | undefined {
  if (numericValues.length === 0) {
    return undefined;
  }

  const numericAggregateCounts: NumericAggregateCount[] = [];
  const min = Math.min(...numericValues);
  const max = Math.max(...numericValues);

  // Set a minimum bucket size of 0.01, since the data is displayed in 2 decimal places.
  // Show at most 10 buckets.
  const bucketSize = Math.max(0.01, (max - min) / 10);
  let maxCount = 0;

  if (min === max) {
    numericAggregateCounts.push({ lower: min, upper: max, count: numericValues.length });
  } else {
    for (let i = min; i < max; i += bucketSize) {
      numericAggregateCounts.push({ lower: i, upper: Math.min(i + bucketSize, max), count: 0 });
    }
  }

  numericAggregateCounts.sort((a, b) => a.lower - b.lower);
  for (const numericValue of numericValues) {
    const bucket = numericAggregateCounts.find(
      (bucket) =>
        numericValue >= bucket.lower &&
        (numericValue < bucket.upper || (numericValue === bucket.upper && numericValue === max)),
    );
    if (bucket) {
      bucket.count++;
      maxCount = Math.max(maxCount, bucket.count);
    }
  }
  return { min, max, maxCount, counts: numericAggregateCounts };
}

export function getAssessmentNumericAggregates(
  assessmentInfo: AssessmentInfo,
  evalResults: RunEvaluationTracesDataEntry[],
): NumericAggregate | undefined {
  if (assessmentInfo.dtype !== 'numeric') {
    return undefined;
  }
  const numericValues = evalResults
    .flatMap((evalResult) => {
      const assessments = assessmentInfo.isOverall
        ? evalResult.overallAssessments
        : evalResult.responseAssessmentsByName[assessmentInfo.name];
      const result = assessments
        ?.map((assessment) => getEvaluationResultAssessmentValue(assessment))
        .filter((value) => value !== undefined && typeof value === 'number');
      return result;
    })
    .filter((value) => !isNil(value));
  return getNumericAggregate(numericValues);
}

function getAssessmentRunValueCounts(
  assessmentInfo: AssessmentInfo,
  evalResults: RunEvaluationTracesDataEntry[],
): AssessmentRunCounts | undefined {
  if (assessmentInfo.dtype === 'numeric') {
    return undefined;
  }
  const valueCounts: AssessmentRunCounts = new Map();
  evalResults.forEach((evalResult) => {
    const assessments = assessmentInfo.isOverall
      ? evalResult.overallAssessments
      : evalResult.responseAssessmentsByName[assessmentInfo.name];
    const valueCountsBySourceId =
      assessments && assessments.length > 0
        ? getUniqueValueCountsBySourceId(assessmentInfo, assessments)
        : [{ value: undefined, count: 1 }];
    const keysToCount = assessmentInfo.containsErrors
      ? [ERROR_KEY, ...assessmentInfo.uniqueValues]
      : assessmentInfo.uniqueValues;
    for (const uniqueValue of keysToCount) {
      const valueCountBySourceId = valueCountsBySourceId.find((valueCount) => valueCount.value === uniqueValue);
      const count = valueCountBySourceId ? valueCountBySourceId.count : 0;
      valueCounts.set(uniqueValue, (valueCounts.get(uniqueValue) || 0) + count);
    }
  });
  return valueCounts;
}

function getAssessmentRunNumericValues(
  assessmentInfo: AssessmentInfo,
  evalResults: RunEvaluationTracesDataEntry[],
): number[] | undefined {
  if (assessmentInfo.dtype !== 'numeric') {
    return undefined;
  }
  const values: number[] = [];
  evalResults.forEach((evalResult) => {
    const assessment = assessmentInfo.isOverall
      ? first(evalResult.overallAssessments)
      : first(evalResult.responseAssessmentsByName[assessmentInfo.name]);
    if (assessment) {
      const value = getEvaluationResultAssessmentValue(assessment);

      if (!isNil(value)) {
        values.push(Number(value));
      }
    }
  });
  return values;
}

function getRootCauseAssessmentCount(
  assessmentInfo: AssessmentInfo,
  evalResults: RunEvaluationTracesDataEntry[],
): number {
  let numRootCause = 0;
  evalResults.forEach((evalResult) => {
    if (isNil(evalResult)) return;
    const overallAssessment = first(evalResult.overallAssessments);
    if (overallAssessment?.rootCauseAssessment?.assessmentName === assessmentInfo.name) {
      numRootCause++;
    }
  });
  return numRootCause;
}

export function getAssessmentAggregates(
  assessmentInfo: AssessmentInfo,
  evalResults: EvalTraceComparisonEntry[],
  allAssessmentFilters: AssessmentFilter[],
): AssessmentAggregates {
  const currentEvalResults = evalResults.map((entry) => entry.currentRunValue).filter((entry) => !isNil(entry));
  const otherEvalResults = evalResults.map((entry) => entry.otherRunValue).filter((entry) => !isNil(entry));

  const currentAssessmentAggregates = getAssessmentRunValueCounts(assessmentInfo, currentEvalResults);
  const otherAssessmentAggregates = getAssessmentRunValueCounts(assessmentInfo, otherEvalResults);

  const currentNumericAggregates = getAssessmentNumericAggregates(assessmentInfo, currentEvalResults);

  const assessmentFilters = allAssessmentFilters.filter((filter) => filter.assessmentName === assessmentInfo.name);

  return {
    assessmentInfo,
    currentCounts: currentAssessmentAggregates,
    otherCounts: otherAssessmentAggregates,
    currentNumericValues: getAssessmentRunNumericValues(assessmentInfo, currentEvalResults),
    otherNumericValues: getAssessmentRunNumericValues(assessmentInfo, otherEvalResults),
    currentNumericAggregate: currentNumericAggregates,
    currentNumRootCause: getRootCauseAssessmentCount(assessmentInfo, currentEvalResults),
    otherNumRootCause: getRootCauseAssessmentCount(assessmentInfo, otherEvalResults),
    assessmentFilters,
  };
}

/**
 * Computes the total aggregate score for an assessment and evaluation results.
 *
 * For pass-fail dtypes, it computes what percentage of the runs have the assessment value as 'yes'.
 * for boolean dtypes, it computes what percentage of the runs have the assessment value as 'true'.
 */
export function getAssessmentAggregateOverallFraction(
  assessmentInfo: AssessmentInfo,
  assessmentRunCounts: AssessmentRunCounts = new Map(),
): number {
  if (assessmentInfo.dtype === 'pass-fail' || assessmentInfo.dtype === 'boolean') {
    let total = 0;
    let passCount = 0;
    for (const [value, count] of assessmentRunCounts) {
      if (isAssessmentPassing(assessmentInfo, value)) {
        passCount += count;
      }
      // We only consider non-null values for the total score.
      if (!isNil(value) && value !== ERROR_KEY) {
        total += count;
      }
    }
    return total > 0 ? passCount / total : 0;
  }
  return 0;
}

function getAssessmentBarChartValueBarItem(
  intl: IntlShape,
  assessmentInfo: AssessmentInfo,
  assessmentFilters: AssessmentFilter[],
  value: string | boolean | number | undefined,
  valueCounts: AssessmentRunCounts,
  runName: string,
  toggleAssessmentFilter: (
    assessmentName: string,
    filterValue: AssessmentValueType,
    run: string,
    filterType: AssessmentFilter['filterType'],
  ) => void,
): StackedRunBarchartItem {
  let numEvals = 0;
  const isErrorOrNull = value === ERROR_KEY || value === undefined;
  for (const [key, count] of valueCounts) {
    if (key !== undefined && key !== ERROR_KEY) {
      numEvals += count;
    }
  }
  const numValue = valueCounts.get(value) || 0;
  const fraction = !isErrorOrNull && numEvals > 0 ? numValue / numEvals : 0;

  const filterType: AssessmentFilter['filterType'] = undefined;

  return {
    value: valueCounts.get(value) || 0,
    fraction,
    isSelected: assessmentFilters.some(
      (filter) =>
        filter.filterValue === value && filter.assessmentName === assessmentInfo.name && filter.run === runName,
    ),
    toggleFilter: () => toggleAssessmentFilter(assessmentInfo.name, value, runName, filterType),
    tooltip: !isErrorOrNull
      ? intl.formatMessage(
          {
            defaultMessage: '{numValue}/{numEvals} for run "{runName}"',
            description: 'Passing assessment tooltip',
          },
          {
            numValue: numValue,
            numEvals,
            runName,
          },
        )
      : intl.formatMessage(
          {
            defaultMessage: '{numValue} for run "{runName}"',
            description: 'Error/null assessment tooltip',
          },
          {
            numValue,
            runName,
          },
        ),
  };
}

function getBarChartKeys(assessmentInfo: AssessmentInfo) {
  const keys = getSortedUniqueValues(assessmentInfo);
  if (assessmentInfo.containsErrors) {
    keys.push(ERROR_KEY);
  }

  return keys;
}

export function getBarChartData(
  intl: IntlShape,
  theme: ThemeType,
  assessmentInfo: AssessmentInfo,
  assessmentFilters: AssessmentFilter[],
  toggleAssessmentFilter: (
    assessmentName: string,
    filterValue: AssessmentValueType,
    run: string,
    filterType: AssessmentFilter['filterType'],
  ) => void,
  displayInfoCounts: AssessmentAggregates,
  currentRunDisplayName?: string,
  compareToRunDisplayName?: string,
): StackedBarchartItem[] {
  const showCompareData = displayInfoCounts.otherCounts !== undefined;

  const barItems: StackedBarchartItem[] = [];

  for (const value of getBarChartKeys(assessmentInfo)) {
    const currentBarItem = displayInfoCounts.currentCounts
      ? getAssessmentBarChartValueBarItem(
          intl,
          assessmentInfo,
          assessmentFilters,
          value,
          displayInfoCounts.currentCounts,
          // For monitoring, there is no run name so we allow this to pass through.
          currentRunDisplayName || 'monitor',
          toggleAssessmentFilter,
        )
      : undefined;

    const otherBarItem =
      showCompareData && compareToRunDisplayName
        ? getAssessmentBarChartValueBarItem(
            intl,
            assessmentInfo,
            assessmentFilters,
            value,
            displayInfoCounts.otherCounts || new Map(),
            compareToRunDisplayName,
            toggleAssessmentFilter,
          )
        : undefined;

    const isErrorOrNull = value === ERROR_KEY || value === undefined;
    const scoreChange =
      showCompareData && currentBarItem && otherBarItem
        ? isErrorOrNull
          ? currentBarItem.value - (otherBarItem?.value || 0)
          : currentBarItem.fraction - (otherBarItem?.fraction || 0)
        : undefined;
    // Only include Error or Null if there's a non-zero value or score change.
    if (currentBarItem && (!isErrorOrNull || currentBarItem.value !== 0 || (scoreChange && scoreChange !== 0))) {
      barItems.push({
        name: getAssessmentBarChartValueText(intl, theme, assessmentInfo, value),
        current: currentBarItem,
        other: otherBarItem,
        backgroundColor: getAssessmentValueBarBackgroundColor(theme, assessmentInfo, value, value === ERROR_KEY),
        scoreChange,
      });
    }
  }

  return barItems;
}

function getSortedUniqueValues(assessmentInfo: AssessmentInfo) {
  const uniqueValuesArray = Array.from(assessmentInfo.uniqueValues);
  if (assessmentInfo.dtype === 'pass-fail') {
    // Always show "YES" and "NO". We don't always show missing to reduce vertical space usage.
    if (!uniqueValuesArray.includes(KnownEvaluationResultAssessmentStringValue.YES)) {
      uniqueValuesArray.push(KnownEvaluationResultAssessmentStringValue.YES);
    }
    if (!uniqueValuesArray.includes(KnownEvaluationResultAssessmentStringValue.NO)) {
      uniqueValuesArray.push(KnownEvaluationResultAssessmentStringValue.NO);
    }
    const sortOrder: string[] = [
      KnownEvaluationResultAssessmentStringValue.YES,
      KnownEvaluationResultAssessmentStringValue.NO,
    ];

    // Sort the unique values based on the order of the known values.
    return uniqueValuesArray.sort((a, b) => {
      const aIndex = sortOrder.indexOf(a as string);
      const bIndex = sortOrder.indexOf(b as string);
      return aIndex - bIndex;
    });
  } else if (assessmentInfo.dtype === 'boolean') {
    // Sort by the value.
    return uniqueValuesArray.sort((a, b) => {
      return (a as boolean) === true ? -1 : 1;
    });
  }

  // Sort the assessment.
  return uniqueValuesArray.sort();
}

function getAssessmentBarChartValueText(
  intl: IntlShape,
  theme: ThemeType,
  assessmentInfo: AssessmentInfo,
  value: string | boolean | number | undefined,
): string {
  if (assessmentInfo.dtype === 'pass-fail') {
    if (value === KnownEvaluationResultAssessmentStringValue.YES) {
      return intl.formatMessage({
        defaultMessage: 'Pass',
        description: 'The label for a passing asseessment above a bar-chart in the summary stats.',
      });
    } else if (value === KnownEvaluationResultAssessmentStringValue.NO) {
      return intl.formatMessage({
        defaultMessage: 'Fail',
        description: 'The label for a failing asseessment above a bar-chart in the summary stats.',
      });
    } else if (value === ERROR_KEY) {
      return intl.formatMessage({
        defaultMessage: 'Error',
        description: 'The label for an error asseessment above a bar-chart in the summary stats.',
      });
    } else {
      return intl.formatMessage({
        defaultMessage: 'null',
        description: 'null assessment label',
      });
    }
  } else if (assessmentInfo.dtype === 'boolean') {
    if (value === true) {
      return intl.formatMessage({
        defaultMessage: 'True',
        description: 'True assessment label',
      });
    } else if (value === false) {
      return intl.formatMessage({
        defaultMessage: 'False',
        description: 'False assessment label',
      });
    } else {
      return intl.formatMessage({
        defaultMessage: 'null',
        description: 'null assessment label',
      });
    }
  }
  return isNil(value) ? 'null' : `${value}`;
}

/**
 * Compute the counts for each of the values given a set of assessments.
 */
export function getUniqueValueCountsBySourceId(
  assessmentInfo: AssessmentInfo,
  assessments: RunEvaluationResultAssessment[],
): {
  value: AssessmentValueType | undefined;
  count: number;
  latestAssessment: RunEvaluationResultAssessment;
}[] {
  const filteredAssessments = assessments.filter((assessment) => assessment.name === assessmentInfo.name);
  // Compute the unique values of assessments.
  let uniqueValues = new Set<AssessmentValueType | undefined>();
  for (const assessment of filteredAssessments) {
    const value = getEvaluationResultAssessmentValue(assessment);
    uniqueValues.add(value);
  }

  // Sort by the latest timestamp.
  filteredAssessments.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));

  // Separate assessments with errors from those without.
  const errorAssessments = filteredAssessments.filter((assessment) => doesAssessmentContainErrors(assessment));
  const validAssessments = filteredAssessments.filter((assessment) => !doesAssessmentContainErrors(assessment));

  // Recompute the unique values after filtering.
  uniqueValues = new Set<AssessmentValueType | undefined>();
  for (const assessment of validAssessments) {
    const value = getEvaluationResultAssessmentValue(assessment);
    uniqueValues.add(value);
  }

  // Compute the counts for each of the unique values.
  const valueCounts: {
    value: AssessmentValueType | undefined;
    count: number;
    latestAssessment: RunEvaluationResultAssessment;
  }[] = [];
  for (const value of uniqueValues) {
    const assessmentsWithValue = filteredAssessments.filter(
      (assessment) => getEvaluationResultAssessmentValue(assessment) === value,
    );
    const count = assessmentsWithValue.length;
    valueCounts.push({ value, count, latestAssessment: assessmentsWithValue[0] });
  }

  // Add an entry for errors.
  if (errorAssessments.length > 0) {
    valueCounts.push({
      value: ERROR_KEY,
      count: errorAssessments.length,
      latestAssessment: errorAssessments[0],
    });
  }

  return valueCounts;
}
