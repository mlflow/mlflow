import type { RunsChartsRunData } from '../components/RunsCharts.common';
import type { RunsChartsParallelCardConfig } from '../runs-charts.types';

export type ParallelCoordinateDataEntry = Record<string, string | number | null>;
export const PARALLEL_CHART_MAX_NUMBER_STRINGS = 30;

// Preprocesses the generic runs/groups data into a format that can be used by the parallel coordinates chart:
// - Extracts the selected params and metrics from the run data
// - Filters out incompatible data
export function processParallelCoordinateData(
  chartRunData: RunsChartsRunData[],
  selectedParams: string[],
  selectedMetrics: string[],
): ParallelCoordinateDataEntry[] {
  const allRuns = chartRunData.map((run) => {
    const result: ParallelCoordinateDataEntry = {
      uuid: run.uuid,
    };
    function computeSelectedAttrs(attrs: string[], runvalue: Record<string, any>) {
      if (Array.isArray(attrs)) {
        for (const attr of attrs) {
          if (attr in runvalue) {
            result[attr] = runvalue[attr].value;
          } else {
            result[attr] = null;
          }
        }
      }
    }
    computeSelectedAttrs(selectedParams, run.params);
    computeSelectedAttrs(selectedMetrics, run.metrics);
    return result;
  });

  return filterParallelCoordinateData(allRuns, selectedParams, selectedMetrics);
}

// Filters the input data for parallel coordinates chart:
// - Remove runs that don't have complete metric and param data for the configured chart
// - For numerical columns, filter out runs with NaN and null values
// - For string columns, show the runs that correspond with the 30 most recent unique string values
// - For columns with both, choose whichever view will show more runs
export function filterParallelCoordinateData(
  allRuns: ParallelCoordinateDataEntry[],
  selectedParams: string[] = [],
  selectedMetrics: string[] = [],
): ParallelCoordinateDataEntry[] {
  if (allRuns.length === 0) {
    return [];
  }

  // We filter out runs that don't have complete metric and param data for the configured chart
  const completedRuns = allRuns.filter(
    (run) =>
      selectedParams.every((param) => run[param] !== null) && selectedMetrics.every((metric) => run[metric] !== null),
  );

  if (completedRuns.length === 0) {
    return [];
  }

  const keys = Object.keys(completedRuns[0]);
  keys.shift(); // remove uuid as a key
  let stringRuns: ParallelCoordinateDataEntry[] = completedRuns;

  // add runs with any string values until 30 unique values in any column
  keys.forEach((key) => {
    const numberRuns = stringRuns.filter((x) => {
      return !isNaN(Number(x[key])) && x[key] !== null;
    }); // this logic should remain the same as in getAxesTypes() for casting

    if (numberRuns.length >= stringRuns.length / 2) {
      stringRuns = numberRuns;
      return;
    }

    const seenVals: Set<string> = new Set();
    const tempRuns = [];
    for (const run of stringRuns) {
      const value = run[key];
      if (seenVals.size === PARALLEL_CHART_MAX_NUMBER_STRINGS && seenVals.has(String(value))) {
        tempRuns.push(run);
      } else if (seenVals.size < PARALLEL_CHART_MAX_NUMBER_STRINGS && isNaN(Number(value))) {
        seenVals.add(String(value));
        tempRuns.push(run);
      }
    }
    stringRuns = tempRuns;
  });

  return stringRuns;
}

export const isParallelChartConfigured = (cardConfig: RunsChartsParallelCardConfig) => {
  const selectedParamsCount = cardConfig.selectedParams?.length || 0;
  const selectedMetricsCount = cardConfig.selectedMetrics?.length || 0;

  return selectedParamsCount + selectedMetricsCount >= 2;
};
