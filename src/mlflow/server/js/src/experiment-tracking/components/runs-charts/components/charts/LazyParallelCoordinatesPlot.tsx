import { LegacySkeleton } from '@databricks/design-system';
import React, { Suspense } from 'react';
import { RunsChartsRunData } from '../RunsCharts.common';

const ParallelCoordinatesPlot = React.lazy(() => import('./ParallelCoordinatesPlot'));

export const MAX_NUMBER_STRINGS = 30;
export type ParallelCoordinateDataEntry = Record<string, string | number | null>;

const LazyParallelCoordinatesPlot = (props: any) => {
  return (
    <Suspense fallback={<LegacySkeleton />}>
      <ParallelCoordinatesPlot {...props}></ParallelCoordinatesPlot>
    </Suspense>
  );
};

// Map all metrics, params and run uuid
// Filter data to only keep the selected params and metrics from each run
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

  return filterParallelCoordinateData(allRuns);
}

// filter data so each column only shows #s or strs, whichever allows more to be displayed
export function filterParallelCoordinateData(allRuns: ParallelCoordinateDataEntry[]): ParallelCoordinateDataEntry[] {
  if (allRuns.length === 0) {
    return allRuns;
  }

  const keys = Object.keys(allRuns[0]);
  keys.shift(); // remove uuid as a key
  let stringRuns: ParallelCoordinateDataEntry[] = allRuns;

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
      if (seenVals.size === MAX_NUMBER_STRINGS && seenVals.has(String(value))) {
        tempRuns.push(run);
      } else if (seenVals.size < MAX_NUMBER_STRINGS && isNaN(Number(value))) {
        seenVals.add(String(value));
        tempRuns.push(run);
      }
    }
    stringRuns = tempRuns;
  });

  return stringRuns;
}

export default LazyParallelCoordinatesPlot;
