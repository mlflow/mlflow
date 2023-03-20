import { Skeleton } from '@databricks/design-system';
import React, { Suspense } from 'react';
import { CompareChartRunData } from './CompareRunsCharts.common';

const ParallelCoordinatesPlot = React.lazy(() => import('./ParallelCoordinatesPlot'));

const LazyParallelCoordinatesPlot = (props: any) => {
  return (
    <Suspense fallback={<Skeleton />}>
      <ParallelCoordinatesPlot {...props}></ParallelCoordinatesPlot>
    </Suspense>
  );
};

// Map all metrics, params and run uuid
// Filter data to only keep the selected params and metrics from each run
export function processData(
  chartRunData: CompareChartRunData[],
  selectedParams: string[],
  selectedMetrics: string[],
) {
  return chartRunData.map((run) => {
    const result: Record<string, string | number | null> = {
      uuid: run.runInfo.run_uuid,
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
}
export default LazyParallelCoordinatesPlot;
