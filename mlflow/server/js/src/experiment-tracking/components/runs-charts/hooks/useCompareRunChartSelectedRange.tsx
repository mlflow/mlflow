import { useMemo, useState } from 'react';
import { findChartStepsByTimestampForRuns } from '../utils/findChartStepsByTimestamp';
import { SampledMetricsByRunUuidState } from '../../../types';
import { isNumber, isString } from 'lodash';
import { RunsChartsLineChartXAxisType } from '../components/RunsCharts.common';

/**
 * Hook used in compare run charts. It's responsible for converting selected range
 * (which can be either step or timestamp) to step range, based on chart axis type.
 *
 * @param xAxisKey Can be 'step', 'time' or 'time-relative'
 * @param metricKey
 * @param sampledMetricsByRunUuid Recorded history for metrics for runs in compare chart
 * @param runUuids List of run UUIDs in compare chart
 */
export const useCompareRunChartSelectedRange = (
  xAxisKey: RunsChartsLineChartXAxisType,
  metricKey: string,
  sampledMetricsByRunUuid: SampledMetricsByRunUuidState,
  runUuids: string[],
) => {
  const [range, setRange] = useState<[number | string, number | string] | undefined>(undefined);
  const [offsetTimestamp, setOffsetTimestamp] = useState<[number, number] | undefined>(undefined);

  const stepRange = useMemo<[number, number] | undefined>(() => {
    if (!range) {
      return undefined;
    }
    if (xAxisKey === RunsChartsLineChartXAxisType.TIME && isString(range[0]) && isString(range[1])) {
      // If we're dealing with absolute time-based chart axis, find corresponding steps based on timestamp
      const bounds = findChartStepsByTimestampForRuns(
        sampledMetricsByRunUuid,
        runUuids,
        metricKey,
        range as [string, string],
      );
      return bounds;
    }

    if (
      xAxisKey === RunsChartsLineChartXAxisType.TIME_RELATIVE &&
      offsetTimestamp &&
      isNumber(range[0]) &&
      isNumber(range[1])
    ) {
      // If we're dealing with absolute time-based chart axis, find corresponding steps based on timestamp
      const bounds = findChartStepsByTimestampForRuns(
        sampledMetricsByRunUuid,
        runUuids,
        metricKey,
        offsetTimestamp as [number, number],
      );
      return bounds;
    }

    if (xAxisKey === RunsChartsLineChartXAxisType.STEP && isNumber(range[0]) && isNumber(range[1])) {
      // If we're dealing with step-based chart axis, use those steps but incremented/decremented
      const lowerBound = Math.floor(range[0]);
      const upperBound = Math.ceil(range[1]);
      return lowerBound && upperBound ? [lowerBound - 1, upperBound + 1] : undefined;
    }

    // return undefined for xAxisKey === 'metric' because there isn't
    // necessarily a mapping between value range and step range
    return undefined;
  }, [xAxisKey, metricKey, range, sampledMetricsByRunUuid, runUuids, offsetTimestamp]);

  return {
    /**
     * Sets actually selected range in chart (can be timestamp, seconds range or step range)
     */
    setRange,
    /**
     * If there's an offset timestamp calculated from relative runs, set it using this function
     */
    setOffsetTimestamp,
    /**
     * Resulting step range
     */
    stepRange,
    /**
     * Actual raw range to be passed down to the chart component
     */
    range,
  };
};
