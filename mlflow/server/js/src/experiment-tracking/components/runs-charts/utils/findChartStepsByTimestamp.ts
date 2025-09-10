import { type Dictionary, compact, first, isUndefined, last, max, min, minBy, orderBy, values } from 'lodash';
import type { MetricEntity, SampledMetricsByRunUuidState } from '../../../types';
import type { SampledMetricsByRun } from '../hooks/useSampledMetricHistory';

/**
 * This function consumes chart timestamp range and returns
 * corresponding step range for a given metric basing on a history.
 */
export const findChartStepsByTimestamp = (
  // Currently fetched metric history
  currentMetricHistory: {
    [rangeKey: string]: {
      loading?: boolean | undefined;
      metricsHistory?: MetricEntity[] | undefined;
    };
  },
  // Timestamp range - either textual ("2022-10-23 10:00:00") or numeric (milliseconds)
  range: [string | number, string | number],
  // If set to true, will return entire boundaries from history if timestamps are not found.
  // Otherwise, will return undefined.
  useDefaultIfNotFound = true,
): [number, number] | undefined => {
  // First, let's compile a history of all metric values from all ranges,
  // then sort it by timestamp
  const flatHistory = orderBy(
    compact(
      values(currentMetricHistory)
        .map(({ metricsHistory }) => metricsHistory)
        .flat(),
    ),
    'timestamp',
  );

  // If there's no sufficient entries, return nothing
  if (flatHistory.length < 2) {
    return undefined;
  }

  // We consume textual ranges produced by charts so we have
  // to convert them to timestamps
  const lowerBound = new Date(range[0]).valueOf();
  const upperBound = new Date(range[1]).valueOf();

  // First, try to find the lower entry using loop
  let lowerEntry = useDefaultIfNotFound ? first(flatHistory) : undefined;

  for (let index = 0; index < flatHistory.length; index++) {
    const entry = flatHistory[index];
    if (entry.timestamp > lowerBound) {
      lowerEntry = flatHistory[index - 1] || entry;
      break;
    }
  }

  // Repeat for the upper entry
  let upperEntry = useDefaultIfNotFound ? last(flatHistory) : undefined;

  for (let index = flatHistory.length - 1; index >= 0; index--) {
    const entry = flatHistory[index];
    if (entry.timestamp < upperBound) {
      upperEntry = flatHistory[index + 1] || entry;
      break;
    }
  }

  // If boundaries are not found, return nothing
  if (isUndefined(lowerEntry) || isUndefined(upperEntry)) {
    return undefined;
  }

  // Return found boundary entries
  return [lowerEntry.step, upperEntry.step];
};

/**
 * Finds the chart steps by absolute timestamp for multiple runs.
 *
 * @param sampledMetrics - The sampled metrics by run UUID state.
 * @param runUuids - The array of run UUIDs.
 * @param metricKey - The metric key.
 * @param range - The range of timestamps.
 * @returns The lower and upper bounds of the chart steps, or undefined if not found.
 */
export const findChartStepsByTimestampForRuns = (
  sampledMetrics: SampledMetricsByRunUuidState,
  runUuids: string[],
  metricKey: string,
  range: [string | number, string | number],
): [number, number] | undefined => {
  const stepRangesPerRun = compact(
    runUuids.map((runUuid) => {
      const metricHistoryForRun = sampledMetrics[runUuid]?.[metricKey];
      return metricHistoryForRun ? findChartStepsByTimestamp(metricHistoryForRun, range, false) : undefined;
    }),
  );
  const lowerBound = min(stepRangesPerRun.map(([bound]) => bound));
  const upperBound = max(stepRangesPerRun.map(([, bound]) => bound));

  if (!isUndefined(lowerBound) && !isUndefined(upperBound)) {
    return [lowerBound, upperBound];
  }

  return undefined;
};

/**
 * This function consumes chart relative time range and returns
 * corresponding step range for a given metric basing on a history.
 *
 * @param currentlyVisibleMetrics currentlyVisibleMetrics is a dictionary of currently rendered metric traces for run
 * @param runUuids a list of run UUIDs to process
 * @param relativeRange a relative time range in seconds
 * @returns a range of steps or undefined if no relevant are found
 */
export const findAbsoluteTimestampRangeForRelativeRange = (
  currentlyVisibleMetrics: Dictionary<SampledMetricsByRun>,
  runUuids: string[],
  relativeRange: [number, number],
  multiplier = 1000,
): [number, number] | undefined => {
  const stepRangesPerRun = compact(
    runUuids.map((runUuid) => {
      const runData = currentlyVisibleMetrics[runUuid];

      if (!runData) {
        return null;
      }

      // omit the "runUuid" key so we can conveniently access the metrics data
      const { runUuid: _, ...runMetrics } = runData;

      // concat all the metrics history for the run
      const visibleMetricHistoryForRun = values(runMetrics).flatMap((metric) => metric.metricsHistory ?? []);

      // Find the timestamp offset for the run. Should be equal to lowest timestamp value for each run.
      const timestampOffset = minBy(visibleMetricHistoryForRun, 'timestamp')?.timestamp || 0;

      // Convert relative time range to timestamp range. Relative range comes
      // in seconds so we have to multiply it by 1000 to get milliseconds.
      return [relativeRange[0] * multiplier + timestampOffset, relativeRange[1] * multiplier + timestampOffset] as [
        number,
        number,
      ];
    }),
  );
  const lowerBound = min(stepRangesPerRun.map(([bound]) => bound));
  const upperBound = max(stepRangesPerRun.map(([, bound]) => bound));
  if (!isUndefined(lowerBound) && !isUndefined(upperBound)) {
    return [lowerBound, upperBound];
  }

  return undefined;
};
