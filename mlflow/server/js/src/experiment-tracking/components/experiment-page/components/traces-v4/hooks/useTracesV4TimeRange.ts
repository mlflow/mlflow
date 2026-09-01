import { useCallback, useMemo } from 'react';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import { shouldEnableTracesTableStatePersistence } from '@databricks/web-shared/model-trace-explorer';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useMonitoringConfig } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig';
import {
  DEFAULT_TRACES_V4_TIME_LABEL,
  getStartEndForLabel,
  toValidTimeLabel,
  type StartEndTime,
  type TracesV4TimeLabel,
} from '../utils/timeRange';

const START_TIME_LABEL_PARAM = 'startTimeLabel';
const START_TIME_PARAM = 'startTime';
const END_TIME_PARAM = 'endTime';

// Bump if the persisted shape changes so stale entries reset.
const TIME_RANGE_STORAGE_VERSION = 1;
const TIME_RANGE_STORAGE_KEY_PREFIX = 'mlflow.traces-v4.time-range';

/** The time-range state the caller sets. `startTime`/`endTime` (ISO) matter only for `CUSTOM`. */
export interface TracesV4TimeRangeSelection {
  timeLabel: TracesV4TimeLabel;
  startTime?: string;
  endTime?: string;
}

export interface UseTracesV4TimeRangeResult {
  /** Effective label (URL → persisted → default), always resolved. */
  timeLabel: TracesV4TimeLabel;
  /** Resolved ISO bounds — computed for relative labels, explicit for `CUSTOM`. */
  startTime?: string;
  endTime?: string;
  setTimeRange: (next: TracesV4TimeRangeSelection) => void;
  /** ms-string bounds the data layer consumes (replicates `useMonitoringFiltersTimeRange`). */
  timeRangeMs: StartEndTime;
}

/**
 * Standalone time-range state for the V4 traces tab. Deliberately depends on **zero** shared
 * time-range persistence (not `useMonitoringFilters`) so a returning v3 user's saved selection can't
 * bleed into v4: the label lives in the URL (deep-linkable) with a v4-only localStorage fallback key.
 *
 * The effective label resolves `urlLabel ?? persistedLabel ?? DEFAULT` synchronously with no mount
 * side-effect — the default shows without writing it to the URL. Only pure/stateless monitoring infra
 * (`useMonitoringConfig` for `dateNow`) is reused.
 */
export const useTracesV4TimeRange = (experimentId: string): UseTracesV4TimeRangeResult => {
  const { dateNow } = useMonitoringConfig();
  const persistenceEnabled = shouldEnableTracesTableStatePersistence();

  // OSS's `useSearchParams` is the raw react-router hook (no read-selector overload), so read each
  // param off the `searchParams` object directly.
  const [searchParams, setSearchParams] = useSearchParams();
  const urlLabelRaw = searchParams.get(START_TIME_LABEL_PARAM);
  const urlStartTime = searchParams.get(START_TIME_PARAM) ?? undefined;
  const urlEndTime = searchParams.get(END_TIME_PARAM) ?? undefined;

  const [persisted, setPersisted] = useLocalStorage<TracesV4TimeRangeSelection | undefined>({
    key: `${TIME_RANGE_STORAGE_KEY_PREFIX}.${experimentId}`,
    version: TIME_RANGE_STORAGE_VERSION,
    initialValue: undefined,
  });

  const urlLabel = toValidTimeLabel(urlLabelRaw);
  const persistedLabel = persistenceEnabled ? toValidTimeLabel(persisted?.timeLabel) : undefined;
  const timeLabel = urlLabel ?? persistedLabel ?? DEFAULT_TRACES_V4_TIME_LABEL;

  // For CUSTOM, bounds come from the URL (or the persisted fallback when the URL is bare); for a
  // relative label they're computed from `dateNow`.
  const { startTime, endTime } = useMemo<StartEndTime>(() => {
    if (timeLabel === 'CUSTOM') {
      return {
        startTime: urlStartTime ?? (urlLabel ? undefined : persisted?.startTime),
        endTime: urlEndTime ?? (urlLabel ? undefined : persisted?.endTime),
      };
    }
    return getStartEndForLabel(dateNow, timeLabel);
  }, [timeLabel, urlLabel, urlStartTime, urlEndTime, persisted?.startTime, persisted?.endTime, dateNow]);

  const setTimeRange = useCallback(
    (next: TracesV4TimeRangeSelection) => {
      if (persistenceEnabled) {
        setPersisted(next);
      }
      setSearchParams((params) => {
        params.set(START_TIME_LABEL_PARAM, next.timeLabel);
        // Only CUSTOM carries explicit bounds in the URL; every other label derives them from `dateNow`.
        if (next.timeLabel === 'CUSTOM' && next.startTime) {
          params.set(START_TIME_PARAM, next.startTime);
        } else {
          params.delete(START_TIME_PARAM);
        }
        if (next.timeLabel === 'CUSTOM' && next.endTime) {
          params.set(END_TIME_PARAM, next.endTime);
        } else {
          params.delete(END_TIME_PARAM);
        }
        return params;
      });
    },
    [persistenceEnabled, setPersisted, setSearchParams],
  );

  const timeRangeMs = useMemo<StartEndTime>(
    () => ({
      startTime: startTime ? new Date(startTime).getTime().toString() : undefined,
      endTime: endTime ? new Date(endTime).getTime().toString() : undefined,
    }),
    [startTime, endTime],
  );

  return { timeLabel, startTime, endTime, setTimeRange, timeRangeMs };
};
