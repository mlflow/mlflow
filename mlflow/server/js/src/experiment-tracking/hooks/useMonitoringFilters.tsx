import { useCallback, useMemo } from 'react';
import { useSearchParams } from '../../common/utils/RoutingUtils';
import { useMonitoringConfig } from './useMonitoringConfig';

export const START_TIME_LABEL_QUERY_PARAM_KEY = 'startTimeLabel';
const START_TIME_QUERY_PARAM_KEY = 'startTime';
const END_TIME_QUERY_PARAM_KEY = 'endTime';

export type START_TIME_LABEL =
  | 'LAST_HOUR'
  | 'LAST_24_HOURS'
  | 'LAST_7_DAYS'
  | 'LAST_30_DAYS'
  | 'LAST_YEAR'
  | 'ALL'
  | 'CUSTOM';
export const DEFAULT_START_TIME_LABEL: START_TIME_LABEL = 'LAST_7_DAYS';

export interface MonitoringFilters {
  startTimeLabel?: START_TIME_LABEL;
  startTime?: string;
  endTime?: string;
}

/**
 * Query param-powered hook that returns the monitoring filters from the URL.
 */
export const useMonitoringFilters = () => {
  const monitoringConfig = useMonitoringConfig();

  const [searchParams, setSearchParams] = useSearchParams();

  const startTimeLabel =
    (searchParams.get(START_TIME_LABEL_QUERY_PARAM_KEY) as START_TIME_LABEL | undefined) || DEFAULT_START_TIME_LABEL;
  let startTime = searchParams.get(START_TIME_QUERY_PARAM_KEY) || undefined;
  let endTime = searchParams.get(END_TIME_QUERY_PARAM_KEY) ?? undefined;
  if (startTimeLabel !== 'CUSTOM') {
    const absoluteStartEndTime = getAbsoluteStartEndTime(monitoringConfig.dateNow, { startTimeLabel });
    startTime = absoluteStartEndTime.startTime;
    endTime = absoluteStartEndTime.endTime;
  } else {
    startTime = searchParams.get(START_TIME_QUERY_PARAM_KEY) || undefined;
    endTime = searchParams.get(END_TIME_QUERY_PARAM_KEY) ?? undefined;
  }

  const monitoringFilters = useMemo<MonitoringFilters>(
    () => ({
      startTimeLabel,
      startTime,
      endTime,
    }),
    [startTimeLabel, startTime, endTime],
  );

  const setMonitoringFilters = useCallback(
    (monitoringFilters: MonitoringFilters | undefined, replace = false) => {
      setSearchParams(
        (params) => {
          if (monitoringFilters?.startTime === undefined) {
            params.delete(START_TIME_QUERY_PARAM_KEY);
          } else if (monitoringFilters.startTimeLabel === 'CUSTOM') {
            params.set(START_TIME_QUERY_PARAM_KEY, monitoringFilters.startTime);
          }
          if (monitoringFilters?.endTime === undefined) {
            params.delete(END_TIME_QUERY_PARAM_KEY);
          } else if (monitoringFilters.startTimeLabel === 'CUSTOM') {
            params.set(END_TIME_QUERY_PARAM_KEY, monitoringFilters.endTime);
          }
          if (monitoringFilters?.startTimeLabel === undefined) {
            params.delete(START_TIME_LABEL_QUERY_PARAM_KEY);
          } else {
            params.set(START_TIME_LABEL_QUERY_PARAM_KEY, monitoringFilters.startTimeLabel);
          }
          return params;
        },
        { replace },
      );
    },
    [setSearchParams],
  );

  return [monitoringFilters, setMonitoringFilters] as const;
};

export function getAbsoluteStartEndTime(
  dateNow: Date,
  monitoringFilters: MonitoringFilters,
): {
  startTime: string | undefined;
  endTime: string | undefined;
} {
  if (monitoringFilters.startTimeLabel && monitoringFilters.startTimeLabel !== 'CUSTOM') {
    return startTimeLabelToStartEndTime(dateNow, monitoringFilters.startTimeLabel);
  }
  return {
    startTime: monitoringFilters.startTime,
    endTime: monitoringFilters.endTime,
  };
}

export function startTimeLabelToStartEndTime(
  dateNow: Date,
  startTimeLabel: START_TIME_LABEL,
): {
  startTime: string | undefined;
  endTime: string | undefined;
} {
  switch (startTimeLabel) {
    case 'LAST_HOUR':
      return {
        startTime: new Date(new Date(dateNow).setUTCHours(new Date().getUTCHours() - 1)).toISOString(),
        endTime: dateNow.toISOString(),
      };
    case 'LAST_24_HOURS':
      return {
        startTime: new Date(new Date(dateNow).setUTCDate(new Date().getUTCDate() - 1)).toISOString(),
        endTime: dateNow.toISOString(),
      };
    case 'LAST_7_DAYS':
      return {
        startTime: new Date(new Date(dateNow).setUTCDate(new Date().getUTCDate() - 7)).toISOString(),
        endTime: dateNow.toISOString(),
      };
    case 'LAST_30_DAYS':
      return {
        startTime: new Date(new Date(dateNow).setUTCDate(new Date().getUTCDate() - 30)).toISOString(),
        endTime: dateNow.toISOString(),
      };
    case 'LAST_YEAR':
      return {
        startTime: new Date(new Date(dateNow).setUTCFullYear(new Date().getUTCFullYear() - 1)).toISOString(),
        endTime: dateNow.toISOString(),
      };
    case 'ALL':
      return {
        startTime: undefined,
        endTime: dateNow.toISOString(),
      };
    default:
      throw new Error(`Unexpected start time label: ${startTimeLabel}`);
  }
}
