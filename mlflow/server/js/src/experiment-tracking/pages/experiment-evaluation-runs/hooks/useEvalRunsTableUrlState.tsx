import type { OnChangeFn, SortingState } from '@tanstack/react-table';
import { useCallback, useMemo, useState } from 'react';

import type { RunsGroupByConfig } from '../../../components/experiment-page/utils/experimentPage.group-row-utils';
import {
  RunGroupingAggregateFunction,
  RunGroupingMode,
} from '../../../components/experiment-page/utils/experimentPage.row-types';

import { useSearchParams } from '../../../../common/utils/RoutingUtils';

export const EVAL_RUNS_GROUP_BY_QUERY_PARAM = 'evalRunsGroupBy';
export const EVAL_RUNS_SORTING_QUERY_PARAM = 'evalRunsSorting';

const GROUPING_MODES = new Set<string>(Object.values(RunGroupingMode));
const GROUPING_AGGREGATE_FUNCTIONS = new Set<string>(Object.values(RunGroupingAggregateFunction));

const isRunsGroupByConfig = (value: unknown): value is RunsGroupByConfig => {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const candidate = value as Partial<RunsGroupByConfig>;
  return (
    GROUPING_AGGREGATE_FUNCTIONS.has(candidate.aggregateFunction ?? '') &&
    Array.isArray(candidate.groupByKeys) &&
    candidate.groupByKeys.length > 0 &&
    candidate.groupByKeys.every(
      (key) =>
        typeof key === 'object' &&
        key !== null &&
        GROUPING_MODES.has(key.mode) &&
        typeof key.groupByData === 'string' &&
        key.groupByData.length > 0,
    )
  );
};

const parseGroupBy = (value: string | null, fallback: RunsGroupByConfig | null): RunsGroupByConfig | null => {
  if (value === null) {
    return fallback;
  }
  try {
    const parsed: unknown = JSON.parse(value);
    return parsed === null || isRunsGroupByConfig(parsed) ? parsed : fallback;
  } catch {
    return fallback;
  }
};

const parseSorting = (value: string | null): SortingState => {
  if (value === null) {
    return [];
  }
  try {
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed) &&
      parsed.every(
        (entry) =>
          typeof entry === 'object' &&
          entry !== null &&
          typeof entry.id === 'string' &&
          entry.id.length > 0 &&
          typeof entry.desc === 'boolean',
      )
      ? parsed
      : [];
  } catch {
    return [];
  }
};

export const useEvalRunsSorting = (enabled = true) => {
  const [searchParams, setSearchParams] = useSearchParams();
  const sortingParam = enabled ? searchParams.get(EVAL_RUNS_SORTING_QUERY_PARAM) : null;
  const sorting = useMemo(() => parseSorting(sortingParam), [sortingParam]);

  const setSorting = useCallback<OnChangeFn<SortingState>>(
    (updater) => {
      setSearchParams(
        (params) => {
          const currentSorting = parseSorting(params.get(EVAL_RUNS_SORTING_QUERY_PARAM));
          const nextSorting = typeof updater === 'function' ? updater(currentSorting) : updater;
          if (nextSorting.length > 0) {
            params.set(EVAL_RUNS_SORTING_QUERY_PARAM, JSON.stringify(nextSorting));
          } else {
            params.delete(EVAL_RUNS_SORTING_QUERY_PARAM);
          }
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  return [sorting, setSorting] as const;
};

export const useEvalRunsGrouping = (fallback: RunsGroupByConfig | null, enabled = true) => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [localGroupBy, setLocalGroupBy] = useState<RunsGroupByConfig | null>(fallback);
  const groupByParam = enabled ? searchParams.get(EVAL_RUNS_GROUP_BY_QUERY_PARAM) : null;
  const urlGroupBy = useMemo(() => parseGroupBy(groupByParam, fallback), [fallback, groupByParam]);

  const setUrlGroupBy = useCallback(
    (nextGroupBy: RunsGroupByConfig | null) => {
      setSearchParams(
        (params) => {
          params.set(EVAL_RUNS_GROUP_BY_QUERY_PARAM, JSON.stringify(nextGroupBy));
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  return enabled ? ([urlGroupBy, setUrlGroupBy] as const) : ([localGroupBy, setLocalGroupBy] as const);
};
