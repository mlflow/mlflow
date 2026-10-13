import type { OnChangeFn, SortingState } from '@tanstack/react-table';
import { useCallback, useMemo } from 'react';

import { useSearchParams } from '../../../../common/utils/RoutingUtils';

export const EVAL_RUNS_SORTING_QUERY_PARAM = 'evalRunsSorting';

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
