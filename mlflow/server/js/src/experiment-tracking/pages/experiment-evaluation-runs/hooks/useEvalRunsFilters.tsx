import { useCallback, useMemo, useState, type SetStateAction } from 'react';

import type { EvalRunsFilterClause } from '../evalRunsFilterFields';
import { buildClausesFilterString, EvalRunsFilterOperator } from '../evalRunsFilterFields';
import { EvalRunsDatePreset, usePersistedDatePreset } from './usePersistedDatePreset';
import { useSearchParams } from '../../../../common/utils/RoutingUtils';
import { useComposedSearchParamsSetter } from './useComposedSearchParamsSetter';

export { EvalRunsDatePreset } from './usePersistedDatePreset';

const PRESET_DURATIONS_MS: Record<Exclude<EvalRunsDatePreset, EvalRunsDatePreset.ALL>, number> = {
  [EvalRunsDatePreset.LAST_HOUR]: 60 * 60 * 1000,
  [EvalRunsDatePreset.LAST_24_HOURS]: 24 * 60 * 60 * 1000,
  [EvalRunsDatePreset.LAST_7_DAYS]: 7 * 24 * 60 * 60 * 1000,
  [EvalRunsDatePreset.LAST_30_DAYS]: 30 * 24 * 60 * 60 * 1000,
  [EvalRunsDatePreset.LAST_YEAR]: 365 * 24 * 60 * 60 * 1000,
};

export interface EvalRunsFiltersState {
  datePreset: EvalRunsDatePreset;
  /** Structured multi-key filter rows (field/operator/value), Traces-filter style. Run type is one
   *  of the filterable fields — there is no separate type preset. */
  clauses: EvalRunsFilterClause[];
}

export const EVAL_RUNS_SEARCH_FILTER_QUERY_PARAM = 'evalRunsSearchFilter';
export const EVAL_RUNS_FILTER_CLAUSES_QUERY_PARAM = 'evalRunsFilterClauses';

const FILTER_OPERATORS = new Set<string>(Object.values(EvalRunsFilterOperator));

const parseFilterClauses = (value: string | null): EvalRunsFilterClause[] => {
  if (!value) {
    return [];
  }
  try {
    const parsed = JSON.parse(value);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter(
      (clause): clause is EvalRunsFilterClause =>
        typeof clause === 'object' &&
        clause !== null &&
        typeof clause.field === 'string' &&
        FILTER_OPERATORS.has(clause.operator) &&
        typeof clause.value === 'string',
    );
  } catch {
    return [];
  }
};

/** Default date window for the V2 toolbar. The Date button's clear (X) resets here. */
export const EVAL_RUNS_DEFAULT_DATE_PRESET_V2 = EvalRunsDatePreset.ALL;

const buildEvalRunsFilterString = (state: EvalRunsFiltersState, searchFilter: string, now = Date.now()) => {
  const expressions: string[] = [];

  const trimmedSearch = searchFilter.trim();
  if (trimmedSearch) {
    expressions.push(trimmedSearch);
  }

  if (state.datePreset !== EvalRunsDatePreset.ALL) {
    // NB: the timestamp must be unquoted - a quoted number is rejected by the parser.
    expressions.push(`attributes.start_time > ${now - PRESET_DURATIONS_MS[state.datePreset]}`);
  }

  const clausesExpression = buildClausesFilterString(state.clauses);
  if (clausesExpression) {
    expressions.push(clausesExpression);
  }

  return expressions.join(' AND ');
};

export const useEvalRunsFilters = ({
  fallbackDatePreset = EvalRunsDatePreset.ALL,
  experimentId,
  persistDatePreset = false,
  enabled = true,
}: {
  /** Initial date window. */
  fallbackDatePreset?: EvalRunsDatePreset;
  /** Scopes the persisted preset per experiment, matching the Traces V4 time range. */
  experimentId: string;
  /** Persists the date preset to the URL and localStorage when enabled. */
  persistDatePreset?: boolean;
  /** When false, the search filter is local state, and the URL filters and clauses are not read. */
  enabled?: boolean;
}) => {
  const [searchParams, setRouterSearchParams] = useSearchParams();
  const setSearchParams = useComposedSearchParamsSetter(searchParams, setRouterSearchParams);
  const [datePreset, setDatePreset] = usePersistedDatePreset({
    enabled: persistDatePreset,
    experimentId,
    fallbackDatePreset,
    searchParams,
    setSearchParams,
  });
  const [localSearchFilter, setLocalSearchFilter] = useState('');
  const searchFilter = enabled ? (searchParams.get(EVAL_RUNS_SEARCH_FILTER_QUERY_PARAM) ?? '') : localSearchFilter;
  const filterClausesParam = enabled ? searchParams.get(EVAL_RUNS_FILTER_CLAUSES_QUERY_PARAM) : null;
  const clauses = useMemo(() => parseFilterClauses(filterClausesParam), [filterClausesParam]);

  const setSearchFilter = useCallback(
    (next: SetStateAction<string>) => {
      if (!enabled) {
        setLocalSearchFilter(next);
        return;
      }
      setSearchParams(
        (params) => {
          const current = params.get(EVAL_RUNS_SEARCH_FILTER_QUERY_PARAM) ?? '';
          const value = typeof next === 'function' ? next(current) : next;
          if (value) {
            params.set(EVAL_RUNS_SEARCH_FILTER_QUERY_PARAM, value);
          } else {
            params.delete(EVAL_RUNS_SEARCH_FILTER_QUERY_PARAM);
          }
          return params;
        },
        { replace: true },
      );
    },
    [enabled, setSearchParams],
  );

  const setClauses = useCallback(
    (next: SetStateAction<EvalRunsFilterClause[]>) => {
      if (!enabled) {
        return;
      }
      setSearchParams(
        (params) => {
          const current = parseFilterClauses(params.get(EVAL_RUNS_FILTER_CLAUSES_QUERY_PARAM));
          const nextClauses = typeof next === 'function' ? next(current) : next;
          if (nextClauses.length === 0) {
            params.delete(EVAL_RUNS_FILTER_CLAUSES_QUERY_PARAM);
          } else {
            params.set(EVAL_RUNS_FILTER_CLAUSES_QUERY_PARAM, JSON.stringify(nextClauses));
          }
          return params;
        },
        { replace: true },
      );
    },
    [enabled, setSearchParams],
  );

  const datasetClauses = clauses.filter((clause) => clause.field === 'dataset.name');
  const selectedDataset =
    datasetClauses.length === 1 && datasetClauses[0].operator === EvalRunsFilterOperator.EQUALS
      ? datasetClauses[0].value
      : undefined;
  const setSelectedDataset = useCallback(
    (name: string | undefined) => {
      setClauses((currentClauses) => [
        ...currentClauses.filter((clause) => clause.field !== 'dataset.name'),
        ...(name === undefined
          ? []
          : [{ field: 'dataset.name', operator: EvalRunsFilterOperator.EQUALS, value: name }]),
      ]);
    },
    [setClauses],
  );

  const filters = useMemo<EvalRunsFiltersState>(() => ({ datePreset, clauses }), [datePreset, clauses]);

  const isAnyFilterActive =
    searchFilter.trim().length > 0 || datePreset !== EvalRunsDatePreset.ALL || clauses.length > 0;

  // Recomputing on every render would make the relative date boundary drift and
  // change the query key, refetching constantly. Pin it to the filter state.
  const filterString = useMemo(
    () => (enabled ? buildEvalRunsFilterString(filters, searchFilter) : searchFilter),
    [enabled, filters, searchFilter],
  );
  // The same filter with the date window measured from now, for each request (including a refresh).
  const resolveFilterString = useCallback(
    () => (enabled ? buildEvalRunsFilterString(filters, searchFilter) : searchFilter),
    [enabled, filters, searchFilter],
  );

  return {
    searchFilter,
    filters,
    filterString,
    resolveFilterString,
    isAnyFilterActive,
    setSearchFilter,
    setDatePreset,
    setClauses,
    selectedDataset,
    setSelectedDataset,
    hasDatasetFilter: datasetClauses.length > 0,
  };
};
