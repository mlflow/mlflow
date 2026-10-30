import { useCallback, useMemo } from 'react';
import {
  FilterOp,
  isComparisonSortableTraceColumn,
  type TraceFilterModel,
  type TraceSortColumnId,
  type SortDirection,
} from '@databricks/web-shared/traces-table';
import { useSearchParams } from '../../../common/utils/RoutingUtils';

const FILTER_OPERATORS = new Set<string>(Object.values(FilterOp));
const isClause = (value: unknown): value is TraceFilterModel[number] =>
  typeof value === 'object' &&
  value !== null &&
  'field' in value &&
  typeof value.field === 'string' &&
  'operator' in value &&
  typeof value.operator === 'string' &&
  FILTER_OPERATORS.has(value.operator) &&
  'value' in value &&
  typeof value.value === 'string' &&
  (!('key' in value) || value.key === undefined || typeof value.key === 'string');

const parseFilters = (raw: string | null): { filterModel: TraceFilterModel; invalidFilters: boolean } => {
  if (raw === null) {
    return { filterModel: [], invalidFilters: false };
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.every(isClause)) {
      return { filterModel: parsed, invalidFilters: false };
    }
  } catch {
    // Keep the invalid state observable; treating it as no filters could show unrelated results.
  }
  return { filterModel: [], invalidFilters: true };
};

/** Comparison criteria never overwrite single-run URL state or drawer/run-selection links. */
export const useEvalRunComparisonTracesUrlState = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const search = searchParams.get('comparisonQ') ?? '';
  const filtersRaw = searchParams.get('comparisonFilters');
  const sortRaw = searchParams.get('comparisonSort');
  const dirRaw = searchParams.get('comparisonDir');
  const isGroupedBySession = searchParams.get('comparisonGroupBy') === 'session';
  const filters = useMemo(() => parseFilters(filtersRaw), [filtersRaw]);
  const validSort =
    sortRaw !== null && isComparisonSortableTraceColumn(sortRaw) && (dirRaw === 'asc' || dirRaw === 'desc');
  const sort: TraceSortColumnId = validSort ? sortRaw : 'start_time';
  const dir: SortDirection = validSort ? dirRaw : 'desc';
  const setSearch = useCallback(
    (next: string) => {
      setSearchParams((params) => {
        if (next) {
          params.set('comparisonQ', next);
        } else {
          params.delete('comparisonQ');
        }
        return params;
      });
    },
    [setSearchParams],
  );
  const setFilterModel = useCallback(
    (next: TraceFilterModel) => {
      setSearchParams((params) => {
        if (next.length) {
          params.set('comparisonFilters', JSON.stringify(next));
        } else {
          params.delete('comparisonFilters');
        }
        return params;
      });
    },
    [setSearchParams],
  );
  const setSort = useCallback(
    (column: TraceSortColumnId, direction: SortDirection) => {
      setSearchParams((params) => {
        params.set('comparisonSort', column);
        params.set('comparisonDir', direction);
        return params;
      });
    },
    [setSearchParams],
  );
  const clearFilters = useCallback(() => setFilterModel([]), [setFilterModel]);
  const setIsGroupedBySession = useCallback(
    (next: boolean) => {
      setSearchParams((params) => {
        if (next) {
          params.set('comparisonGroupBy', 'session');
        } else {
          params.delete('comparisonGroupBy');
        }
        return params;
      });
    },
    [setSearchParams],
  );
  const clearAll = useCallback(() => {
    setSearchParams((params) => {
      params.delete('comparisonQ');
      params.delete('comparisonFilters');
      return params;
    });
  }, [setSearchParams]);

  return {
    search,
    ...filters,
    sort,
    dir,
    isGroupedBySession,
    setIsGroupedBySession,
    setSearch,
    setFilterModel,
    setSort,
    clearFilters,
    clearAll,
    criteriaKey: JSON.stringify([search, filtersRaw, sort, dir, isGroupedBySession]),
  };
};
