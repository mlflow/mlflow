import { useMemo } from 'react';
import {
  useListDatasetRecordsQuery,
  type DatasetRecord,
} from '../hooks/useDatasetsQueries';
import { type RecordColumnId, type SortDirection } from '../utils/constants';

interface UseDatasetRecordsPageQueryParams {
  datasetId: string;
  /** 1-based page index. */
  pageIndex: number;
  pageSize: number;
  /** Search string. Empty disables filtering. Client-side match across inputs/expectations JSON + record id. */
  searchValue: string;
  /** Column id to sort by. Non-sortable columns fall back to default (last_updated DESC). */
  sort: RecordColumnId;
  dir: SortDirection;
}

/**
 * Accessor map. Keys are UI column ids; values are functions that return the comparable
 * primitive on the record. `last_updated` maps to the proto field `last_update_time` —
 * the UI id is intentionally different from the proto field name.
 *
 * Anything not in this map is not sortable; the query returns the filter order unchanged.
 */
type SortAccessor = (record: DatasetRecord) => string | undefined;

const SORT_ACCESSORS = {
  dataset_record_id: (r) => r.dataset_record_id,
  create_time: (r) => r.create_time,
  created_by: (r) => r.created_by,
  last_updated: (r) => r.last_update_time,
  last_updated_by: (r) => r.last_updated_by,
} satisfies Record<string, SortAccessor>;

type SortKey = keyof typeof SORT_ACCESSORS;
const isSortKey = (sort: RecordColumnId): sort is SortKey => Object.prototype.hasOwnProperty.call(SORT_ACCESSORS, sort);

// Safe lookup: non-sortable column ids return `undefined`, which the query hook treats as
// "no sort" and returns the filter order unchanged.
const getSortAccessor = (sort: RecordColumnId): SortAccessor | undefined =>
  isSortKey(sort) ? SORT_ACCESSORS[sort] : undefined;

interface DatasetRecordsPage {
  records: DatasetRecord[];
  totalRecords: number;
  isLoading: boolean;
  isFetching: boolean;
  error: unknown;
  refetch: () => void;
  allRecords: DatasetRecord[];
  /**
   * Timestamp (ms since epoch) of the last successful underlying query resolution, or `0`
   * if the data has never loaded. Forwarded from React Query so the page can render a
   * "Updated N seconds ago" indicator without tracking its own state.
   */
  dataUpdatedAt: number;
}

/**
 * Backed by the shared `useListDatasetRecordsQuery` which loops over all server pages —
 * the records endpoint doesn't accept a server-side filter, so we need the full list
 * locally to support search. Pagination + search slicing happen client-side.
 *
 * Trade-off acknowledged: this scales to thousands of records but not to millions. If
 * that becomes a constraint we'll need a backend filter; until then this matches the
 * legacy tab's behavior while keeping the V2 UI snappy.
 */
export const useDatasetRecordsPageQuery = ({
  datasetId,
  pageIndex,
  pageSize,
  searchValue,
  sort,
  dir,
}: UseDatasetRecordsPageQueryParams): DatasetRecordsPage => {
  const query = useListDatasetRecordsQuery(datasetId);
  const allRecords = useMemo(() => query.data ?? [], [query.data]);

  // Pre-stringify each record's searchable fields once per data load — keystroke filtering
  // would otherwise repeat JSON.stringify over the entire list on every input change.
  const haystacks = useMemo(() => {
    const map = new Map<string, string>();
    for (const record of allRecords) {
      const parts = [
        record.dataset_record_id,
        record.inputs ? JSON.stringify(record.inputs) : '',
        record.expectations ? JSON.stringify(record.expectations) : '',
      ];
      map.set(record.dataset_record_id, parts.join(' ').toLowerCase());
    }
    return map;
  }, [allRecords]);

  const filtered = useMemo(() => {
    const trimmed = searchValue.trim().toLowerCase();
    if (!trimmed) {
      return allRecords;
    }
    return allRecords.filter((record) => haystacks.get(record.dataset_record_id)?.includes(trimmed));
  }, [allRecords, haystacks, searchValue]);

  // ISO-8601 timestamps sort chronologically under lex order, so a single localeCompare
  // path handles both date and identifier columns. `dataset_record_id` is the stable
  // tiebreaker so equal sort keys never reorder spuriously between renders.
  const sorted = useMemo(() => {
    const accessor = getSortAccessor(sort);
    if (!accessor) return filtered;
    const sign = dir === 'asc' ? 1 : -1;
    return [...filtered].sort((a, b) => {
      const av = accessor(a) ?? '';
      const bv = accessor(b) ?? '';
      if (av !== bv) return sign * av.localeCompare(bv);
      return a.dataset_record_id.localeCompare(b.dataset_record_id);
    });
  }, [filtered, sort, dir]);

  const start = (pageIndex - 1) * pageSize;
  const records = useMemo(() => sorted.slice(start, start + pageSize), [sorted, start, pageSize]);

  return {
    records,
    totalRecords: sorted.length,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    error: query.error,
    refetch: query.refetch,
    allRecords,
    dataUpdatedAt: query.dataUpdatedAt,
  };
};
