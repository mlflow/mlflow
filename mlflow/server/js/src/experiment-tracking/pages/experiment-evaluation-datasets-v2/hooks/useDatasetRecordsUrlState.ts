import { useCallback } from 'react';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useNumberSearchParam } from './useNumberSearchParam';
import {
  DEFAULT_SORT_COLUMN,
  DEFAULT_SORT_DIR,
  RECORD_COLUMN_IDS,
  type RecordColumnId,
  type SortDirection,
} from '../utils/constants';

const Q_PARAM = 'q';
const PAGE_PARAM = 'page';
const RECORD_ID_PARAM = 'recordId';
const SORT_PARAM = 'sort';
const DIR_PARAM = 'dir';

const RECORD_COLUMN_ID_SET = new Set<string>(RECORD_COLUMN_IDS);
const isRecordColumnId = (value: string | null): value is RecordColumnId =>
  value !== null && RECORD_COLUMN_ID_SET.has(value);

interface DatasetRecordsUrlState {
  search: string;
  setSearch: (next: string) => void;
  pageIndex: number;
  setPageIndex: (next: number) => void;
  recordId: string | undefined;
  setRecordId: (next: string | undefined) => void;
  /** Active sort column id. Falls back to the default when absent or unrecognized. */
  sort: RecordColumnId;
  dir: SortDirection;
  setSort: (column: RecordColumnId, direction: SortDirection) => void;
}

/**
 * Owns the records-page URL state: search query, 1-based page index, selected record id,
 * and sort column + direction.
 *
 * Search changes always reset the page back to 1 — without this, the cursor from the previous
 * query would be applied against a filtered result set and the user could land on an empty page.
 *
 * Sort defaults to `last_updated DESC` (matches V1). Writing the default removes both URL
 * params so URLs stay tidy when the user hasn't changed sort. The query hook validates the
 * column id, so an unrecognized `?sort=` value silently degrades to default ordering.
 */
export const useDatasetRecordsUrlState = (): DatasetRecordsUrlState => {
  const [search, setSearchParams] = useSearchParams((params) => params.get(Q_PARAM) ?? '');
  const [pageIndex, setPageIndex] = useNumberSearchParam({ key: PAGE_PARAM, defaultValue: 1, min: 1 });
  const [recordId] = useSearchParams((params) => params.get(RECORD_ID_PARAM) ?? undefined);
  const [sortRaw] = useSearchParams((params) => params.get(SORT_PARAM));
  const [dirRaw] = useSearchParams((params) => params.get(DIR_PARAM));

  const sort: RecordColumnId = isRecordColumnId(sortRaw) ? sortRaw : DEFAULT_SORT_COLUMN;
  const dir: SortDirection = dirRaw === 'asc' ? 'asc' : dirRaw === 'desc' ? 'desc' : DEFAULT_SORT_DIR;

  const setSearch = useCallback(
    (next: string) => {
      setSearchParams((params) => {
        if (next) {
          params.set(Q_PARAM, next);
        } else {
          params.delete(Q_PARAM);
        }
        params.delete(PAGE_PARAM);
        return params;
      });
    },
    [setSearchParams],
  );

  const setRecordId = useCallback(
    (next: string | undefined) => {
      setSearchParams((params) => {
        if (next) {
          params.set(RECORD_ID_PARAM, next);
        } else {
          params.delete(RECORD_ID_PARAM);
        }
        return params;
      });
    },
    [setSearchParams],
  );

  const setSort = useCallback(
    (column: RecordColumnId, direction: SortDirection) => {
      setSearchParams((params) => {
        if (column === DEFAULT_SORT_COLUMN && direction === DEFAULT_SORT_DIR) {
          params.delete(SORT_PARAM);
          params.delete(DIR_PARAM);
        } else {
          params.set(SORT_PARAM, column);
          params.set(DIR_PARAM, direction);
        }
        // Sorting reframes the result set — staying on page N would land the user on an
        // arbitrary slice of the resorted data rather than the top of what they just asked for.
        params.delete(PAGE_PARAM);
        return params;
      });
    },
    [setSearchParams],
  );

  return { search, setSearch, pageIndex, setPageIndex, recordId, setRecordId, sort, dir, setSort };
};
