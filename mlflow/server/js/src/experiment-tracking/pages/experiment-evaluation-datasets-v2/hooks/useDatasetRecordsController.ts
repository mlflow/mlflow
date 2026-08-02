import { useCallback, useEffect, useMemo } from 'react';
import type { DatasetRecord } from './useDatasetsQueries';
import { useDatasetRecordsPageQuery } from './useDatasetRecordsPageQuery';
import { useDatasetRecordsUrlState } from './useDatasetRecordsUrlState';
import { useBulkRecordSelection } from './useBulkRecordSelection';
import { useDebouncedSearchInput } from './useDebouncedSearchInput';
import { usePersistedTablePreferences } from './usePersistedTablePreferences';
import {
  DEFAULT_RECORD_PAGE_SIZE,
  DEFAULT_VISIBLE_RECORD_COLUMNS,
  RECORD_COLUMN_IDS,
  SEARCH_DEBOUNCE_MS,
  type RecordColumnId,
} from '../utils/constants';
import { clampPageIndex } from '../utils/clampPageIndex';

interface UseDatasetRecordsControllerParams {
  experimentId: string;
  datasetId: string;
}

export interface UseDatasetRecordsControllerResult {
  url: ReturnType<typeof useDatasetRecordsUrlState>;
  records: ReturnType<typeof useDatasetRecordsPageQuery>;
  selectedRecord: DatasetRecord | undefined;
  bulk: ReturnType<typeof useBulkRecordSelection>;
  tablePrefs: ReturnType<typeof usePersistedTablePreferences<RecordColumnId>>;
  /**
   * Debounced search input wired to URL state. The controller flushes any pending write
   * before a page-index transition so a mid-debounce keystroke doesn't clobber the
   * just-clicked Next-page navigation.
   */
  searchInput: ReturnType<typeof useDebouncedSearchInput>;
  /** Navigate to a page, flushing any pending debounced search write first. */
  setPageIndex: (next: number) => void;
  flags: {
    hasActiveSearch: boolean;
    hasNoRecordsAtAll: boolean;
    hasNoSearchResults: boolean;
  };
}

/**
 * Orchestrates the records-page data layer: URL state, dataset/records queries, column
 * persistence, bulk selection, plus the cross-cutting effects that respond to query
 * state. Owned here (rather than in the page component) so the data lifecycle has a
 * single testable seam — the page becomes a thin render layer.
 */
export const useDatasetRecordsController = ({
  experimentId,
  datasetId,
}: UseDatasetRecordsControllerParams): UseDatasetRecordsControllerResult => {
  const url = useDatasetRecordsUrlState();

  const searchInput = useDebouncedSearchInput({
    committedValue: url.search,
    onCommit: url.setSearch,
    debounceMs: SEARCH_DEBOUNCE_MS,
  });

  // Any URL transition that would race with a pending debounced search write must flush
  // first. Without this, typing in the search box and immediately clicking pagination lands
  // on `?page=N`, then the debounced search write fires and resets the page back to 1.
  const { setPageIndex: rawSetPageIndex } = url;
  const { flush: flushSearch } = searchInput;
  const setPageIndex = useCallback(
    (next: number) => {
      flushSearch();
      rawSetPageIndex(next);
    },
    [flushSearch, rawSetPageIndex],
  );

  const records = useDatasetRecordsPageQuery({
    datasetId,
    pageIndex: url.pageIndex,
    pageSize: DEFAULT_RECORD_PAGE_SIZE,
    searchValue: url.search,
    sort: url.sort,
    dir: url.dir,
  });

  // Recover from out-of-range URL pageIndex (e.g. stale link, or post-delete shrinkage).
  // Runs after the result settles so we don't clobber the URL while data is still loading in.
  useEffect(() => {
    if (records.isLoading) return;
    const clamped = clampPageIndex(url.pageIndex, records.totalRecords, DEFAULT_RECORD_PAGE_SIZE);
    if (clamped !== url.pageIndex) {
      setPageIndex(clamped);
    }
  }, [records.isLoading, records.totalRecords, url.pageIndex, setPageIndex]);

  const tablePrefs = usePersistedTablePreferences({
    experimentId,
    datasetId,
    allColumns: RECORD_COLUMN_IDS,
    defaultVisible: DEFAULT_VISIBLE_RECORD_COLUMNS,
  });

  const visibleIds = useMemo(() => records.records.map((record) => record.dataset_record_id), [records.records]);
  const bulk = useBulkRecordSelection(visibleIds);

  // Clear bulk selection when the filter changes — selected records that don't match the new
  // search wouldn't be meaningfully "selectable" from the user's POV, so a stale carry-over
  // would invite accidental deletion. Selection persists across pagination so users can
  // accumulate selections across pages (mirrors the GenAI traces selection model).
  const { clear: clearBulk } = bulk;
  useEffect(() => {
    clearBulk();
  }, [url.search, clearBulk]);

  const selectedRecord = useMemo(
    () => (url.recordId ? records.allRecords.find((record) => record.dataset_record_id === url.recordId) : undefined),
    [records.allRecords, url.recordId],
  );

  // Records have loaded but the URL points at a record that no longer exists (concurrent
  // delete, stale link): close the side panel instead of rendering an empty surface.
  //
  // Why this is a known-but-deferred risk: this path bypasses the page-level
  // `useGuardedTransition`, so unsaved edits in the side panel are silently discarded when
  // (a) another tab deletes the open record, (b) a refetch drops the row, or (c) the user
  // bulk-deletes the open record. Acceptable today because dirty state is in-memory only
  // and wouldn't survive a reload anyway; the proper fix is to route this close through
  // the page's `requestTransition` so the guard can surface its unsaved-changes modal.
  const { setRecordId } = url;
  useEffect(() => {
    if (url.recordId && !selectedRecord && !records.isLoading && records.allRecords.length > 0) {
      setRecordId(undefined);
    }
  }, [url.recordId, selectedRecord, records.isLoading, records.allRecords.length, setRecordId]);

  const hasActiveSearch = url.search.trim().length > 0;
  const hasNoRecordsAtAll = !records.isLoading && records.allRecords.length === 0;
  const hasNoSearchResults = !records.isLoading && records.totalRecords === 0 && hasActiveSearch;

  return {
    url,
    records,
    selectedRecord,
    bulk,
    tablePrefs,
    searchInput,
    setPageIndex,
    flags: { hasActiveSearch, hasNoRecordsAtAll, hasNoSearchResults },
  };
};
