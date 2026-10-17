import { useCallback, useMemo, useState, type RefObject } from 'react';
import type { InputRef } from '@databricks/design-system';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { FilterOp, type FilterClause, type TraceFilterModel } from '@databricks/web-shared/traces-table';
import { TracesV4Search } from '../components/TracesV4Search';
import type { TracesV4ToolbarSlots } from '../components/TracesV4Toolbar';
import type { UseDebouncedSearchInputResult } from './useDebouncedSearchInput';

interface UseTracesV4UnifiedSearchParams {
  filterModel: TraceFilterModel;
  setFilterModel: (next: TraceFilterModel) => void;
  tagFilters: { key: string; value: string }[];
  clearTagFilters: () => void;
  searchInput: UseDebouncedSearchInputResult;
}

interface RenderSearchControlParams {
  renderFilterEditor: TracesV4ToolbarSlots['renderFilterEditor'];
  traces: ModelTraceInfoV3[];
  inputRef: RefObject<InputRef>;
  /** Resets the in-memory recent searches when it changes (e.g. another experiment or run). */
  resetKey: string;
}

const isSameClause = (left: FilterClause, right: FilterClause) =>
  left.field === right.field &&
  left.key === right.key &&
  left.operator === right.operator &&
  left.value === right.value;

/**
 * State for the unified traces search, which shows the applied filters as pills inside the search box
 * and turns typed `field operator value` queries into filter clauses.
 */
export const useTracesV4UnifiedSearch = ({
  filterModel,
  setFilterModel,
  tagFilters,
  clearTagFilters,
  searchInput,
}: UseTracesV4UnifiedSearchParams) => {
  const [searchMinimumWidth, setSearchMinimumWidth] = useState(340);

  // Click-to-filter tags live in the URL. They show as pills too, and editing or removing any pill
  // moves them into the filter model so the pills keep a single source.
  const searchFilterModel = useMemo(
    () => [
      ...filterModel,
      ...tagFilters.map(({ key, value }) => ({ field: 'tag', key, operator: FilterOp.EQUALS, value })),
    ],
    [filterModel, tagFilters],
  );

  const setSearchFilters = useCallback(
    (next: TraceFilterModel) => {
      setFilterModel(next);
      if (tagFilters.length > 0) {
        clearTagFilters();
      }
    },
    [setFilterModel, tagFilters.length, clearTagFilters],
  );

  // Picking a filter replaces the typed query that produced it.
  const addSearchFilter = useCallback(
    (clause: FilterClause) => {
      if (!searchFilterModel.some((existing) => isSameClause(existing, clause))) {
        setFilterModel([...filterModel, clause]);
      }
      searchInput.clear();
    },
    [filterModel, searchFilterModel, setFilterModel, searchInput],
  );

  const runTextSearch = useCallback(
    (text: string) => {
      searchInput.setInput(text);
      searchInput.submit();
    },
    [searchInput],
  );

  const renderSearchControl = ({ renderFilterEditor, traces, inputRef, resetKey }: RenderSearchControlParams) =>
    renderFilterEditor((openEditor) => (
      <TracesV4Search
        key={resetKey}
        traces={traces}
        value={searchInput.input}
        onChange={searchInput.setInput}
        onClear={searchInput.clear}
        onSubmit={searchInput.submit}
        filterModel={searchFilterModel}
        onOpenFilter={openEditor}
        onRemoveFilter={(index) => setSearchFilters(searchFilterModel.filter((_, i) => i !== index))}
        onRunTextSearch={runTextSearch}
        onApplyFilter={addSearchFilter}
        onMinimumWidthChange={setSearchMinimumWidth}
        inputRef={inputRef}
      />
    ));

  return { searchFilterModel, setSearchFilters, searchMinimumWidth, renderSearchControl };
};
