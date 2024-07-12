import { useDesignSystemTheme } from '@databricks/design-system';

import { useExperimentTraces } from './hooks/useExperimentTraces';
import { TracesViewTable } from './TracesViewTable';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { TraceDataDrawer } from './TraceDataDrawer';
import { useEditExperimentTraceTags } from './hooks/useEditExperimentTraceTags';
import { TracesViewControls } from './TracesViewControls';
import { SortingState } from '@tanstack/react-table';
import { compact, isFunction, isNil, uniq } from 'lodash';
import { useExperimentViewTracesUIState } from './hooks/useExperimentViewTracesUIState';
import { ExperimentViewTracesTableColumns, getTraceInfoTotalTokens } from './TracesView.utils';
import { useActiveExperimentTrace } from './hooks/useActiveExperimentTrace';
import { ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';

export const TRACE_AUTO_REFRESH_INTERVAL = 30000;

const defaultSorting: SortingState = [{ id: ExperimentViewTracesTableColumns.timestampMs, desc: true }];

export const TracesView = ({
  experimentIds,
  runUuid,
  disabledColumns,
}: {
  experimentIds: string[];
  runUuid?: string;
  /**
   * Columns that should be disabled in the table.
   * Disabled columns are hidden and are not available to be toggled at all.
   */
  disabledColumns?: ExperimentViewTracesTableColumns[];
}) => {
  const timeoutRef = useRef<number | undefined>(undefined);
  const [filter, setFilter] = useState<string>('');
  const [sorting, setSorting] = useState<SortingState>(defaultSorting);
  const [rowSelection, setRowSelection] = useState<{ [id: string]: boolean }>({});

  const [selectedTraceId, setSelectedTraceId] = useActiveExperimentTrace();

  const { traces, loading, error, hasNextPage, hasPreviousPage, fetchNextPage, fetchPrevPage, refreshCurrentPage } =
    useExperimentTraces(experimentIds, sorting, filter, runUuid);

  const onTraceClicked = useCallback(
    ({ request_id }: ModelTraceInfo) => setSelectedTraceId(request_id),
    [setSelectedTraceId],
  );

  // clear row selections when the page changes.
  // the backend specifies a max of 100 deletions,
  // plus it's confusing to have selections on a
  // page that the user can't see
  const onNextPage = useCallback(() => {
    fetchNextPage();
    setRowSelection({});
  }, [fetchNextPage]);

  const onPreviousPage = useCallback(() => {
    fetchPrevPage();
    setRowSelection({});
  }, [fetchPrevPage]);

  // auto-refresh traces
  useEffect(() => {
    // if the hook reruns, clear the current timeout, since we'll be scheduling another
    window.clearTimeout(timeoutRef.current);

    const scheduleRefresh = async () => {
      // only refresh if the user is on the first page
      // otherwise it might mess with browsing old traces
      if (loading || hasPreviousPage) return;

      await refreshCurrentPage(true);

      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = window.setTimeout(scheduleRefresh, TRACE_AUTO_REFRESH_INTERVAL);
    };

    timeoutRef.current = window.setTimeout(scheduleRefresh, TRACE_AUTO_REFRESH_INTERVAL);
    return () => window.clearTimeout(timeoutRef.current);
  }, [refreshCurrentPage, loading, hasPreviousPage]);

  const { theme } = useDesignSystemTheme();

  // Try to find the trace info for the currently selected trace id
  const selectedTraceInfo = useMemo(() => {
    if (!selectedTraceId) return undefined;
    return traces.find((trace) => trace.request_id === selectedTraceId);
  }, [selectedTraceId, traces]);

  const {
    // hiddenColumns is a list of columns that are hidden by the user.
    uiState,
    toggleHiddenColumn,
  } = useExperimentViewTracesUIState(experimentIds);

  const existingTagKeys = useMemo(
    () => uniq(compact(traces.flatMap((trace) => trace.tags?.map((tag) => tag.key)))),
    [traces],
  );

  const { showEditTagsModalForTrace, EditTagsModal } = useEditExperimentTraceTags({
    onSuccess: () => refreshCurrentPage(true),
    existingTagKeys,
  });

  const usingFilters = filter !== '';

  const anyTraceContainsTokenCount = traces.some((trace) => !isNil(getTraceInfoTotalTokens(trace)));

  // Automatically disabled columns: hide the token count column if there's no trace that contains token count information.
  const autoDisabledColumns = useMemo(
    () => (!anyTraceContainsTokenCount ? [ExperimentViewTracesTableColumns.totalTokens] : []),
    [anyTraceContainsTokenCount],
  );

  // Combine columns that are disabled by parent component and columns that are disabled automatically.
  const allDisabledColumns = useMemo(
    () => [...(disabledColumns ?? []), ...autoDisabledColumns],
    [disabledColumns, autoDisabledColumns],
  );

  const allHiddenColumns = useMemo(
    () => [...(uiState.hiddenColumns ?? []), ...allDisabledColumns],
    [uiState, allDisabledColumns],
  );

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        height: '100%',
        overflow: 'hidden',
      }}
    >
      <TracesViewControls
        experimentIds={experimentIds}
        filter={filter}
        onChangeFilter={setFilter}
        hiddenColumns={uiState.hiddenColumns ?? []}
        disabledColumns={allDisabledColumns}
        toggleHiddenColumn={toggleHiddenColumn}
        rowSelection={rowSelection}
        setRowSelection={setRowSelection}
        refreshTraces={refreshCurrentPage}
      />
      <TracesViewTable
        traces={traces}
        loading={loading}
        error={error}
        onTraceClicked={onTraceClicked}
        onTraceTagsEdit={showEditTagsModalForTrace}
        hasNextPage={hasNextPage}
        hasPreviousPage={hasPreviousPage}
        onPreviousPage={onPreviousPage}
        onNextPage={onNextPage}
        onTagsUpdated={refreshCurrentPage}
        usingFilters={usingFilters}
        onResetFilters={() => setFilter('')}
        hiddenColumns={allHiddenColumns}
        disableTokenColumn={!anyTraceContainsTokenCount}
        setSorting={(sortingSetter) => {
          // If header is clicked enough times, tanstack table switches to "no sort" mode.
          // In that case, we should just reverse the direction of the current sort instead.
          if (isFunction(sortingSetter)) {
            return setSorting((currentState) => {
              const newState = sortingSetter(currentState);
              const currentSortBy = currentState[0];
              if ((!newState || newState.length === 0) && currentSortBy) {
                return [{ id: currentSortBy.id, desc: !currentSortBy.desc }];
              }
              return newState;
            });
          }
        }}
        sorting={sorting}
        rowSelection={rowSelection}
        setRowSelection={setRowSelection}
      />
      {selectedTraceId && (
        <TraceDataDrawer
          traceInfo={selectedTraceInfo}
          loadingTraceInfo={loading}
          requestId={selectedTraceId}
          onClose={() => setSelectedTraceId(undefined)}
        />
      )}
      {EditTagsModal}
    </div>
  );
};
