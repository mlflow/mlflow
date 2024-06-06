import { useDesignSystemTheme } from '@databricks/design-system';
import { ExperimentViewRunsModeSwitch } from '../experiment-page/components/runs/ExperimentViewRunsModeSwitch';

import { useExperimentTraces } from './hooks/useExperimentTraces';
import { TracesViewTable } from './TracesViewTable';
import { useMemo, useState } from 'react';
import { TraceDataDrawer } from './TraceDataDrawer';
import { useEditExperimentTraceTags } from './hooks/useEditExperimentTraceTags';
import { TracesViewControls } from './TracesViewControls';
import { SortingState } from '@tanstack/react-table';
import { compact, isFunction, isNil, uniq } from 'lodash';
import { useExperimentViewTracesUIState } from './hooks/useExperimentViewTracesUIState';
import { ExperimentViewTracesTableColumns, getTraceInfoTotalTokens } from './TracesView.utils';
import { useActiveExperimentTrace } from './hooks/useActiveExperimentTrace';
import { useShouldShowCombinedRunsTab } from '../experiment-page/hooks/useShouldShowCombinedRunsTab';

const defaultSorting: SortingState = [{ id: ExperimentViewTracesTableColumns.timestampMs, desc: true }];

export const TracesView = ({
  experimentIds,
  runUuid,
  disabledColumns = [],
}: {
  experimentIds: string[];
  runUuid?: string;
  /**
   * Columns that should be disabled in the table.
   * Disabled columns are hidden and are not available to be toggled at all.
   */
  disabledColumns?: ExperimentViewTracesTableColumns[];
}) => {
  const [filter, setFilter] = useState<string>('');
  const [sorting, setSorting] = useState<SortingState>(defaultSorting);

  const [selectedTraceId, setSelectedTraceId] = useActiveExperimentTrace();

  const { traces, loading, error, hasNextPage, hasPreviousPage, fetchNextPage, fetchPrevPage, refreshCurrentPage } =
    useExperimentTraces(experimentIds, sorting, filter, runUuid);

  const { theme } = useDesignSystemTheme();

  // Try to find the trace info for the currently selected trace id
  const selectedTraceInfo = useMemo(() => {
    if (!selectedTraceId) return undefined;
    return traces.find((trace) => trace.request_id === selectedTraceId);
  }, [selectedTraceId, traces]);

  const {
    // hiddenColumns is a list of columns that are hidden by the user.
    uiState: { hiddenColumns: hiddenColumnsByUser = [] },
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
    () => [...disabledColumns, ...autoDisabledColumns],
    [disabledColumns, autoDisabledColumns],
  );

  const allHiddenColumns = useMemo(
    () => [...hiddenColumnsByUser, ...allDisabledColumns],
    [hiddenColumnsByUser, allDisabledColumns],
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
        filter={filter}
        onChangeFilter={setFilter}
        hiddenColumns={hiddenColumnsByUser}
        disabledColumns={allDisabledColumns}
        toggleHiddenColumn={toggleHiddenColumn}
      />
      <TracesViewTable
        traces={traces}
        loading={loading}
        error={error}
        onTraceClicked={({ request_id }) => setSelectedTraceId(request_id)}
        onTraceTagsEdit={showEditTagsModalForTrace}
        hasNextPage={hasNextPage}
        hasPreviousPage={hasPreviousPage}
        onPreviousPage={fetchPrevPage}
        onNextPage={fetchNextPage}
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
