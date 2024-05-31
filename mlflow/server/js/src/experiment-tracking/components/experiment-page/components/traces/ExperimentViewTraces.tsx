import { useDesignSystemTheme } from '@databricks/design-system';
import { ExperimentViewRunsModeSwitch } from '../runs/ExperimentViewRunsModeSwitch';

import { useExperimentTraces } from './hooks/useExperimentTraces';
import { ExperimentViewTracesTable } from './ExperimentViewTracesTable';
import { useMemo, useState } from 'react';
import { ExperimentViewTraceDataDrawer } from './ExperimentViewTraceDataDrawer';
import { useEditExperimentTraceTags } from './hooks/useEditExperimentTraceTags';
import { ExperimentViewTracesControls } from './ExperimentViewTracesControls';
import { SortingState } from '@tanstack/react-table';
import { compact, isFunction, isNil, uniq } from 'lodash';
import { useExperimentViewTracesUIState } from './hooks/useExperimentViewTracesUIState';
import { ExperimentViewTracesTableColumns, getTraceInfoTotalTokens } from './ExperimentViewTraces.utils';
import { useActiveExperimentTrace } from './hooks/useActiveExperimentTrace';

const defaultSorting: SortingState = [{ id: ExperimentViewTracesTableColumns.timestampMs, desc: true }];

export const ExperimentViewTraces = ({ experimentIds }: { experimentIds: string[] }) => {
  const [filter, setFilter] = useState<string>('');
  const [sorting, setSorting] = useState<SortingState>(defaultSorting);

  const [selectedTraceId, setSelectedTraceId] = useActiveExperimentTrace();

  const { traces, loading, error, hasNextPage, hasPreviousPage, fetchNextPage, fetchPrevPage, refreshCurrentPage } =
    useExperimentTraces(experimentIds, sorting, filter);

  const { theme } = useDesignSystemTheme();

  // Try to find the trace info for the currently selected trace id
  const selectedTraceInfo = useMemo(() => {
    if (!selectedTraceId) return undefined;
    return traces.find((trace) => trace.request_id === selectedTraceId);
  }, [selectedTraceId, traces]);

  const {
    uiState: { hiddenColumns },
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
  const disabledColumns = useMemo(
    () => (!anyTraceContainsTokenCount ? [ExperimentViewTracesTableColumns.totalTokens] : []),
    [anyTraceContainsTokenCount],
  );

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        marginTop: theme.spacing.md,
        height: '100%',
        overflow: 'hidden',
      }}
    >
      <ExperimentViewTracesControls
        filter={filter}
        onChangeFilter={setFilter}
        hiddenColumns={hiddenColumns}
        disabledColumns={disabledColumns}
        toggleHiddenColumn={toggleHiddenColumn}
      />
      <ExperimentViewRunsModeSwitch hideBorder={false} />
      <ExperimentViewTracesTable
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
        hiddenColumns={hiddenColumns}
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
        <ExperimentViewTraceDataDrawer
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
