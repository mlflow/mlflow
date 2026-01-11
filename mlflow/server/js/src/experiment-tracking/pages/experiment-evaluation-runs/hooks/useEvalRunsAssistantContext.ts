import { useMemo } from 'react';
import type { RowSelectionState } from '@tanstack/react-table';

import { useRegisterAssistantContext } from '@mlflow/mlflow/src/shared/web-shared/assistant';

/**
 * Hook that registers evaluation runs context with the assistant.
 * Should be called in ExperimentEvaluationRunsPageImpl.
 *
 * @param experimentId - The current experiment ID
 * @param selectedRunUuid - The currently open/selected run UUID
 * @param rowSelection - The row selection state from the table
 */
export const useEvalRunsAssistantContext = (
  experimentId: string,
  selectedRunUuid: string | undefined,
  rowSelection: RowSelectionState,
) => {
  // Extract selected run IDs from row selection state
  const selectedRunIds = useMemo(() => {
    const ids = Object.keys(rowSelection).filter((id) => rowSelection[id]);
    return ids.length > 0 ? ids : undefined;
  }, [rowSelection]);

  // Register context values with the assistant
  useRegisterAssistantContext('experimentId', experimentId);
  useRegisterAssistantContext('runId', selectedRunUuid);
  useRegisterAssistantContext('selectedRunIds', selectedRunIds);
};
