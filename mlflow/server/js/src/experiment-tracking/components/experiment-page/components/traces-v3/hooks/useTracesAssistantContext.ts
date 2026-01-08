import { useMemo } from 'react';

import { useRegisterAssistantContext } from '@mlflow/mlflow/src/shared/web-shared/assistant';
import { useActiveEvaluation } from '@mlflow/mlflow/src/shared/web-shared/genai-traces-table/hooks/useActiveEvaluation';
import { useGenAiTraceTableRowSelection } from '@mlflow/mlflow/src/shared/web-shared/genai-traces-table/hooks/useGenAiTraceTableRowSelection';

/**
 * Hook that registers trace-related context with the assistant.
 * Should be called in TracesV3ViewImpl or similar component.
 *
 * @param experimentId - The current experiment ID
 */
export const useTracesAssistantContext = (experimentId: string) => {
  // Get the active trace (from drawer)
  const [activeTraceId] = useActiveEvaluation();

  // Get selected traces (from table)
  const { rowSelection } = useGenAiTraceTableRowSelection();

  // Extract selected trace IDs from row selection state
  const selectedTraceIds = useMemo(() => {
    const ids = Object.keys(rowSelection).filter((id) => rowSelection[id]);
    return ids.length > 0 ? ids : undefined;
  }, [rowSelection]);

  // Register context values with the assistant
  // experimentId is also auto-extracted from URL, but we register explicitly for clarity
  useRegisterAssistantContext('experimentId', experimentId);
  // traceId may differ from URL if drawer is opened programmatically
  useRegisterAssistantContext('traceId', activeTraceId);
  // Selection state is only available via registration (not in URL)
  useRegisterAssistantContext('selectedTraceIds', selectedTraceIds);
};
