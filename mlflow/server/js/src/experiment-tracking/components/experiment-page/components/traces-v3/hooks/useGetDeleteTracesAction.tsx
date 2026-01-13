import type { TraceActions } from '@databricks/web-shared/genai-traces-table';
import { useMemo } from 'react';
import { useDeleteTracesMutation } from '../../../../evaluations/hooks/useDeleteTraces';
import type { ModelTraceLocation } from '@databricks/web-shared/model-trace-explorer';

export const useGetDeleteTracesAction = ({
  traceSearchLocations,
  reviewAppId,
  labelingSessionId,
  traceIdToItemIdMap,
}: {
  traceSearchLocations: ModelTraceLocation[];
  reviewAppId?: string;
  labelingSessionId?: string;
  traceIdToItemIdMap?: Map<string, string>;
}) => {
  const deleteTracesMutation = useDeleteTracesMutation();

  const deleteTracesAction: TraceActions['deleteTracesAction'] = useMemo(
    () => {
      return {
        deleteTraces: async (experimentId: string, traceIds: string[]) =>
          deleteTracesMutation.mutateAsync({
            experimentId,
            traceRequestIds: traceIds,
          }),
      };
    },
    // prettier-ignore
    [
      deleteTracesMutation,
    ],
  );
  return deleteTracesAction;
};
