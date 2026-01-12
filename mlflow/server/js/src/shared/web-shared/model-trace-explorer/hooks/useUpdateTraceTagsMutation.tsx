import { useMutation } from '@databricks/web-shared/query-client';

import { shouldUseTracesV4API } from '../FeatureUtils';
import type { ModelTrace } from '../ModelTrace.types';
import { doesTraceSupportV4API, isV3ModelTraceInfo } from '../ModelTraceExplorer.utils';
import { TracesServiceV3, TracesServiceV4 } from '../api';

/**
 * Updates (sets+deletes) tags for a given trace.
 * Supports both v3 and v4 API endpoints based on the provided trace info.
 */
export const useUpdateTraceTagsMutation = ({ onSuccess }: { onSuccess?: () => void }) => {
  return useMutation({
    mutationFn: async ({
      deletedTags,
      modelTraceInfo,
      newTags,
    }: {
      newTags: {
        key: string;
        value: string;
      }[];
      deletedTags: {
        key: string;
        value: string;
      }[];
      modelTraceInfo: ModelTrace['info'];
    }) => {
      // Use v4 API endpoints if conditions are met
      // TODO: Squash all this logic into a single util function (in both model-trace-explorer and genai-traces-table)
      if (
        shouldUseTracesV4API() &&
        isV3ModelTraceInfo(modelTraceInfo) &&
        modelTraceInfo.trace_location &&
        doesTraceSupportV4API(modelTraceInfo)
      ) {
        const creationPromises = newTags.map((tag) =>
          TracesServiceV4.setTraceTagV4({
            tag,
            traceLocation: modelTraceInfo.trace_location,
            traceId: modelTraceInfo.trace_id,
          }),
        );
        const deletionPromises = deletedTags.map((tag) =>
          TracesServiceV4.deleteTraceTagV4({
            tagKey: tag.key,
            traceLocation: modelTraceInfo.trace_location,
            traceId: modelTraceInfo.trace_id,
          }),
        );
        return Promise.all([...creationPromises, ...deletionPromises]);
      }

      const traceId = isV3ModelTraceInfo(modelTraceInfo) ? modelTraceInfo.trace_id : (modelTraceInfo.request_id ?? '');
      // Otherwise, fallback to v3 API endpoints
      const creationPromises = newTags.map((tag) =>
        TracesServiceV3.setTraceTagV3({
          tag,
          traceId,
        }),
      );
      const deletionPromises = deletedTags.map((tag) =>
        TracesServiceV3.deleteTraceTagV3({
          tagKey: tag.key,
          traceId,
        }),
      );
      return Promise.all([...creationPromises, ...deletionPromises]);
    },
    onSuccess,
  });
};
