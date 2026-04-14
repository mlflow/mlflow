import { useEffect } from 'react';

import { fetchOrFail } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { useQuery } from '../../query-client/queryClient';
import { getAjaxUrl } from '../ModelTraceExplorer.request.utils';

const TRACE_ATTACHMENT_QUERY_KEY = 'traceAttachment';

/**
 * Fetches a trace attachment and returns a blob object URL for rendering.
 * Handles blob URL cleanup on unmount or when the attachment changes.
 */
export const useTraceAttachment = ({
  traceId,
  attachmentId,
  contentType,
}: {
  traceId: string;
  attachmentId: string;
  contentType: string;
}) => {
  const { data, isLoading, error } = useQuery({
    queryKey: [TRACE_ATTACHMENT_QUERY_KEY, traceId, attachmentId],
    queryFn: async () => {
      const url = getAjaxUrl(
        `ajax-api/2.0/mlflow/get-trace-artifact?request_id=${encodeURIComponent(traceId)}&path=${encodeURIComponent(attachmentId)}`,
      );
      const response = await fetchOrFail(url);
      const blob = await response.blob();
      return {
        objectUrl: URL.createObjectURL(new Blob([blob], { type: contentType })),
        contentLength: blob.size,
      };
    },
    refetchOnWindowFocus: false,
  });

  useEffect(() => {
    return () => {
      if (data?.objectUrl) {
        URL.revokeObjectURL(data.objectUrl);
      }
    };
  }, [data?.objectUrl]);

  return {
    objectUrl: data?.objectUrl ?? null,
    contentLength: data?.contentLength ?? 0,
    isLoading,
    error,
  };
};
