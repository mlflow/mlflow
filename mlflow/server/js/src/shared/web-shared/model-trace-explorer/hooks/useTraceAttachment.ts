import { useCallback, useEffect } from 'react';

import { fetchOrFail } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { exceedsRenderSizeLimit } from '../../media-rendering-utils';
import { useQuery } from '../../query-client/queryClient';
import { fetchAndDownload } from '../attachment-utils';
import { getAjaxUrl } from '../ModelTraceExplorer.request.utils';

const TRACE_ATTACHMENT_QUERY_KEY = 'traceAttachment';

/**
 * Fetches a trace attachment and returns a blob object URL for rendering.
 * Handles blob URL cleanup on unmount or when the attachment changes.
 *
 * When `size` is provided and exceeds the render limit for the content type,
 * the fetch is skipped entirely and only `contentLength` is returned, allowing
 * callers to show a download link without downloading the full blob.
 * A `triggerDownload` callback is provided so users can still download on demand.
 */
export const useTraceAttachment = ({
  traceId,
  attachmentId,
  contentType,
  size,
}: {
  traceId: string;
  attachmentId: string;
  contentType: string;
  size?: number;
}) => {
  // Skip the fetch when we already know the content exceeds the render limit
  const skipFetch = size !== undefined && exceedsRenderSizeLimit(contentType, size);

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
    enabled: !skipFetch,
    refetchOnWindowFocus: false,
  });

  useEffect(() => {
    return () => {
      if (data?.objectUrl) {
        URL.revokeObjectURL(data.objectUrl);
      }
    };
  }, [data?.objectUrl]);

  const triggerDownload = useCallback(
    () => fetchAndDownload(traceId, attachmentId, contentType),
    [traceId, attachmentId, contentType],
  );

  return {
    objectUrl: data?.objectUrl ?? null,
    contentLength: skipFetch ? (size ?? 0) : (data?.contentLength ?? 0),
    isLoading: skipFetch ? false : isLoading,
    error,
    triggerDownload: skipFetch ? triggerDownload : undefined,
  };
};
