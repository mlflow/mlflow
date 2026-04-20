import { useCallback, useEffect, useState } from 'react';

import { exceedsRenderSizeLimit } from '../media-rendering-utils';
import { fetchOrFail, getAjaxUrl } from './ModelTraceExplorer.request.utils';

async function getTraceAttachment(requestId: string, attachmentId: string): Promise<ArrayBuffer | undefined> {
  try {
    const url = getAjaxUrl(
      `ajax-api/2.0/mlflow/get-trace-artifact?request_id=${encodeURIComponent(requestId)}&path=${encodeURIComponent(attachmentId)}`,
    );
    const response = await fetchOrFail(url);
    return await response.arrayBuffer();
  } catch {
    return undefined;
  }
}

/**
 * Programmatically fetches a blob and triggers a browser download.
 */
export async function fetchAndDownload(traceId: string, attachmentId: string, contentType: string) {
  const url = getAjaxUrl(
    `ajax-api/2.0/mlflow/get-trace-artifact?request_id=${encodeURIComponent(traceId)}&path=${encodeURIComponent(attachmentId)}`,
  );
  const response = await fetchOrFail(url);
  const arrayBuffer = await response.arrayBuffer();
  const blob = new Blob([arrayBuffer], { type: contentType });
  const blobUrl = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = blobUrl;
  a.download = `attachment-${attachmentId}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(blobUrl);
}

export function parseAttachmentUri(
  uri: string,
): { attachmentId: string; traceId: string; contentType: string; size?: number } | null {
  try {
    const parsed = new URL(uri);
    if (parsed.protocol !== 'mlflow-attachment:') {
      return null;
    }
    const attachmentId = parsed.hostname;
    const contentType = parsed.searchParams.get('content_type');
    const traceId = parsed.searchParams.get('trace_id');
    if (!attachmentId || !contentType || !traceId) {
      return null;
    }
    const sizeStr = parsed.searchParams.get('size');
    const parsedSize = sizeStr ? Number(sizeStr) : undefined;
    const size =
      parsedSize !== undefined && Number.isFinite(parsedSize) && Number.isInteger(parsedSize) && parsedSize > 0
        ? parsedSize
        : undefined;
    return { attachmentId, contentType, traceId, ...(size !== undefined ? { size } : {}) };
  } catch {
    return null;
  }
}

/**
 * Hook that fetches an attachment by URI and returns a blob URL for rendering.
 * Handles cleanup of blob URLs on unmount or when the URI changes.
 */
export function useAttachmentUrl(uri: string | null): {
  url: string | null;
  contentLength: number;
  contentType: string | null;
  loading: boolean;
  error: boolean;
  triggerDownload?: () => Promise<void>;
} {
  const parsed = uri ? parseAttachmentUri(uri) : null;

  // If the URI encodes a size that exceeds the render limit, skip the fetch entirely
  // and let callers show a download link immediately.
  const skipFetch = Boolean(parsed?.size !== undefined && exceedsRenderSizeLimit(parsed.contentType, parsed.size));

  const [url, setUrl] = useState<string | null>(null);
  const [contentLength, setContentLength] = useState(skipFetch && parsed?.size ? parsed.size : 0);
  const [loading, setLoading] = useState(Boolean(parsed) && !skipFetch);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!parsed || skipFetch) {
      setUrl(null);
      setContentLength(skipFetch && parsed?.size ? parsed.size : 0);
      setLoading(false);
      setError(false);
      return;
    }

    let revoked = false;
    let localUrl: string | null = null;
    setLoading(true);
    setError(false);
    setUrl(null);

    getTraceAttachment(parsed.traceId, parsed.attachmentId).then(
      (data) => {
        if (revoked) {
          return;
        }
        if (data) {
          const blob = new Blob([data], { type: parsed.contentType });
          const blobUrl = URL.createObjectURL(blob);
          localUrl = blobUrl;
          setUrl(blobUrl);
          setContentLength(blob.size);
        } else {
          setError(true);
        }
        setLoading(false);
      },
      () => {
        if (!revoked) {
          setError(true);
          setLoading(false);
        }
      },
    );

    return () => {
      revoked = true;
      if (localUrl) {
        URL.revokeObjectURL(localUrl);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uri]);

  const triggerDownload = useCallback(() => {
    if (parsed) {
      return fetchAndDownload(parsed.traceId, parsed.attachmentId, parsed.contentType);
    }
    return Promise.resolve();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uri]);

  return {
    url,
    loading,
    error,
    contentLength,
    contentType: parsed?.contentType ?? null,
    triggerDownload: skipFetch ? triggerDownload : undefined,
  };
}

export function isAttachmentUri(value: string): boolean {
  return value.startsWith('mlflow-attachment://');
}

export function containsAttachmentUri(value: string): boolean {
  return value.includes('mlflow-attachment://');
}

/**
 * URL transform for react-markdown that preserves mlflow-attachment:// URIs.
 * Without this, the default transform strips non-standard protocols.
 * Falls back to default sanitization for all other URLs to prevent XSS.
 */
export function attachmentAwareUrlTransform(url: string): string {
  if (url.startsWith('mlflow-attachment:')) {
    return url;
  }
  if (url.startsWith('data:')) {
    return url;
  }
  if (url.startsWith('blob:')) {
    return url;
  }
  // Delegate to default sanitization for all other URLs (blocks javascript: etc.)
  if (url.startsWith('http:') || url.startsWith('https:') || url.startsWith('mailto:')) {
    return url;
  }
  return '';
}
