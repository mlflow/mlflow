import { useEffect, useState } from 'react';

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

export function parseAttachmentUri(uri: string): { attachmentId: string; traceId: string; contentType: string } | null {
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
    return { attachmentId, contentType, traceId };
  } catch {
    return null;
  }
}

/**
 * Hook that fetches an attachment by URI and returns a blob URL for rendering.
 * Handles cleanup of blob URLs on unmount or when the URI changes.
 */
export function useAttachmentUrl(uri: string | null): { url: string | null; loading: boolean; error: boolean } {
  const parsed = uri ? parseAttachmentUri(uri) : null;
  const [url, setUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(!!parsed);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!parsed) {
      setUrl(null);
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
          const blobUrl = URL.createObjectURL(new Blob([data], { type: parsed.contentType }));
          localUrl = blobUrl;
          setUrl(blobUrl);
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

  return { url, loading, error };
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
