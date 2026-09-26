import { useCallback, useEffect, useState } from 'react';

import { getArtifactProxyDownloadUrl, isEligibleArtifactProxyUri } from '@mlflow/mlflow/src/common/utils/artifactProxy';
import { useTraceArtifactLocation } from './contexts/TraceArtifactLocationContext';
import { exceedsRenderSizeLimit } from '../media-rendering-utils';
import { fetchOrFail, getAjaxUrl } from './ModelTraceExplorer.request.utils';

/** Tag holding the artifact root that the server itself resolves trace attachments against. */
const MLFLOW_ARTIFACT_LOCATION_TAG = 'mlflow.artifactLocation';

/** Server-side prefix under the trace artifact root where attachments are stored. */
const TRACE_ATTACHMENT_PATH_PREFIX = 'attachments';

/**
 * Reads `mlflow.artifactLocation` off a trace info. Trace tags arrive either as a
 * `{key, value}` list (tracking API) or as a plain object (notebook / V3 payloads).
 */
export function getTraceArtifactLocation(
  traceInfo?: { tags?: { key: string; value: string }[] | { [key: string]: string } } | null,
): string | undefined {
  const tags = traceInfo?.tags;
  if (!tags) {
    return undefined;
  }
  if (Array.isArray(tags)) {
    return tags.find((tag) => tag.key === MLFLOW_ARTIFACT_LOCATION_TAG)?.value;
  }
  return tags[MLFLOW_ARTIFACT_LOCATION_TAG];
}

/**
 * Builds the URL used to fetch a single trace attachment.
 *
 * When the trace's stored artifact location is an eligible MLflow artifact-proxy URI, the
 * attachment is read straight from it, mirroring how the server resolves the same request
 * (`get_trace_artifact_handler` joins `attachments/<path>` onto the trace artifact repo).
 * Otherwise this falls back to the tracking endpoint.
 */
export function getTraceAttachmentUrl(traceId: string, attachmentId: string, traceArtifactLocation?: string): string {
  if (isEligibleArtifactProxyUri(traceArtifactLocation)) {
    return getArtifactProxyDownloadUrl(traceArtifactLocation, `${TRACE_ATTACHMENT_PATH_PREFIX}/${attachmentId}`);
  }
  return getAjaxUrl(
    `ajax-api/2.0/mlflow/get-trace-artifact?request_id=${encodeURIComponent(traceId)}&path=${encodeURIComponent(attachmentId)}`,
  );
}

async function getTraceAttachment(
  requestId: string,
  attachmentId: string,
  traceArtifactLocation?: string,
): Promise<ArrayBuffer | undefined> {
  try {
    const response = await fetchOrFail(getTraceAttachmentUrl(requestId, attachmentId, traceArtifactLocation));
    return await response.arrayBuffer();
  } catch {
    return undefined;
  }
}

/**
 * Programmatically fetches a blob and triggers a browser download.
 */
export async function fetchAndDownload(
  traceId: string,
  attachmentId: string,
  contentType: string,
  traceArtifactLocation?: string,
) {
  const url = getTraceAttachmentUrl(traceId, attachmentId, traceArtifactLocation);
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
  const traceArtifactLocation = useTraceArtifactLocation(parsed?.traceId);

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

    getTraceAttachment(parsed.traceId, parsed.attachmentId, traceArtifactLocation).then(
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
  }, [uri, traceArtifactLocation]);

  const triggerDownload = useCallback(() => {
    if (parsed) {
      return fetchAndDownload(parsed.traceId, parsed.attachmentId, parsed.contentType, traceArtifactLocation);
    }
    return Promise.resolve();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uri, traceArtifactLocation]);

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
