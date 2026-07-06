/**
 * Shared utilities for deciding whether to auto-render media content inline
 * or show a download link instead. Used by both trace attachment rendering
 * and artifact viewing.
 */

import { Typography } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

// Auto-render size thresholds (bytes). Files exceeding these show a download link.
const MAX_IMAGE_RENDER_BYTES = 10 * 1024 * 1024; // 10 MB
const MAX_AUDIO_RENDER_BYTES = 50 * 1024 * 1024; // 50 MB
const MAX_PDF_RENDER_BYTES = 20 * 1024 * 1024; // 20 MB
const MAX_VIDEO_RENDER_BYTES = 50 * 1024 * 1024; // 50 MB

/**
 * Returns the maximum file size (in bytes) that should be auto-rendered
 * inline for a given content type. Files exceeding this should show a
 * download link instead to prevent browser performance issues.
 *
 * Returns 0 for unknown content types (always show download link).
 */
export function getMaxRenderSize(contentType: string): number {
  if (contentType.startsWith('image/')) return MAX_IMAGE_RENDER_BYTES;
  if (contentType.startsWith('audio/')) return MAX_AUDIO_RENDER_BYTES;
  if (contentType.startsWith('video/')) return MAX_VIDEO_RENDER_BYTES;
  if (contentType === 'application/pdf') return MAX_PDF_RENDER_BYTES;
  return 0;
}

/**
 * Returns true if the content exceeds the auto-render size limit for its type.
 */
export function exceedsRenderSizeLimit(contentType: string, contentLength: number): boolean {
  return contentLength > getMaxRenderSize(contentType);
}

/**
 * Formats a byte count as a human-readable string (e.g., "1.5 MB").
 */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Download link shown when media content exceeds the rendering size limit.
 *
 * When `url` is provided, renders a standard `<a>` download link.
 * When `onFetchDownload` is provided instead, renders a clickable link that
 * fetches the content on demand — used when the blob wasn't pre-fetched
 * because the URI's size param indicated it exceeded the render limit.
 */
export function DownloadLink({
  url,
  contentType,
  contentLength,
  filename,
  onClick,
  onFetchDownload,
}: {
  url?: string | null;
  contentType: string;
  contentLength: number;
  filename?: string;
  onClick?: (e: React.MouseEvent<HTMLAnchorElement>) => void;
  onFetchDownload?: () => Promise<void>;
}) {
  const label = (
    <FormattedMessage
      defaultMessage="Download {contentType} ({size})"
      description="Download link for media content that exceeds the rendering size limit"
      values={{ contentType, size: formatFileSize(contentLength) }}
    />
  );

  if (url) {
    return (
      <a href={url} download={filename} onClick={onClick}>
        {label}
      </a>
    );
  }

  if (onFetchDownload) {
    return (
      <Typography.Link
        componentId="shared.media-rendering-utils.fetch-download"
        onClick={() => void onFetchDownload().catch(() => undefined)}
      >
        {label}
      </Typography.Link>
    );
  }

  return <span>{label}</span>;
}
