import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import { useDesignSystemTheme } from '@databricks/design-system';

import { ModelTraceExplorerAttachmentRenderer } from '../field-renderers/ModelTraceExplorerAttachmentRenderer';
import { parseAttachmentUri } from '../field-renderers/attachment-utils';

/**
 * Schema (API) for our custom MediaRenderer. It renders trace media from a
 * single `url`:
 *
 *  - `mlflow-attachment://` URIs are fetched from the trace artifact store and
 *    dispatched by content type — `image/*` (with a click-to-expand preview),
 *    `audio/*` (an inline player), and `application/pdf` (an embedded viewer),
 *    with a download fallback for blobs over the size limit. This is how MLflow
 *    surfaces extracted media: images, audio (always base64 `input_audio` ->
 *    attachment), and PDFs all arrive as attachment URIs.
 *  - `http(s)://` URLs and `data:` URIs are loaded directly by the browser.
 *    Only images appear in this form (audio is never a plain URL in a trace),
 *    so the direct branch renders an `<img>`.
 */
export const MediaRendererApi = {
  name: 'MediaRenderer',
  schema: z
    .object({
      url: DynamicStringSchema.describe(
        'The media source. Accepts an http(s):// URL, a data: URI, or an mlflow-attachment:// URI (fetched from the trace artifact store and rendered as a blob; image/audio/PDF are dispatched by content type).',
      ),
      alt: z.string().describe('Alternative text describing the media.').optional(),
      weight: z.number().describe('Relative flex weight when placed directly inside a Row/Column.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

export const MediaRenderer = createComponentImplementation(MediaRendererApi, ({ props }) => {
  const { theme } = useDesignSystemTheme();

  const url = typeof props.url === 'string' ? props.url : String(props.url ?? '');
  const alt = typeof props.alt === 'string' ? props.alt : '';
  const weight = typeof props.weight === 'number' ? props.weight : undefined;

  const wrapperStyle = weight !== undefined ? { flex: `${weight}`, minWidth: 0, minHeight: 0 } : undefined;

  // mlflow-attachment:// URIs are self-describing (they carry attachmentId,
  // traceId, contentType, and size), so we can hand them straight to the shared
  // renderer, which fetches the blob, dispatches by content type (image/audio/
  // PDF), and handles the size-limit download fallback and preview modal.
  const attachment = parseAttachmentUri(url);
  if (attachment) {
    return (
      <div css={wrapperStyle}>
        <ModelTraceExplorerAttachmentRenderer
          title={alt}
          attachmentId={attachment.attachmentId}
          traceId={attachment.traceId}
          contentType={attachment.contentType}
          size={attachment.size}
        />
      </div>
    );
  }

  // Direct http(s):// URLs and data: URIs can be loaded by the browser as-is.
  // Only images appear as direct URLs in a trace, so we render an <img>.
  return (
    <div css={wrapperStyle}>
      <img
        src={url}
        alt={alt}
        css={{
          display: 'block',
          maxWidth: '100%',
          borderRadius: theme.borders.borderRadiusSm,
        }}
      />
    </div>
  );
});
