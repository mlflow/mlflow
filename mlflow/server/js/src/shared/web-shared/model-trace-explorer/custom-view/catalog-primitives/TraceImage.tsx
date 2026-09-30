import { z } from 'zod';
import { createComponentImplementation, type ReactComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import { FormattedMessage } from '@databricks/i18n';
import { Typography } from '@databricks/design-system';

import { ModelTraceExplorerAttachmentRenderer } from '../../field-renderers/ModelTraceExplorerAttachmentRenderer';
import { parseAttachmentUri } from '../../attachment-utils';
import { asString } from '../catalogPrimitiveUtils';

const TraceImageApi = {
  name: 'TraceImage',
  schema: z
    .object({
      uri: DynamicStringSchema.describe('An mlflow-attachment:// URI for an image stored on the current trace.'),
      title: DynamicStringSchema.describe('Optional heading shown above the image.').optional(),
      weight: z.number().optional(),
    })
    .strict(),
} satisfies ComponentApi;

export const TraceImage: ReactComponentImplementation = createComponentImplementation(TraceImageApi, ({ props }) => {
  const uri = asString(props.uri);
  const title = props.title ? asString(props.title) : '';
  const attachment = parseAttachmentUri(uri);

  if (!attachment || !attachment.contentType.startsWith('image/')) {
    return (
      <Typography.Text color="secondary">
        <FormattedMessage
          defaultMessage="Image unavailable"
          description="Fallback shown when a custom trace view image binding is missing or invalid"
        />
      </Typography.Text>
    );
  }

  return (
    <ModelTraceExplorerAttachmentRenderer
      title={title}
      attachmentId={attachment.attachmentId}
      traceId={attachment.traceId}
      contentType={attachment.contentType}
      size={attachment.size}
    />
  );
});
