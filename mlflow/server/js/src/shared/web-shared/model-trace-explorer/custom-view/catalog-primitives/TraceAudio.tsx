import { z } from 'zod';
import { createComponentImplementation, type ReactComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import { FormattedMessage } from '@databricks/i18n';
import { Typography } from '@databricks/design-system';

import { ModelTraceExplorerAttachmentRenderer } from '../../field-renderers/ModelTraceExplorerAttachmentRenderer';
import { parseAttachmentUri } from '../../attachment-utils';
import { asString } from '../catalogPrimitiveUtils';

const TraceAudioApi = {
  name: 'TraceAudio',
  schema: z
    .object({
      uri: DynamicStringSchema.describe('An mlflow-attachment:// URI for an audio clip stored on the current trace.'),
      title: DynamicStringSchema.describe('Optional heading shown above the audio player.').optional(),
      weight: z.number().describe('Relative flex weight when placed directly inside a Row/Column.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

export const TraceAudio: ReactComponentImplementation = createComponentImplementation(TraceAudioApi, ({ props }) => {
  const uri = asString(props.uri);
  const title = props.title ? asString(props.title) : '';
  const weight = typeof props.weight === 'number' ? props.weight : undefined;
  const flexStyle = weight !== undefined ? { flex: `${weight}`, minWidth: 0 } : undefined;
  const attachment = parseAttachmentUri(uri);

  if (!attachment || !attachment.contentType.startsWith('audio/')) {
    return (
      <Typography.Text color="secondary" css={flexStyle}>
        <FormattedMessage
          defaultMessage="Audio unavailable"
          description="Fallback shown when a custom trace view audio binding is missing or invalid"
        />
      </Typography.Text>
    );
  }

  return (
    <div css={flexStyle}>
      <ModelTraceExplorerAttachmentRenderer
        title={title}
        attachmentId={attachment.attachmentId}
        traceId={attachment.traceId}
        contentType={attachment.contentType}
        size={attachment.size}
      />
    </div>
  );
});
