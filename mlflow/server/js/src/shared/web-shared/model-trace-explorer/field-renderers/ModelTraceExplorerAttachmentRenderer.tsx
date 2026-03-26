import { LegacySkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useTraceAttachment } from '../hooks/useTraceAttachment';

export const ModelTraceExplorerAttachmentRenderer = ({
  title,
  attachmentId,
  traceId,
  contentType,
}: {
  title: string;
  attachmentId: string;
  traceId: string;
  contentType: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const { objectUrl, isLoading, error } = useTraceAttachment({ traceId, attachmentId, contentType });

  if (error) {
    return (
      <Typography.Text color="error">
        <FormattedMessage
          defaultMessage="Failed to load attachment"
          description="Error message when trace attachment fails to load"
        />
      </Typography.Text>
    );
  }

  if (isLoading || !objectUrl) {
    return <LegacySkeleton />;
  }

  if (contentType.startsWith('image/')) {
    return (
      <div css={{ padding: theme.spacing.sm }}>
        {title && (
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
            {title}
          </Typography.Text>
        )}
        <img
          src={objectUrl}
          alt={`Attachment ${attachmentId}`}
          css={{ maxWidth: '100%', maxHeight: 400, borderRadius: theme.borders.borderRadiusSm }}
        />
      </div>
    );
  }

  if (contentType.startsWith('audio/')) {
    return (
      <div css={{ padding: theme.spacing.sm }}>
        {title && (
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
            {title}
          </Typography.Text>
        )}
        {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
        <audio controls>
          <source src={objectUrl} type={contentType} />
        </audio>
      </div>
    );
  }

  if (contentType === 'application/pdf') {
    return (
      <div css={{ padding: theme.spacing.sm }}>
        {title && (
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
            {title}
          </Typography.Text>
        )}
        <iframe
          src={objectUrl}
          title={`PDF Attachment ${attachmentId}`}
          css={{
            width: '100%',
            height: 600,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusSm,
          }}
        />
      </div>
    );
  }

  return (
    <div css={{ padding: theme.spacing.sm }}>
      <a href={objectUrl} download={`attachment-${attachmentId}`}>
        <FormattedMessage
          defaultMessage="Download attachment ({contentType})"
          description="Download link for trace attachment with unknown content type"
          values={{ contentType }}
        />
      </a>
    </div>
  );
};
