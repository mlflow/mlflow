import { useEffect, useState } from 'react';

import { Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { getTraceAttachment } from '../oss-notebook-renderer/mlflow-fetch-utils';

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
  const [mediaUrl, setMediaUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let revoked = false;
    getTraceAttachment(traceId, attachmentId).then((data) => {
      if (revoked) {
        return;
      }
      if (data) {
        const url = URL.createObjectURL(new Blob([data], { type: contentType }));
        setMediaUrl(url);
      } else {
        setError('Failed to load attachment');
      }
    });
    return () => {
      revoked = true;
      if (mediaUrl) {
        URL.revokeObjectURL(mediaUrl);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attachmentId, traceId, contentType]);

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

  if (!mediaUrl) {
    return <Spinner size="small" />;
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
          src={mediaUrl}
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
          <source src={mediaUrl} type={contentType} />
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
          src={mediaUrl}
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
      <a href={mediaUrl} download={`attachment-${attachmentId}`}>
        <FormattedMessage
          defaultMessage="Download attachment ({contentType})"
          description="Download link for trace attachment with unknown content type"
          values={{ contentType }}
        />
      </a>
    </div>
  );
};
