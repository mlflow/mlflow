import { useState } from 'react';

import { LegacySkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { exceedsRenderSizeLimit, formatFileSize } from '../../media-rendering-utils';
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
  const { objectUrl, contentLength, isLoading, error } = useTraceAttachment({
    traceId,
    attachmentId,
    contentType,
  });
  const [previewVisible, setPreviewVisible] = useState(false);

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

  const exceedsRenderLimit = exceedsRenderSizeLimit(contentType, contentLength);

  if (exceedsRenderLimit) {
    return (
      <div css={{ padding: theme.spacing.sm }}>
        {title && (
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
            {title}
          </Typography.Text>
        )}
        <a href={objectUrl} download={`attachment-${attachmentId}`}>
          <FormattedMessage
            defaultMessage="Download {contentType} ({size})"
            description="Download link for trace attachment that exceeds the rendering size limit"
            values={{ contentType, size: formatFileSize(contentLength) }}
          />
        </a>
      </div>
    );
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
          css={{
            maxWidth: '100%',
            maxHeight: 200,
            borderRadius: theme.borders.borderRadiusSm,
            cursor: 'pointer',
            '&:hover': { boxShadow: `0 0 4px ${theme.colors.border}` },
          }}
          onClick={() => setPreviewVisible(true)}
        />
        {previewVisible && (
          <div
            css={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: 'rgba(0, 0, 0, 0.7)',
              zIndex: 1000,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
            }}
            onClick={() => setPreviewVisible(false)}
          >
            <img
              src={objectUrl}
              alt={`Attachment ${attachmentId}`}
              css={{ maxWidth: '90vw', maxHeight: '90vh', borderRadius: 4 }}
            />
          </div>
        )}
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
        <audio controls css={{ width: '100%', maxWidth: 500 }} src={objectUrl} />
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
