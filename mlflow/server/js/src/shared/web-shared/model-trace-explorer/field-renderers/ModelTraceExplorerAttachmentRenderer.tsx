import { useMemo, useState } from 'react';

import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  LegacySkeleton,
  Modal,
  Typography,
  useDesignSystemEventComponentCallbacks,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { DownloadLink, exceedsRenderSizeLimit } from '../../media-rendering-utils';
import { useTraceAttachment } from '../hooks/useTraceAttachment';

export const ModelTraceExplorerAttachmentRenderer = ({
  title,
  attachmentId,
  traceId,
  contentType,
  size,
}: {
  title: string;
  attachmentId: string;
  traceId: string;
  contentType: string;
  size?: number;
}) => {
  const { theme } = useDesignSystemTheme();
  const { objectUrl, contentLength, isLoading, error, triggerDownload } = useTraceAttachment({
    traceId,
    attachmentId,
    contentType,
    size,
  });
  const [previewVisible, setPreviewVisible] = useState(false);
  const audioEvents = useMemo(() => [DesignSystemEventProviderAnalyticsEventTypes.OnClick], []);
  const audioEventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Button,
    componentId: 'shared.model-trace-explorer.attachment-audio-play',
    analyticsEvents: audioEvents,
  });
  const downloadEvents = useMemo(() => [DesignSystemEventProviderAnalyticsEventTypes.OnClick], []);
  const downloadEventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Button,
    componentId: 'shared.model-trace-explorer.attachment-download',
    analyticsEvents: downloadEvents,
  });

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

  if (isLoading) {
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
        {objectUrl ? (
          <DownloadLink
            url={objectUrl}
            contentType={contentType}
            contentLength={contentLength}
            filename={`attachment-${attachmentId}`}
          />
        ) : (
          <DownloadLink
            contentType={contentType}
            contentLength={contentLength}
            filename={`attachment-${attachmentId}`}
            onFetchDownload={triggerDownload}
          />
        )}
      </div>
    );
  }

  if (!objectUrl) {
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
          css={{
            maxWidth: '100%',
            maxHeight: 200,
            borderRadius: theme.borders.borderRadiusSm,
            cursor: 'pointer',
            '&:hover': { boxShadow: `0 0 4px ${theme.colors.border}` },
          }}
          onClick={() => setPreviewVisible(true)}
        />
        <Modal
          componentId="shared.model-trace-explorer.attachment-image-preview"
          title=""
          visible={previewVisible}
          onCancel={() => setPreviewVisible(false)}
          onOk={() => setPreviewVisible(false)}
        >
          <img
            src={objectUrl}
            alt={`Attachment ${attachmentId}`}
            css={{ maxWidth: '100%', maxHeight: '70vh', display: 'block' }}
          />
        </Modal>
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
        <audio
          controls
          css={{ width: '100%', maxWidth: 500 }}
          src={objectUrl}
          onPlay={(e) => audioEventContext.onClick(e as any)}
        />
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
      <a href={objectUrl} download={`attachment-${attachmentId}`} onClick={(e) => downloadEventContext.onClick(e)}>
        <FormattedMessage
          defaultMessage="Download attachment ({contentType})"
          description="Download link for trace attachment with unknown content type"
          values={{ contentType }}
        />
      </a>
    </div>
  );
};
