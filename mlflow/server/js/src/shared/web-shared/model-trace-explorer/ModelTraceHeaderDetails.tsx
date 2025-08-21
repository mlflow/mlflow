import { Overflow, Tag, TagColors, Typography, useDesignSystemTheme, Tooltip } from '@databricks/design-system';
import { Notification } from '@databricks/design-system';
import { useCallback, useMemo, useState } from 'react';

import { isUserFacingTag, parseJSONSafe } from '@mlflow/mlflow/src/common/utils/TagUtils';
import { useIntl } from 'react-intl';
import { truncateToFirstLineWithMaxLength } from '@mlflow/mlflow/src/common/utils/StringUtils';
import type { ModelTrace, ModelTraceInfoV3 } from './ModelTrace.types';
import {
  getModelTraceId,
  getModelTraceSpanStartTime,
  getModelTraceSpanEndTime,
  getModelTraceSpanParentId,
} from './ModelTraceExplorer.utils';

const BASE_TAG_COMPONENT_ID = 'mlflow.model_trace_explorer.header_details';
const BASE_NOTIFICATION_COMPONENT_ID = 'mlflow.model_trace_explorer.header_details.notification';

const ModelTraceHeaderMetricSection = ({
  label,
  value,
  tagKey,
  color = 'teal',
  getTruncatedLabel,
  getComponentId,
  onCopy,
}: {
  label: string;
  value: string;
  tagKey: string;
  color?: TagColors;
  getTruncatedLabel: (label: string) => string;
  getComponentId: (key: string) => string;
  onCopy: () => void;
}) => {
  const { theme } = useDesignSystemTheme();

  const handleClick = () => {
    navigator.clipboard.writeText(value);
    onCopy();
  };

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'row',
        gap: theme.spacing.sm,
      }}
    >
      <Typography.Text size="md" color="secondary">
        {label}
      </Typography.Text>
      <Tooltip componentId={getComponentId(tagKey)} content={value} maxWidth={400}>
        <Tag componentId={getComponentId(tagKey)} color={color} onClick={handleClick} css={{ cursor: 'pointer' }}>
          {getTruncatedLabel(value)}
        </Tag>
      </Tooltip>
    </div>
  );
};

export const ModelTraceHeaderDetails = ({ modelTrace }: { modelTrace: ModelTrace }) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [showNotification, setShowNotification] = useState(false);

  const tags = Object.entries(modelTrace.info.tags ?? {}).filter(([key]) => isUserFacingTag(key));

  const modelTraceId = getModelTraceId(modelTrace);

  const tokenUsage = parseJSONSafe(
    (modelTrace.info as ModelTraceInfoV3)?.trace_metadata?.['mlflow.trace.tokenUsage'] ?? '{}',
  );

  const totalTokens = useMemo(() => tokenUsage?.total_tokens, [tokenUsage]);

  const spans = modelTrace?.trace_data?.spans ?? modelTrace?.data?.spans;

  const calculateLatency = useCallback((): string | undefined => {
    const rootSpan = spans?.find((span) => !getModelTraceSpanParentId(span));

    if (rootSpan) {
      const startTime = getModelTraceSpanStartTime(rootSpan);
      const endTime = getModelTraceSpanEndTime(rootSpan);

      if (startTime && endTime && endTime > startTime) {
        return `${((endTime - startTime) / 1000000).toFixed(1)}s`;
      }
    }

    return undefined;
  }, [spans]);

  const latency = calculateLatency();

  const idLabel = intl.formatMessage({
    defaultMessage: 'ID',
    description: 'Label for the ID section',
  });

  const tokenCountLabel = intl.formatMessage({
    defaultMessage: 'Token count',
    description: 'Label for the token count section',
  });

  const latencyLabel = intl.formatMessage({
    defaultMessage: 'Latency',
    description: 'Label for the latency section',
  });

  const tagsLabel = intl.formatMessage({ defaultMessage: 'Tags', description: 'Label for the tags section' });

  const notificationSuccessMessage = intl.formatMessage({
    defaultMessage: 'Copied to clipboard',
    description: 'Success message for the notification',
  });

  const getComponentId = useCallback((key: string) => `${BASE_TAG_COMPONENT_ID}.tag-${key}`, []);

  const handleTagClick = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const getTruncatedLabel = (label: string) => truncateToFirstLineWithMaxLength(label, 40);

  const handleCopy = useCallback(() => {
    setShowNotification(true);
    setTimeout(() => setShowNotification(false), 2000);
  }, []);

  return (
    <>
      <div css={{ display: 'flex', flexDirection: 'row', gap: theme.spacing.md, flexWrap: 'wrap' }}>
        {modelTraceId && (
          <ModelTraceHeaderMetricSection
            label={idLabel}
            value={modelTraceId}
            tagKey={modelTraceId}
            color="pink"
            getTruncatedLabel={getTruncatedLabel}
            getComponentId={getComponentId}
            onCopy={handleCopy}
          />
        )}
        {totalTokens && (
          <ModelTraceHeaderMetricSection
            label={tokenCountLabel}
            value={totalTokens.toString()}
            tagKey="token-count"
            color="default"
            getTruncatedLabel={getTruncatedLabel}
            getComponentId={getComponentId}
            onCopy={handleCopy}
          />
        )}
        {latency && (
          <ModelTraceHeaderMetricSection
            label={latencyLabel}
            value={latency}
            tagKey="latency"
            color="default"
            getTruncatedLabel={getTruncatedLabel}
            getComponentId={getComponentId}
            onCopy={handleCopy}
          />
        )}
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexDirection: 'row',
            gap: theme.spacing.sm,
          }}
        >
          <Typography.Text size="md" color="secondary">
            {tagsLabel}
          </Typography.Text>
          <Overflow noMargin>
            {tags.map(([key, value]) => {
              const tagKey = `${key}-${value}`;
              const fullText = `${key}: ${value}`;

              return (
                <Tooltip key={key} componentId={getComponentId(tagKey)} content={fullText}>
                  <Tag
                    componentId={getComponentId(tagKey)}
                    color="teal"
                    onClick={() => {
                      handleTagClick(fullText);
                      handleCopy();
                    }}
                    css={{ cursor: 'pointer' }}
                  >
                    {getTruncatedLabel(`${key}: ${value}`)}
                  </Tag>
                </Tooltip>
              );
            })}
          </Overflow>
        </div>
      </div>

      {showNotification && (
        <Notification.Provider>
          <Notification.Root severity="success" componentId={BASE_NOTIFICATION_COMPONENT_ID}>
            <Notification.Title>{notificationSuccessMessage}</Notification.Title>
          </Notification.Root>
          <Notification.Viewport />
        </Notification.Provider>
      )}
    </>
  );
};
