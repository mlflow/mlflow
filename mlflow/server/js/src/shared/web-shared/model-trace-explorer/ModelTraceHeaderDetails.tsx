import {
  Overflow,
  Tag,
  TagColors,
  Typography,
  useDesignSystemTheme,
  Tooltip,
  ClockIcon,
  SpeechBubbleIcon,
  UserIcon,
} from '@databricks/design-system';
import { Notification } from '@databricks/design-system';
import { useCallback, useMemo, useState } from 'react';

import { FormattedMessage, useIntl } from 'react-intl';
import type { ModelTrace, ModelTraceInfoV3 } from './ModelTrace.types';
import { getModelTraceId } from './ModelTraceExplorer.utils';
import { spanTimeFormatter } from './timeline-tree/TimelineTree.utils';
import { useModelTraceExplorerViewState } from './ModelTraceExplorerViewStateContext';
import { isUserFacingTag, parseJSONSafe, truncateToFirstLineWithMaxLength } from './TagUtils';
import { MLFLOW_TRACE_SESSION_KEY, MLFLOW_TRACE_TOKEN_USAGE_KEY, MLFLOW_TRACE_USER_KEY } from './ModelTrace.types';
import { ModelTraceHeaderMetadataPill } from './ModelTraceHeaderMetadataPill';

const BASE_TAG_COMPONENT_ID = 'mlflow.model_trace_explorer.header_details';
const BASE_NOTIFICATION_COMPONENT_ID = 'mlflow.model_trace_explorer.header_details.notification';

const ModelTraceHeaderMetricSection = ({
  label,
  value,
  icon,
  tagKey,
  color = 'teal',
  getTruncatedLabel,
  getComponentId,
  onCopy,
}: {
  label: React.ReactNode;
  value: string;
  icon?: React.ReactNode;
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
          <span css={{ display: 'flex', flexDirection: 'row', alignItems: 'center', gap: theme.spacing.xs }}>
            {icon && <span>{icon}</span>}
            <span>{getTruncatedLabel(value)}</span>
          </span>
        </Tag>
      </Tooltip>
    </div>
  );
};

export const ModelTraceHeaderDetails = ({ modelTrace }: { modelTrace: ModelTrace }) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [showNotification, setShowNotification] = useState(false);
  const { rootNode } = useModelTraceExplorerViewState();

  const tags = Object.entries(modelTrace.info.tags ?? {}).filter(([key]) => isUserFacingTag(key));

  const modelTraceId = getModelTraceId(modelTrace);

  const tokenUsage = useMemo(() => {
    const tokenUsage = parseJSONSafe(
      (modelTrace.info as ModelTraceInfoV3)?.trace_metadata?.[MLFLOW_TRACE_TOKEN_USAGE_KEY] ?? '{}',
    );
    return tokenUsage;
  }, [modelTrace.info]);

  const totalTokens = useMemo(() => tokenUsage?.total_tokens, [tokenUsage]);

  const latency = useMemo((): string | undefined => {
    if (rootNode) {
      return spanTimeFormatter(rootNode.end - rootNode.start);
    }

    return undefined;
  }, [rootNode]);

  const sessionId = useMemo(() => {
    return (modelTrace.info as ModelTraceInfoV3)?.trace_metadata?.[MLFLOW_TRACE_SESSION_KEY];
  }, [modelTrace.info]);

  const userId = useMemo(() => {
    return (modelTrace.info as ModelTraceInfoV3)?.trace_metadata?.[MLFLOW_TRACE_USER_KEY];
  }, [modelTrace.info]);

  const getComponentId = useCallback((key: string) => `${BASE_TAG_COMPONENT_ID}.tag-${key}`, []);

  const handleTagClick = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const getTruncatedLabel = (label: string) => truncateToFirstLineWithMaxLength(label, 40);
  const getTruncatedSessionLabel = (label: string) => (label.length > 10 ? `${label.slice(0, 10)}...` : label);

  const handleCopy = useCallback(() => {
    setShowNotification(true);
    setTimeout(() => setShowNotification(false), 2000);
  }, []);

  return (
    <>
      <div css={{ display: 'flex', flexDirection: 'row', gap: theme.spacing.md, flexWrap: 'wrap' }}>
        {modelTraceId && (
          <ModelTraceHeaderMetricSection
            label={<FormattedMessage defaultMessage="ID" description="Label for the ID section" />}
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
            label={<FormattedMessage defaultMessage="Token count" description="Label for the token count section" />}
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
            label={<FormattedMessage defaultMessage="Latency" description="Label for the latency section" />}
            icon={<ClockIcon css={{ fontSize: 12, display: 'flex', alignItems: 'center', justifyContent: 'center' }} />}
            value={latency}
            tagKey="latency"
            color="default"
            getTruncatedLabel={getTruncatedLabel}
            getComponentId={getComponentId}
            onCopy={handleCopy}
          />
        )}
        {sessionId && (
          <ModelTraceHeaderMetricSection
            label={<FormattedMessage defaultMessage="Session ID" description="Label for the session id section" />}
            icon={<SpeechBubbleIcon css={{ fontSize: 12, display: 'flex' }} />}
            value={sessionId}
            tagKey="session"
            color="default"
            getTruncatedLabel={getTruncatedSessionLabel}
            getComponentId={getComponentId}
            onCopy={handleCopy}
          />
        )}
        {userId && (
          <ModelTraceHeaderMetricSection
            label={<FormattedMessage defaultMessage="User" description="Label for the user id section" />}
            icon={<UserIcon css={{ fontSize: 12, display: 'flex' }} />}
            value={userId}
            tagKey="user"
            color="default"
            getTruncatedLabel={getTruncatedLabel}
            getComponentId={getComponentId}
            onCopy={handleCopy}
          />
        )}
        <ModelTraceHeaderMetadataPill
          traceMetadata={(modelTrace.info as ModelTraceInfoV3)?.trace_metadata}
          getTruncatedLabel={getTruncatedLabel}
          getComponentId={getComponentId}
        />
        {tags.length > 0 && (
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
              <FormattedMessage defaultMessage="Tags" description="Label for the tags section" />
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
        )}
      </div>

      {showNotification && (
        <Notification.Provider>
          <Notification.Root severity="success" componentId={BASE_NOTIFICATION_COMPONENT_ID}>
            <Notification.Title>
              <FormattedMessage
                defaultMessage="Copied to clipboard"
                description="Success message for the notification"
              />
            </Notification.Title>
          </Notification.Root>
          <Notification.Viewport />
        </Notification.Provider>
      )}
    </>
  );
};
