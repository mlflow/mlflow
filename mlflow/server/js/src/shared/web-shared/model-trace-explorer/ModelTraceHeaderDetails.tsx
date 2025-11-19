import { useCallback, useMemo, useState } from 'react';

import {
  Overflow,
  Tag,
  Typography,
  useDesignSystemTheme,
  Tooltip,
  ClockIcon,
  Notification,
  UserIcon,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { type ModelTrace, type ModelTraceInfoV3 } from './ModelTrace.types';
import { createTraceV4LongIdentifier, doesTraceSupportV4API, isV3ModelTraceInfo } from './ModelTraceExplorer.utils';
import { ModelTraceHeaderMetricSection } from './ModelTraceExplorerMetricSection';
import { useModelTraceExplorerViewState } from './ModelTraceExplorerViewStateContext';
import { ModelTraceHeaderMetadataPill } from './ModelTraceHeaderMetadataPill';
import { ModelTraceHeaderSessionIdTag } from './ModelTraceHeaderSessionIdTag';
import { useParams } from './RoutingUtils';
import { isUserFacingTag, parseJSONSafe, truncateToFirstLineWithMaxLength } from './TagUtils';
import { SESSION_ID_METADATA_KEY, MLFLOW_TRACE_USER_KEY, TOKEN_USAGE_METADATA_KEY } from './constants';
import { spanTimeFormatter } from './timeline-tree/TimelineTree.utils';

const BASE_NOTIFICATION_COMPONENT_ID = 'mlflow.model_trace_explorer.header_details.notification';

export const ModelTraceHeaderDetails = ({ modelTraceInfo }: { modelTraceInfo: ModelTrace['info'] }) => {
  const { theme } = useDesignSystemTheme();
  const [showNotification, setShowNotification] = useState(false);
  const { rootNode } = useModelTraceExplorerViewState();
  const { experimentId } = useParams();

  const tags = Object.entries(modelTraceInfo.tags ?? {}).filter(([key]) => isUserFacingTag(key));

  const [modelTraceId, modelTraceIdToDisplay] = useMemo(() => {
    if (doesTraceSupportV4API(modelTraceInfo) && isV3ModelTraceInfo(modelTraceInfo)) {
      return [createTraceV4LongIdentifier(modelTraceInfo), modelTraceInfo.trace_id];
    }
    return [isV3ModelTraceInfo(modelTraceInfo) ? modelTraceInfo.trace_id : modelTraceInfo.request_id ?? ''];
  }, [modelTraceInfo]);

  const tokenUsage = useMemo(() => {
    const tokenUsage = parseJSONSafe(
      (modelTraceInfo as ModelTraceInfoV3)?.trace_metadata?.[TOKEN_USAGE_METADATA_KEY] ?? '{}',
    );
    return tokenUsage;
  }, [modelTraceInfo]);

  const totalTokens = useMemo(() => tokenUsage?.total_tokens, [tokenUsage]);

  const sessionId = useMemo(() => {
    return (modelTraceInfo as ModelTraceInfoV3)?.trace_metadata?.[SESSION_ID_METADATA_KEY];
  }, [modelTraceInfo]);

  const userId = useMemo(() => {
    return (modelTraceInfo as ModelTraceInfoV3)?.trace_metadata?.[MLFLOW_TRACE_USER_KEY];
  }, [modelTraceInfo]);

  const latency = useMemo((): string | undefined => {
    if (rootNode) {
      return spanTimeFormatter(rootNode.end - rootNode.start);
    }

    return undefined;
  }, [rootNode]);

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
            displayValue={modelTraceIdToDisplay}
            color="purple"
            getTruncatedLabel={getTruncatedLabel}
            onCopy={handleCopy}
          />
        )}
        {totalTokens && (
          <ModelTraceHeaderMetricSection
            label={<FormattedMessage defaultMessage="Token count" description="Label for the token count section" />}
            value={totalTokens.toString()}
            getTruncatedLabel={getTruncatedLabel}
            onCopy={handleCopy}
          />
        )}
        {latency && (
          <ModelTraceHeaderMetricSection
            label={<FormattedMessage defaultMessage="Latency" description="Label for the latency section" />}
            icon={<ClockIcon css={{ fontSize: 12, display: 'flex', alignItems: 'center', justifyContent: 'center' }} />}
            value={latency}
            getTruncatedLabel={getTruncatedLabel}
            onCopy={handleCopy}
          />
        )}
        {sessionId && experimentId && (
          <ModelTraceHeaderSessionIdTag handleCopy={handleCopy} experimentId={experimentId} sessionId={sessionId} />
        )}
        {userId && (
          <ModelTraceHeaderMetricSection
            label={<FormattedMessage defaultMessage="User" description="Label for the user id section" />}
            icon={<UserIcon css={{ fontSize: 12, display: 'flex' }} />}
            value={userId}
            color="default"
            getTruncatedLabel={getTruncatedLabel}
            onCopy={handleCopy}
          />
        )}
        <ModelTraceHeaderMetadataPill
          traceMetadata={(modelTraceInfo as ModelTraceInfoV3)?.trace_metadata}
          getTruncatedLabel={getTruncatedLabel}
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
                const fullText = `${key}: ${value}`;

                return (
                  <Tooltip
                    key={key}
                    componentId="shared.model-trace-explorer.header-details.tooltip"
                    content={fullText}
                  >
                    <Tag
                      componentId="shared.model-trace-explorer.header-details.tag"
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
