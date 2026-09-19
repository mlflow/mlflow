import { useCallback, useMemo, useState } from 'react';

import {
  Button,
  Overflow,
  PencilIcon,
  Tag,
  Typography,
  useDesignSystemTheme,
  Tooltip,
  ClockIcon,
  Notification,
  UserIcon,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTrace, ModelTraceInfoV3, ModelTraceState } from './ModelTrace.types';
import {
  createTraceV4LongIdentifier,
  getTraceCost,
  getTraceTokenUsage,
  getRootSpanTimeToFirstTokenMs,
  isV3ModelTraceInfo,
} from './ModelTraceExplorer.utils';
import { doesTraceSupportV4API } from '../../genai-traces-table/utils/TraceLocationUtils';
import { ModelTraceHeaderMetricSection } from '../ModelTraceExplorerMetricSection';
import { isTraceCostType, ModelTraceExplorerCostHoverCard, type TraceCost } from '../ModelTraceExplorerCostHoverCard';
import {
  isTokenUsageType,
  ModelTraceExplorerTokenUsageHoverCard,
  type TokenUsage,
} from '../ModelTraceExplorerTokenUsageHoverCard';
import { useModelTraceExplorerUpdateTraceContext } from '../contexts/UpdateTraceContext';
import { useModelTraceExplorerViewState } from './ModelTraceExplorerViewStateContext';
import { ModelTraceHeaderMetadataPill } from '../ModelTraceHeaderMetadataPill';
import { ModelTraceHeaderSessionIdTag } from './ModelTraceHeaderSessionIdTag';
import { ModelTraceHeaderStatusTag } from '../ModelTraceHeaderStatusTag';
import { useParams } from '../RoutingUtils';
import { isUserFacingTag, truncateToFirstLineWithMaxLength } from '../TagUtils';
import { SESSION_ID_METADATA_KEY, MLFLOW_TRACE_USER_KEY } from '../constants';
import { spanTimeFormatter } from './timeline-tree/TimelineTree.utils';

const BASE_NOTIFICATION_COMPONENT_ID = 'mlflow.model_trace_explorer.header_details.notification';

export const ModelTraceHeaderDetails = ({ modelTraceInfo }: { modelTraceInfo: ModelTrace['info'] }): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const [showNotification, setShowNotification] = useState(false);
  const { rootNode } = useModelTraceExplorerViewState();
  const { experimentId } = useParams();
  const { onEditTags } = useModelTraceExplorerUpdateTraceContext();
  const tags = Object.entries(modelTraceInfo.tags ?? {}).filter(([key]) => isUserFacingTag(key));

  const [modelTraceId, modelTraceIdToDisplay] = useMemo(() => {
    if (doesTraceSupportV4API(modelTraceInfo) && isV3ModelTraceInfo(modelTraceInfo)) {
      return [createTraceV4LongIdentifier(modelTraceInfo), modelTraceInfo.trace_id];
    }
    return [isV3ModelTraceInfo(modelTraceInfo) ? modelTraceInfo.trace_id : (modelTraceInfo.request_id ?? '')];
  }, [modelTraceInfo]);

  const tokenUsage = useMemo<Partial<TokenUsage> | undefined>(
    () => getTraceTokenUsage(modelTraceInfo as ModelTraceInfoV3) as Partial<TokenUsage> | undefined,
    [modelTraceInfo],
  );

  const cost = useMemo<Partial<TraceCost> | undefined>(
    () => getTraceCost(modelTraceInfo as ModelTraceInfoV3) as Partial<TraceCost> | undefined,
    [modelTraceInfo],
  );

  const sessionId = useMemo(() => {
    return (modelTraceInfo as ModelTraceInfoV3)?.trace_metadata?.[SESSION_ID_METADATA_KEY];
  }, [modelTraceInfo]);

  const userId = useMemo(() => {
    return (modelTraceInfo as ModelTraceInfoV3)?.trace_metadata?.[MLFLOW_TRACE_USER_KEY];
  }, [modelTraceInfo]);

  // Derive status label/icon from TraceInfo (V3 only)
  const statusState: ModelTraceState | undefined = useMemo(
    () => (isV3ModelTraceInfo(modelTraceInfo) ? modelTraceInfo.state : undefined),
    [modelTraceInfo],
  );

  const latency = useMemo((): string | undefined => {
    if (rootNode) {
      return spanTimeFormatter(rootNode.end - rootNode.start);
    }

    return undefined;
  }, [rootNode]);

  const timeToFirstToken = useMemo((): string | undefined => {
    const ms = getRootSpanTimeToFirstTokenMs(rootNode);
    // spanTimeFormatter expects microseconds; the attribute value is in milliseconds.
    return ms !== null ? spanTimeFormatter(ms * 1000) : undefined;
  }, [rootNode]);

  const handleTagClick = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const getTruncatedLabel = (label: string) => truncateToFirstLineWithMaxLength(label, 40);

  const handleCopy = useCallback(() => {
    setShowNotification(true);
    setTimeout(() => setShowNotification(false), 2000);
  }, []);

  return (
    <div css={{ paddingLeft: theme.spacing.md, paddingBottom: theme.spacing.sm }}>
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          gap: theme.spacing.md,
          rowGap: theme.spacing.sm,
          flexWrap: 'wrap',
        }}
      >
        {statusState && (
          <ModelTraceHeaderStatusTag
            statusState={statusState}
            getTruncatedLabel={getTruncatedLabel}
            errorMessage={isV3ModelTraceInfo(modelTraceInfo) ? modelTraceInfo.error_message : undefined}
          />
        )}
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
        {isTokenUsageType(tokenUsage) && <ModelTraceExplorerTokenUsageHoverCard tokenUsage={tokenUsage} />}
        {isTraceCostType(cost) && <ModelTraceExplorerCostHoverCard cost={cost} />}
        {timeToFirstToken && (
          <ModelTraceHeaderMetricSection
            label={
              <FormattedMessage
                defaultMessage="Time to first token"
                description="Label for the time to first token metric in the trace header"
              />
            }
            icon={<ClockIcon css={{ fontSize: 12, display: 'flex', alignItems: 'center', justifyContent: 'center' }} />}
            value={timeToFirstToken}
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
          <ModelTraceHeaderSessionIdTag
            handleCopy={handleCopy}
            experimentId={experimentId}
            sessionId={sessionId}
            traceId={modelTraceId}
          />
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
        {/* Render the Tags section when there are tags, or when the caller opted into editing (so the
            "Add tags" affordance shows even with zero tags). With `onEditTags` undefined the condition
            collapses to the prior `tags.length > 0`, leaving v3 and other consumers unchanged. */}
        {(tags.length > 0 || Boolean(onEditTags)) && (
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
            {tags.length > 0 && (
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
            )}
            {onEditTags && (
              <Button
                componentId="shared.model-trace-explorer.header-details.edit-tags"
                size="small"
                type="tertiary"
                icon={<PencilIcon />}
                onClick={() => onEditTags(modelTraceInfo)}
              >
                {tags.length > 0 ? (
                  <FormattedMessage
                    defaultMessage="Edit"
                    description="Label for the button that edits a trace's tags"
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Add tags"
                    description="Label for the button that adds tags to a trace that has none"
                  />
                )}
              </Button>
            )}
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
    </div>
  );
};
