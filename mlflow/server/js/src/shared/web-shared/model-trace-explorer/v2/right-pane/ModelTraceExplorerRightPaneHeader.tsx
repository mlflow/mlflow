import type { ReactNode } from 'react';
import { useMemo } from 'react';

import {
  Button,
  HoverCard,
  ListIcon,
  ModelsIcon,
  NewWindowIcon,
  SpeechBubbleIcon,
  Tag,
  TokenIcon,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { useCopyController } from '../../../copy/useCopyController';
import { doesTraceSupportV4API } from '../../../genai-traces-table/utils/TraceLocationUtils';
import { formatCostUSD } from '../../CostUtils';
import { getExperimentChatSessionPageRoute } from '../../MlflowUtils';
import type { ModelTrace, ModelTraceSpanNode, SpanCostInfo } from '../ModelTrace.types';
import { createTraceV4LongIdentifier, getSpanExceptionCount, isV3ModelTraceInfo } from '../ModelTraceExplorer.utils';
import { isTraceCostType, type TraceCost } from '../../ModelTraceExplorerCostHoverCard';
import { useModelTraceExplorerContext } from '../ModelTraceExplorerContext';
import { Link, useParams } from '../../RoutingUtils';
import { SELECTED_TRACE_ID_QUERY_PARAM, SESSION_ID_METADATA_KEY } from '../../constants';
import { isUserFacingTag, truncateToFirstLineWithMaxLength } from '../../TagUtils';
import { AssessmentPaneToggle } from '../assessments-pane/AssessmentPaneToggle';
import { getSpanTokenUsage, type SpanTokenUsage } from '../ModelTraceTokenUsage.utils';
import { ModelTraceExplorerLinkedPromptsPopover } from '../../linked-prompts/ModelTraceExplorerLinkedPromptsPopover';
import { MLFLOW_LINKED_PROMPTS_TAG, parseLinkedPrompts } from '../../linked-prompts/utils';

export type HeaderTokenUsage = SpanTokenUsage;

const getUserFacingTags = (tags: ModelTrace['info']['tags']): Array<[string, unknown]> => {
  if (Array.isArray(tags)) {
    return tags.filter(({ key }) => isUserFacingTag(key)).map(({ key, value }) => [key, value]);
  }

  return Object.entries(tags ?? {}).filter(([key]) => isUserFacingTag(key));
};

const MetadataItem = ({ children, tooltip }: { children: ReactNode; tooltip: ReactNode }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Tooltip componentId="shared.model-trace-explorer.right-pane-header-metadata-tooltip" content={tooltip}>
      <span
        css={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          minWidth: 0,
          color: theme.colors.textSecondary,
          svg: {
            color: theme.colors.textSecondary,
            width: 12,
            height: 12,
          },
        }}
      >
        {children}
      </span>
    </Tooltip>
  );
};

const SessionMetadataItem = ({ sessionId, sessionPageUrl }: { sessionId: string; sessionPageUrl?: string }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const copyTooltip = intl.formatMessage({
    defaultMessage: 'Copy session ID',
    description: 'Tooltip for copying a trace session ID',
  });
  const { actionIcon, ariaLabel, copy, handleTooltipOpenChange, tooltipMessage, tooltipOpen } = useCopyController(
    sessionId,
    copyTooltip,
  );

  return (
    <HoverCard
      trigger={
        <span
          css={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: theme.spacing.xs,
            maxWidth: 120,
            minWidth: 0,
            color: theme.colors.textSecondary,
          }}
        >
          <SpeechBubbleIcon />
          <Typography.Text
            color="secondary"
            size="md"
            css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
          >
            {sessionId}
          </Typography.Text>
        </span>
      }
      content={
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, whiteSpace: 'nowrap' }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md }}>
            <Typography.Text css={{ userSelect: 'text' }}>{sessionId}</Typography.Text>
            <Tooltip
              componentId="shared.model-trace-explorer.right-pane-header-copy-session-id-tooltip"
              content={tooltipMessage}
              onOpenChange={handleTooltipOpenChange}
              open={tooltipOpen}
            >
              <Button
                aria-label={ariaLabel}
                componentId="shared.model-trace-explorer.right-pane-header.copy-session-id"
                icon={actionIcon}
                onClick={copy}
                size="small"
                type="tertiary"
                css={{
                  width: theme.spacing.lg,
                  minWidth: theme.spacing.lg,
                  height: theme.spacing.lg,
                  padding: 0,
                  svg: {
                    width: theme.typography.fontSizeSm,
                    height: theme.typography.fontSizeSm,
                  },
                }}
              />
            </Tooltip>
          </div>
          {sessionPageUrl && (
            <div
              css={{
                display: 'flex',
                justifyContent: 'flex-end',
                borderTop: `1px solid ${theme.colors.borderDecorative}`,
                paddingTop: theme.spacing.sm,
              }}
            >
              <Link
                componentId="shared.model-trace-explorer.right-pane-header.session-link"
                to={sessionPageUrl}
                target="_blank"
                rel="noopener noreferrer"
                css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}
              >
                <FormattedMessage
                  defaultMessage="View session"
                  description="Link to view the chat session for a trace"
                />
                <NewWindowIcon css={{ fontSize: theme.typography.fontSizeSm }} />
              </Link>
            </div>
          )}
        </div>
      }
      side="bottom"
      align="start"
    />
  );
};

const BreakdownRow = ({ label, value, bold = false }: { label: ReactNode; value: ReactNode; bold?: boolean }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: theme.spacing.lg,
      }}
    >
      <Typography.Text bold={bold}>{label}</Typography.Text>
      <Tag componentId="shared.model-trace-explorer.right-pane-header-breakdown-tag">
        <span>{value}</span>
      </Tag>
    </div>
  );
};

export const TokenUsageMetadataItem = ({ tokenUsage }: { tokenUsage: HeaderTokenUsage }): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const cacheReadTokens = tokenUsage.cache_read_input_tokens ?? null;
  const cacheCreationTokens = tokenUsage.cache_creation_input_tokens ?? null;

  return (
    <HoverCard
      trigger={
        <span
          css={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: theme.spacing.xs,
            minWidth: 0,
            color: theme.colors.textSecondary,
            svg: {
              color: theme.colors.textSecondary,
              width: 12,
              height: 12,
            },
          }}
        >
          <TokenIcon />
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage
              defaultMessage="{tokenCount, number} tokens"
              description="Compact token count in the trace details header"
              values={{ tokenCount: tokenUsage.total_tokens }}
            />
          </Typography.Text>
        </span>
      }
      content={
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, minWidth: 240 }}>
          <Typography.Title level={3} withoutMargins>
            <FormattedMessage defaultMessage="Usage breakdown" description="Header for token usage breakdown" />
          </Typography.Title>
          <BreakdownRow
            label={<FormattedMessage defaultMessage="Input tokens" description="Label for input token usage" />}
            value={
              tokenUsage.input_tokens ?? (
                <FormattedMessage defaultMessage="N/A" description="Token usage value is unavailable" />
              )
            }
          />
          <BreakdownRow
            label={
              <FormattedMessage defaultMessage="Cache read" description="Label for cache read input token usage" />
            }
            value={
              cacheReadTokens ?? (
                <FormattedMessage defaultMessage="N/A" description="Token usage value is unavailable" />
              )
            }
          />
          <BreakdownRow
            label={
              <FormattedMessage defaultMessage="Cache write" description="Label for cache creation input token usage" />
            }
            value={
              cacheCreationTokens ?? (
                <FormattedMessage defaultMessage="N/A" description="Token usage value is unavailable" />
              )
            }
          />
          <BreakdownRow
            label={<FormattedMessage defaultMessage="Output tokens" description="Label for output token usage" />}
            value={
              tokenUsage.output_tokens ?? (
                <FormattedMessage defaultMessage="N/A" description="Token usage value is unavailable" />
              )
            }
          />
          <div
            css={{
              borderTop: `1px solid ${theme.colors.borderDecorative}`,
              paddingTop: theme.spacing.sm,
            }}
          >
            <BreakdownRow
              label={<FormattedMessage defaultMessage="Total" description="Label for total token usage" />}
              value={tokenUsage.total_tokens}
              bold
            />
          </div>
        </div>
      }
      side="bottom"
      align="start"
    />
  );
};

export const CostMetadataItem = ({
  cost,
  formatTotalCost = formatCostUSD,
}: {
  cost: TraceCost | SpanCostInfo;
  formatTotalCost?: (cost: number) => string;
}): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const totalCost = formatTotalCost(cost.total_cost);

  return (
    <HoverCard
      trigger={
        <span
          css={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: theme.spacing.xs,
            minWidth: 0,
            color: theme.colors.textSecondary,
          }}
        >
          <Typography.Text color="secondary" size="sm">
            {totalCost}
          </Typography.Text>
        </span>
      }
      content={
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, minWidth: 220 }}>
          <Typography.Title level={3} withoutMargins>
            <FormattedMessage defaultMessage="Cost breakdown" description="Header for cost breakdown" />
          </Typography.Title>
          <BreakdownRow
            label={<FormattedMessage defaultMessage="Input cost" description="Label for input cost" />}
            value={formatCostUSD(cost.input_cost)}
          />
          <BreakdownRow
            label={<FormattedMessage defaultMessage="Output cost" description="Label for output cost" />}
            value={formatCostUSD(cost.output_cost)}
          />
          <div
            css={{
              borderTop: `1px solid ${theme.colors.borderDecorative}`,
              paddingTop: theme.spacing.sm,
            }}
          >
            <BreakdownRow
              label={<FormattedMessage defaultMessage="Total" description="Label for total cost" />}
              value={totalCost}
              bold
            />
          </div>
        </div>
      }
      side="bottom"
      align="start"
    />
  );
};

const TagsMetadataItem = ({ tags }: { tags: Array<[string, unknown]> }) => {
  const { theme } = useDesignSystemTheme();

  if (tags.length === 0) {
    return null;
  }

  return (
    <HoverCard
      trigger={
        <span
          css={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: theme.spacing.xs,
            minWidth: 0,
            color: theme.colors.textSecondary,
            svg: {
              color: theme.colors.textSecondary,
              width: 12,
              height: 12,
            },
          }}
        >
          <ListIcon />
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage
              defaultMessage="{tagCount, plural, one {1 tag} other {# tags}}"
              description="Compact trace tags count in the trace details header"
              values={{ tagCount: tags.length }}
            />
          </Typography.Text>
        </span>
      }
      content={
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, maxWidth: 360 }}>
          {tags.map(([key, value]) => (
            <div key={key} css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'baseline' }}>
              <Typography.Text css={{ flexShrink: 0 }}>{truncateToFirstLineWithMaxLength(key, 24)}</Typography.Text>
              <Typography.Text color="secondary" css={{ wordBreak: 'break-word' }}>
                {String(value)}
              </Typography.Text>
            </div>
          ))}
        </div>
      }
      side="bottom"
      align="start"
    />
  );
};

export const ModelTraceExplorerRightPaneHeader = ({
  activeSpan,
  modelTraceInfo,
  showAssessmentsToggle,
}: {
  activeSpan: ModelTraceSpanNode;
  modelTraceInfo?: ModelTrace['info'];
  showAssessmentsToggle: boolean;
}): React.ReactElement | null => {
  const { theme } = useDesignSystemTheme();
  const { experimentId: experimentIdFromParams } = useParams();
  const { experimentId: experimentIdFromContext, rightPaneHeaderActions } = useModelTraceExplorerContext();
  const experimentId = experimentIdFromContext ?? experimentIdFromParams;
  const activeSpanTitle = typeof activeSpan.title === 'string' ? activeSpan.title : undefined;
  const hasException = getSpanExceptionCount(activeSpan) > 0;
  const isRootSpan = !activeSpan.parentId;
  const modelName = activeSpan.modelName;

  const tokenUsage = useMemo<HeaderTokenUsage | undefined>(() => getSpanTokenUsage(activeSpan), [activeSpan]);
  const cost = activeSpan.cost;

  const sessionId =
    isRootSpan && isV3ModelTraceInfo(modelTraceInfo)
      ? modelTraceInfo.trace_metadata?.[SESSION_ID_METADATA_KEY]
      : undefined;
  const traceId = isV3ModelTraceInfo(modelTraceInfo)
    ? doesTraceSupportV4API(modelTraceInfo)
      ? createTraceV4LongIdentifier(modelTraceInfo)
      : modelTraceInfo.trace_id
    : undefined;
  const sessionPageUrl =
    experimentId && sessionId
      ? `${getExperimentChatSessionPageRoute(experimentId, sessionId)}?${new URLSearchParams({
          ...(traceId ? { [SELECTED_TRACE_ID_QUERY_PARAM]: traceId } : {}),
        }).toString()}`
      : undefined;
  const tags = useMemo(
    () => (isRootSpan ? getUserFacingTags(modelTraceInfo?.tags) : []),
    [isRootSpan, modelTraceInfo?.tags],
  );
  const hasLinkedPrompts =
    isRootSpan &&
    isV3ModelTraceInfo(modelTraceInfo) &&
    parseLinkedPrompts(modelTraceInfo.tags?.[MLFLOW_LINKED_PROMPTS_TAG]).length > 0;

  const hasMetadata =
    modelName || tokenUsage || isTraceCostType(cost) || sessionId || tags.length > 0 || hasLinkedPrompts;

  return (
    <div
      css={{
        containerType: 'inline-size',
        boxSizing: 'border-box',
        padding: `${theme.spacing.xs}px ${theme.spacing.md + theme.spacing.xs}px ${theme.spacing.sm}px ${theme.spacing.sm}px`,
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.md,
          minWidth: 0,
          minHeight: theme.spacing.xl + 2 * theme.spacing.sm,
          boxSizing: 'border-box',
          '@container (max-width: 560px)': {
            flexWrap: 'wrap',
            rowGap: theme.spacing.xs,
          },
        }}
      >
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.xs,
            minWidth: 0,
            flexShrink: 1,
          }}
        >
          <span
            css={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: theme.spacing.xl,
              height: theme.spacing.xl,
              flexShrink: 0,
              svg: {
                width: theme.typography.fontSizeLg,
                height: theme.typography.fontSizeLg,
              },
            }}
          >
            {activeSpan.icon}
          </span>
          <Typography.Text
            bold
            size="lg"
            color={hasException ? 'error' : 'primary'}
            title={activeSpanTitle}
            css={{
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              minWidth: 0,
            }}
          >
            {activeSpan.title}
          </Typography.Text>
        </div>
        {hasMetadata && (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.md,
              minWidth: 0,
              overflow: 'hidden',
              flex: 1,
              flexWrap: 'wrap',
              rowGap: theme.spacing.xs,
              boxSizing: 'border-box',
              '@container (max-width: 560px)': {
                order: 3,
                flexBasis: '100%',
                paddingLeft: theme.spacing.xl + theme.spacing.xs,
              },
            }}
          >
            {modelName && (
              <MetadataItem
                tooltip={<FormattedMessage defaultMessage="Model" description="Tooltip for span model metadata item" />}
              >
                <ModelsIcon
                  css={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    alignSelf: 'center',
                    flexShrink: 0,
                    lineHeight: 1,
                  }}
                />
                <Typography.Text
                  color="secondary"
                  size="sm"
                  title={modelName}
                  css={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    maxWidth: 240,
                  }}
                >
                  {modelName}
                </Typography.Text>
              </MetadataItem>
            )}
            {tokenUsage && <TokenUsageMetadataItem tokenUsage={tokenUsage} />}
            {isTraceCostType(cost) && <CostMetadataItem cost={cost} />}
            {sessionId && <SessionMetadataItem sessionId={sessionId} sessionPageUrl={sessionPageUrl} />}
            <TagsMetadataItem tags={tags} />
            {hasLinkedPrompts && isV3ModelTraceInfo(modelTraceInfo) && (
              <ModelTraceExplorerLinkedPromptsPopover experimentId={experimentId} traceInfo={modelTraceInfo} />
            )}
          </div>
        )}
        {(showAssessmentsToggle || rightPaneHeaderActions) && (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.xs,
              flexShrink: 0,
              marginLeft: 'auto',
            }}
          >
            {rightPaneHeaderActions}
            {showAssessmentsToggle && <AssessmentPaneToggle assessmentCount={activeSpan.assessments.length} />}
          </div>
        )}
      </div>
    </div>
  );
};
