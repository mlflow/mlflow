/**
 * Context tags component for the Assistant chat panel.
 * Displays explicitly selected context (traces, runs) as compact tags.
 * Implicit context like experimentId and currentPage is passed to the assistant but not displayed.
 */

import { ForkHorizontalIcon, PlayIcon, Tag, TagColors, Tooltip, useDesignSystemTheme } from '@databricks/design-system';

import { useAssistantPageContext } from './AssistantPageContext';

const MAX_VISIBLE_ITEMS = 3;

function truncateId(id: string, maxLen = 8): string {
  return id.length > maxLen ? `${id.slice(0, maxLen)}...` : id;
}

interface ContextTagGroupProps {
  ids: string[];
  color: TagColors;
  label: string;
  type: 'trace' | 'run';
  Icon: typeof ForkHorizontalIcon;
}

const COMPONENT_IDS = {
  trace: {
    tag: 'mlflow.assistant.chat_panel.context.trace',
    tooltip: 'mlflow.assistant.chat_panel.context.trace.tooltip',
    more: 'mlflow.assistant.chat_panel.context.trace.more',
    moreTooltip: 'mlflow.assistant.chat_panel.context.trace.more.tooltip',
  },
  run: {
    tag: 'mlflow.assistant.chat_panel.context.run',
    tooltip: 'mlflow.assistant.chat_panel.context.run.tooltip',
    more: 'mlflow.assistant.chat_panel.context.run.more',
    moreTooltip: 'mlflow.assistant.chat_panel.context.run.more.tooltip',
  },
} as const;

function ContextTagGroup({ ids, color, label, type, Icon }: ContextTagGroupProps): React.ReactElement | null {
  const { theme } = useDesignSystemTheme();
  const componentIds = COMPONENT_IDS[type];

  if (ids.length === 0) return null;

  const visibleIds = ids.slice(0, MAX_VISIBLE_ITEMS);
  const hiddenCount = ids.length - MAX_VISIBLE_ITEMS;

  return (
    <>
      {visibleIds.map((id) => (
        <Tooltip key={id} componentId={componentIds.tooltip} content={`${label}: ${id}`}>
          <Tag componentId={componentIds.tag} color={color}>
            <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <Icon css={{ fontSize: 12 }} />
              <span>{truncateId(id)}</span>
            </span>
          </Tag>
        </Tooltip>
      ))}
      {hiddenCount > 0 && (
        <Tooltip componentId={componentIds.moreTooltip} content={`${hiddenCount} more ${label.toLowerCase()}s`}>
          <Tag componentId={componentIds.more} color={color}>
            +{hiddenCount}
          </Tag>
        </Tooltip>
      )}
    </>
  );
}

/**
 * Context tags showing current page context.
 */
export function AssistantContextTags(): React.ReactElement | null {
  const { theme } = useDesignSystemTheme();
  const context = useAssistantPageContext();

  const traceId = context['traceId'] as string | undefined;
  const selectedTraceIds = context['selectedTraceIds'] as string[] | undefined;
  const runId = context['runId'] as string | undefined;
  const selectedRunIds = context['selectedRunIds'] as string[] | undefined;

  // Remove duplication of active and selected trace/run IDs
  const traceIds = [...new Set([traceId, ...(selectedTraceIds ?? [])].filter(Boolean))] as string[];
  const runIds = [...new Set([runId, ...(selectedRunIds ?? [])].filter(Boolean))] as string[];

  if (traceIds.length === 0 && runIds.length === 0) return null;

  return (
    <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs, paddingTop: theme.spacing.sm }}>
      <ContextTagGroup ids={traceIds} color="indigo" label="Trace" type="trace" Icon={ForkHorizontalIcon} />
      <ContextTagGroup ids={runIds} color="turquoise" label="Run" type="run" Icon={PlayIcon} />
    </div>
  );
}
