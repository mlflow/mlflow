/**
 * Context tags component for the Assistant chat panel.
 * Displays the current page context (traces, runs) as compact tags.
 */

import {
  ForkHorizontalIcon,
  HomeIcon,
  PlayIcon,
  Tag,
  TagColors,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';

import { useAssistantPageContext } from './AssistantPageContext';

const COMPONENT_ID = 'mlflow.assistant.chat_panel.context';
const MAX_VISIBLE_ITEMS = 3;
const MAX_PAGE_NAME_LENGTH = 30;

function truncateId(id: string, maxLen = 8): string {
  return id.length > maxLen ? `${id.slice(0, maxLen)}...` : id;
}

/**
 * Truncates a page name to fit within the UI.
 * Shortens long IDs (like run/experiment IDs) while preserving the page type.
 */
function truncatePageName(pageName: string, maxLen = MAX_PAGE_NAME_LENGTH): string {
  if (pageName.length <= maxLen) return pageName;

  // Try to preserve the structure: "Page Type > Sub Tab" or just "Page Type"
  const parts = pageName.split(' > ');
  if (parts.length === 2) {
    // Has sub-tab, try to shorten the main part
    const [mainPart, subTab] = parts;
    const availableLen = maxLen - subTab.length - 5; // 5 for " > " and "..."
    if (availableLen > 10) {
      return `${mainPart.slice(0, availableLen)}... > ${subTab}`;
    }
  }

  // Fallback: simple truncation
  return `${pageName.slice(0, maxLen - 3)}...`;
}

interface ContextTagGroupProps {
  ids: string[];
  color: TagColors;
  label: string;
  componentIdPrefix: string;
  Icon: typeof ForkHorizontalIcon;
}

function ContextTagGroup({
  ids,
  color,
  label,
  componentIdPrefix,
  Icon,
}: ContextTagGroupProps): React.ReactElement | null {
  const { theme } = useDesignSystemTheme();

  if (ids.length === 0) return null;

  const visibleIds = ids.slice(0, MAX_VISIBLE_ITEMS);
  const hiddenCount = ids.length - MAX_VISIBLE_ITEMS;

  return (
    <>
      {visibleIds.map((id) => (
        <Tooltip key={id} componentId={`${componentIdPrefix}.tooltip`} content={`${label}: ${id}`}>
          <Tag componentId={componentIdPrefix} color={color}>
            <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <Icon css={{ fontSize: 12 }} />
              <span>{truncateId(id)}</span>
            </span>
          </Tag>
        </Tooltip>
      ))}
      {hiddenCount > 0 && (
        <Tooltip
          componentId={`${componentIdPrefix}.more.tooltip`}
          content={`${hiddenCount} more ${label.toLowerCase()}s`}
        >
          <Tag componentId={`${componentIdPrefix}.more`} color={color}>
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
  const currentPage = context['currentPage'] as string | undefined;

  // Check if current page already shows run/trace info to avoid duplication
  const isOnRunPage = currentPage?.startsWith('Run ');
  // Only consider it a specific trace page if the trace ID is shown in the page title
  const isOnSpecificTracePage = traceId && currentPage?.includes(traceId);

  // Remove duplication of active and selected trace/run IDs
  // Don't show run tag separately if we're on a run page (it's already in the page title)
  const traceIds = [...new Set([traceId, ...(selectedTraceIds ?? [])].filter(Boolean))] as string[];
  const runIds = isOnRunPage
    ? // On run page: only show selected runs, not the current run (already in page title)
      [...new Set([...(selectedRunIds ?? [])].filter((id) => id !== runId && Boolean(id)))] as string[]
    : ([...new Set([runId, ...(selectedRunIds ?? [])].filter(Boolean))] as string[]);

  // Similarly for traces - don't show if the trace ID is already in the page title
  const filteredTraceIds = isOnSpecificTracePage && traceIds.length === 1 ? [] : traceIds;

  if (filteredTraceIds.length === 0 && runIds.length === 0 && !currentPage) return null;

  return (
    <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs, paddingTop: theme.spacing.sm }}>
      {currentPage && (
        <Tooltip componentId={`${COMPONENT_ID}.page.tooltip`} content={`Current page: ${currentPage}`}>
          <Tag componentId={`${COMPONENT_ID}.page`} color="lemon">
            <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, maxWidth: 200 }}>
              <HomeIcon css={{ fontSize: 12, flexShrink: 0 }} />
              <span css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {truncatePageName(currentPage)}
              </span>
            </span>
          </Tag>
        </Tooltip>
      )}
      <ContextTagGroup
        ids={filteredTraceIds}
        color="indigo"
        label="Trace"
        componentIdPrefix={`${COMPONENT_ID}.trace`}
        Icon={ForkHorizontalIcon}
      />
      <ContextTagGroup
        ids={runIds}
        color="turquoise"
        label="Run"
        componentIdPrefix={`${COMPONENT_ID}.run`}
        Icon={PlayIcon}
      />
    </div>
  );
}
