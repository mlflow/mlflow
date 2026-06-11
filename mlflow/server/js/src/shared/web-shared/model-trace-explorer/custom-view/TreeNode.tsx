import { useEffect, useMemo, useState } from 'react';

import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, ChildListSchema, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import { ChevronDownIcon, ChevronRightIcon, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';

import { ModelIconType } from '../ModelTrace.types';
import { ModelTraceExplorerIcon } from '../ModelTraceExplorerIcon';
import { type PanelItem, TreeDepthContext, useTreeDepth, useTreeSelection } from './TreeSelectionContext';

const INDENT_PER_DEPTH = 20;
const CHEVRON_SLOT = 20;
const ROW_HEIGHT = 28;

// Lightweight directive describing one entry in the node's side panel. The host
// (not the author/LLM) turns these into real components from its nodeMap when
// the node is selected — so a node never has to carry the heavy span data.
const PanelItemSchema = z
  .object({
    type: z
      .enum(['input', 'output', 'attributes', 'markdown', 'feedback'])
      .describe(
        'input/output/attributes -> host renders the span field as a KeyValueViewer; markdown -> a Markdown block; feedback -> FeedbackButtons scoped to this node\'s span.',
      ),
    text: z.string().describe('Markdown body (type: markdown).').optional(),
    title: z.string().describe('Section heading (markdown) or value label override.').optional(),
    label: z.string().describe('Feedback prompt text (type: feedback).').optional(),
    name: z.string().describe('Feedback assessment name (type: feedback).').optional(),
  })
  .strict();

/**
 * Schema (API) for the composable TreeNode component. A node renders a row
 * (icon + label + optional badge) and an optional nested set of child
 * TreeNodes. Authors attach `panelItems` directives describing what the host
 * should build in the TreeView side panel when the node is selected (the span's
 * input/output/attributes, a markdown summary, feedback buttons). This keeps
 * nodes tiny: they map 1:1 to spans or represent higher-level milestones, and
 * the host fills in the heavy data on demand.
 */
export const TreeNodeApi = {
  name: 'TreeNode',
  schema: z
    .object({
      label: DynamicStringSchema.describe('The node label (e.g. the span name). Used when `title` is absent.').optional(),
      title: DynamicStringSchema.describe('Preferred node heading; falls back to `label`.').optional(),
      icon: z.string().describe('The span icon type (matches the Details & Timeline icons).').optional(),
      hasException: z.boolean().describe('Renders the icon in the danger tone with an exception marker.').optional(),
      isRootSpan: z.boolean().describe('Renders the icon in the root/AI tone.').optional(),
      badge: DynamicStringSchema.describe('Optional trailing tag text (e.g. an assessment count).').optional(),
      spanId: DynamicStringSchema.describe(
        'Optional span id this node represents. Lets markdown #span:<id> deeplinks target this node and scopes the side panel + embedded feedback.',
      ).optional(),
      panelItems: z
        .array(PanelItemSchema)
        .describe(
          'Directives for the side panel shown when this node is selected. The host builds the actual components (KeyValueViewer / Markdown / FeedbackButtons) from the span data.',
        )
        .optional(),
      children: ChildListSchema.describe('Nested TreeNode ids, rendered indented below this node.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

const asString = (value: unknown): string => (typeof value === 'string' ? value : String(value ?? ''));

const isModelIconType = (value: unknown): value is ModelIconType =>
  typeof value === 'string' && (Object.values(ModelIconType) as string[]).includes(value);

export const TreeNode = createComponentImplementation(TreeNodeApi, ({ props, context, buildChild }) => {
  const { theme } = useDesignSystemTheme();
  const depth = useTreeDepth();
  const selection = useTreeSelection();
  const [expanded, setExpanded] = useState(true);

  const nodeId = context.componentModel.id;
  const spanId = typeof props.spanId === 'string' && props.spanId ? props.spanId : undefined;

  const heading = props.title ? asString(props.title) : props.label ? asString(props.label) : '';
  const badge = props.badge != null && props.badge !== '' ? asString(props.badge) : undefined;

  const panelItems = useMemo<PanelItem[]>(
    () => (Array.isArray(props.panelItems) ? (props.panelItems as PanelItem[]) : []),
    [props.panelItems],
  );
  const childIds = useMemo(() => (Array.isArray(props.children) ? (props.children as string[]) : []), [props.children]);

  const hasBody = childIds.length > 0;
  const selectable = selection.enabled && panelItems.length > 0;
  const isSelected = selection.selectedNodeId === nodeId;

  // Register the span this node represents so markdown `#span:<id>` deeplinks
  // (in any markdown) can resolve to and rebuild this node's side panel.
  useEffect(() => {
    if (!spanId) {
      return undefined;
    }
    selection.registerSpan(spanId, { nodeId, panelItems });
    return () => selection.unregisterSpan(spanId);
  }, [spanId, nodeId, panelItems, selection.registerSpan, selection.unregisterSpan]);

  return (
    <div>
      <div
        onClick={selectable ? () => selection.select(nodeId, panelItems, spanId) : undefined}
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          height: ROW_HEIGHT,
          paddingLeft: depth * INDENT_PER_DEPTH,
          paddingRight: theme.spacing.sm,
          cursor: selectable ? 'pointer' : 'default',
          borderRadius: theme.borders.borderRadiusSm,
          backgroundColor: isSelected ? theme.colors.actionDefaultBackgroundPress : undefined,
          '&:hover':
            selectable && !isSelected ? { backgroundColor: theme.colors.actionTertiaryBackgroundHover } : undefined,
        }}
      >
        {hasBody ? (
          <span
            role="button"
            tabIndex={0}
            aria-label={expanded ? 'Collapse' : 'Expand'}
            onClick={(event) => {
              event.stopPropagation();
              setExpanded((prev) => !prev);
            }}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.stopPropagation();
                setExpanded((prev) => !prev);
              }
            }}
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: CHEVRON_SLOT,
              flexShrink: 0,
              cursor: 'pointer',
              color: theme.colors.textSecondary,
            }}
          >
            {expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          </span>
        ) : (
          <span css={{ width: CHEVRON_SLOT, flexShrink: 0 }} />
        )}
        <span css={{ display: 'flex', flexShrink: 0 }}>
          <ModelTraceExplorerIcon
            type={isModelIconType(props.icon) ? props.icon : ModelIconType.UNKNOWN}
            hasException={props.hasException === true}
            isRootSpan={props.isRootSpan === true}
          />
        </span>
        <Typography.Text
          css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1, minWidth: 0 }}
        >
          {heading}
        </Typography.Text>
        {badge && (
          <Tag componentId="shared.model-trace-explorer.custom-view.tree-node.badge" color="default">
            {badge}
          </Tag>
        )}
      </div>

      {expanded && hasBody && (
        <TreeDepthContext.Provider value={depth + 1}>
          {childIds.map((id) => (
            <div key={id}>{buildChild(id)}</div>
          ))}
        </TreeDepthContext.Provider>
      )}
    </div>
  );
});
