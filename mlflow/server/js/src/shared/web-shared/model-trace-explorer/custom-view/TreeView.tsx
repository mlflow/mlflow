import { useEffect, useMemo, useState } from 'react';

import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import {
  Checkbox,
  ChevronDownIcon,
  ChevronRightIcon,
  Slider,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';

import { ModelIconType } from '../ModelTrace.types';
import { ModelTraceExplorerIcon } from '../ModelTraceExplorerIcon';
import { useNodeSelection } from './NodeSelectionContext';
import { useTreeCheck } from './TreeCheckContext';

const ROW_HEIGHT = 28;
const INDENT_PER_DEPTH = 20;
const CHEVRON_SLOT = 20;

// A declarative, structured filter applied to nodes via their `attributes`.
// Intentionally mirrors the trace explorer's SpanFilterState concepts (type,
// exceptions, log level) so it can be produced deterministically by UI controls
// or a hand-written JSON predicate — no LLM/natural-language parsing required.
const TreeFilterSchema = z
  .object({
    types: z.array(z.string()).describe('Keep nodes whose `attributes.type` is in this list.').optional(),
    nameContains: z.string().describe('Keep nodes whose label contains this substring (case-insensitive).').optional(),
    hasException: z.boolean().describe('Keep only nodes whose `attributes.hasException` is true.').optional(),
    minLogLevel: z.number().describe('Keep nodes whose `attributes.logLevel` is at least this value.').optional(),
    minDurationMs: z.number().describe('Keep nodes whose `attributes.durationMs` is at least this value.').optional(),
  })
  .strict();

export type TreeFilter = z.infer<typeof TreeFilterSchema>;

// Validates/normalizes a raw (possibly user-edited) filter object. Returns the
// parsed filter, or an error string if it doesn't match the schema. This is the
// "schema layer" on top of JSON.parse so malformed filters fail gracefully.
export const parseTreeFilter = (raw: unknown): { filter?: TreeFilter; error?: string } => {
  const result = TreeFilterSchema.safeParse(raw ?? {});
  if (!result.success) {
    return { error: result.error.issues.map((issue) => issue.message).join('; ') };
  }
  return { filter: result.data };
};

type RawNode = {
  id?: unknown;
  label?: unknown;
  icon?: unknown;
  hasException?: unknown;
  isRootSpan?: unknown;
  badge?: unknown;
  attributes?: Record<string, unknown>;
  children?: RawNode[];
};

/**
 * Schema (API) for the generic TreeView primitive. It renders a collapsible,
 * indented tree of nodes (mirroring the Details & Timeline span tree) and can
 * scope the visible nodes with a structured `filter` applied to each node's
 * `attributes`, plus a live execution-time-window slider. It is domain-agnostic:
 * callers describe nodes with a label, an optional icon/badge, opaque
 * `attributes` used only for filtering, and nested `children`. Works for any
 * number of nodes / nesting depth.
 */
export const TreeViewApi = {
  name: 'TreeView',
  schema: z
    .object({
      title: DynamicStringSchema.describe('Optional heading shown above the tree.').optional(),
      // Recursive node data. Typed loosely for the generic binder (values are
      // inlined literals); the catalog.json documents the precise shape.
      nodes: z.array(z.any()).describe('The root nodes of the tree, in display order.'),
      filter: TreeFilterSchema.describe('Optional structured filter applied to node attributes.').optional(),
      emptyMessage: DynamicStringSchema.describe('Text shown when no nodes match.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

const asString = (value: unknown): string => (typeof value === 'string' ? value : String(value ?? ''));

const isModelIconType = (value: unknown): value is ModelIconType =>
  typeof value === 'string' && (Object.values(ModelIconType) as string[]).includes(value);

// True when a node's label + attributes satisfy every provided filter condition.
// Exported so the host can compute the same "scoped" set (e.g. for the feedback
// carousel) using the exact predicate the TreeView renders with.
export const matchesTreeFilter = (
  { label, attributes }: { label: unknown; attributes?: Record<string, unknown> },
  filter: TreeFilter,
): boolean => {
  const attrs = attributes ?? {};
  const type = attrs.type;
  const logLevel = typeof attrs.logLevel === 'number' ? attrs.logLevel : undefined;
  const durationMs = typeof attrs.durationMs === 'number' ? attrs.durationMs : undefined;
  const hasException = attrs.hasException === true;

  if (filter.types && filter.types.length > 0 && !(typeof type === 'string' && filter.types.includes(type))) {
    return false;
  }
  if (filter.nameContains && !asString(label).toLowerCase().includes(filter.nameContains.toLowerCase())) {
    return false;
  }
  if (filter.hasException && !hasException) {
    return false;
  }
  if (filter.minLogLevel != null && !(logLevel != null && logLevel >= filter.minLogLevel)) {
    return false;
  }
  if (filter.minDurationMs != null && !(durationMs != null && durationMs >= filter.minDurationMs)) {
    return false;
  }
  return true;
};

const nodeSelfMatches = (node: RawNode, filter: TreeFilter): boolean =>
  matchesTreeFilter({ label: node.label, attributes: node.attributes }, filter);

// A selected execution time window, in ms offsets from the trace start.
type TimeWindow = { t0: number; t1: number };

const getNodeTimeWindow = (node: RawNode): { start?: number; end?: number } => {
  const attrs = node.attributes ?? {};
  const start = typeof attrs.startOffsetMs === 'number' ? attrs.startOffsetMs : undefined;
  const end = typeof attrs.endOffsetMs === 'number' ? attrs.endOffsetMs : undefined;
  return { start, end };
};

// Overlap semantics: a span is in the window if it was active at any point
// during it (it started before the window ends and ended after it begins).
// Nodes without time data are always kept.
const nodeInWindow = (node: RawNode, { t0, t1 }: TimeWindow): boolean => {
  const { start, end } = getNodeTimeWindow(node);
  if (start == null) {
    return true;
  }
  const spanEnd = end ?? start;
  return start <= t1 && spanEnd >= t0;
};

type TimeMark = {
  offsetMs: number;
  label: string;
  icon: ModelIconType;
  hasException: boolean;
  isRootSpan: boolean;
};

// Derives the overall window bounds and a per-span start marker list from the
// (recursive) node tree. Returns undefined when no node carries time data.
const collectTimeData = (nodes: RawNode[]): { min: number; max: number; marks: TimeMark[] } | undefined => {
  let min = Infinity;
  let max = -Infinity;
  const marks: TimeMark[] = [];
  const visit = (node: RawNode) => {
    const { start, end } = getNodeTimeWindow(node);
    if (start != null) {
      min = Math.min(min, start);
      max = Math.max(max, end ?? start);
      marks.push({
        offsetMs: start,
        label: asString(node.label),
        icon: isModelIconType(node.icon) ? node.icon : ModelIconType.UNKNOWN,
        hasException: node.hasException === true,
        isRootSpan: node.isRootSpan === true,
      });
    }
    (node.children ?? []).forEach(visit);
  };
  nodes.forEach(visit);
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    return undefined;
  }
  return { min, max, marks };
};

// Recursively prunes the tree by the structured filter AND the time window: a
// node is kept if it self-matches both or has any kept descendant, so the
// surviving tree stays connected (ancestors of matches are preserved, like the
// trace explorer's "show parents" behavior).
const filterNodes = (
  nodes: RawNode[],
  filter: TreeFilter | undefined,
  window: TimeWindow | undefined,
): RawNode[] => {
  const hasFilter = Boolean(filter && Object.keys(filter).length > 0);
  if (!hasFilter && !window) {
    return nodes;
  }
  const result: RawNode[] = [];
  for (const node of nodes) {
    const keptChildren = filterNodes(node.children ?? [], filter, window);
    const selfMatches =
      (!hasFilter || nodeSelfMatches(node, filter as TreeFilter)) && (!window || nodeInWindow(node, window));
    if (selfMatches || keptChildren.length > 0) {
      result.push({ ...node, children: keptChildren });
    }
  }
  return result;
};

const formatOffset = (ms: number): string => (ms < 1000 ? `+${Math.round(ms)}ms` : `+${(ms / 1000).toFixed(2)}s`);

const TreeNodeRow = ({ node, depth }: { node: RawNode; depth: number }) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(true);
  const selection = useNodeSelection();
  const check = useTreeCheck();
  const children = Array.isArray(node.children) ? node.children : [];
  const hasChildren = children.length > 0;
  const badge = node.badge != null && node.badge !== '' ? asString(node.badge) : undefined;

  const nodeId = node.id != null ? asString(node.id) : undefined;
  const selectable = selection.enabled && Boolean(selection.onSelect) && nodeId !== undefined;
  const isSelected = selectable && selection.selectedId === nodeId;

  const checkable = check.enabled && Boolean(check.onToggle) && nodeId !== undefined;
  const isChecked = checkable && check.checkedIds.has(nodeId as string);

  const handleSelect = () => {
    if (selectable && nodeId !== undefined) {
      selection.onSelect?.(nodeId);
    }
  };

  return (
    <div>
      <div
        onClick={selectable ? handleSelect : undefined}
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
        {hasChildren ? (
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
        {checkable && (
          <span
            css={{ display: 'flex', flexShrink: 0 }}
            // Don't let toggling the checkbox also trigger row selection.
            onClick={(event) => event.stopPropagation()}
          >
            <Checkbox
              componentId="shared.model-trace-explorer.custom-view.tree-view.check"
              isChecked={isChecked}
              onChange={(next) => nodeId !== undefined && check.onToggle?.(nodeId, Boolean(next))}
              aria-label={`Select ${asString(node.label)}`}
            />
          </span>
        )}
        <span css={{ display: 'flex', flexShrink: 0 }}>
          <ModelTraceExplorerIcon
            type={isModelIconType(node.icon) ? node.icon : ModelIconType.UNKNOWN}
            hasException={node.hasException === true}
            isRootSpan={node.isRootSpan === true}
          />
        </span>
        <Typography.Text
          css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1, minWidth: 0 }}
        >
          {asString(node.label)}
        </Typography.Text>
        {badge && (
          <Tag componentId="shared.model-trace-explorer.custom-view.tree-view.badge" color="default">
            {badge}
          </Tag>
        )}
      </div>
      {hasChildren && expanded && (
        <div>
          {children.map((child, index) => (
            <TreeNodeRow key={index} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
};

// A two-thumb range slider over the trace's execution timeline. Per-span start
// markers sit above the track (hover for the span name + offset, using the same
// span-type icons as the tree), and an elapsed-time label tracks each thumb.
const TimeWindowSlider = ({
  timeData,
  range,
  onChange,
}: {
  timeData: { min: number; max: number; marks: TimeMark[] };
  range: [number, number];
  onChange: (next: [number, number]) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const { min, max, marks } = timeData;
  const span = max - min || 1;
  const pct = (value: number) => ((value - min) / span) * 100;
  const step = Math.max(1, Math.round(span / 200));

  return (
    <div css={{ display: 'flex', flexDirection: 'column', paddingInline: theme.spacing.xs }}>
      <Typography.Text size="sm" color="secondary">
        Execution time window
      </Typography.Text>
      {/* Thumb labels, positioned above each handle. */}
      <div css={{ position: 'relative', height: 18 }}>
        {([0, 1] as const).map((thumb) => (
          <Typography.Text
            key={thumb}
            size="sm"
            css={{
              position: 'absolute',
              left: `${pct(range[thumb])}%`,
              transform: 'translateX(-50%)',
              whiteSpace: 'nowrap',
            }}
          >
            {formatOffset(range[thumb])}
          </Typography.Text>
        ))}
      </div>
      {/* Per-span start markers, using the same span-type icons as the tree. */}
      <div css={{ position: 'relative', height: 20, marginInline: theme.spacing.xs }}>
        {marks.map((mark, index) => (
          <span
            key={index}
            title={`${mark.label} (${formatOffset(mark.offsetMs)})`}
            css={{
              position: 'absolute',
              left: `${pct(mark.offsetMs)}%`,
              top: 0,
              transform: 'translateX(-50%)',
              display: 'flex',
            }}
          >
            <ModelTraceExplorerIcon type={mark.icon} hasException={mark.hasException} isRootSpan={mark.isRootSpan} />
          </span>
        ))}
      </div>
      <Slider.Root
        min={min}
        max={max}
        step={step}
        value={range}
        onValueChange={(next) => onChange([next[0], next[1]])}
        style={{ width: '100%' }}
      >
        <Slider.Track>
          <Slider.Range />
        </Slider.Track>
        <Slider.Thumb aria-label="Window start" />
        <Slider.Thumb aria-label="Window end" />
      </Slider.Root>
    </div>
  );
};

export const TreeView = createComponentImplementation(TreeViewApi, ({ props }) => {
  const { theme } = useDesignSystemTheme();

  const rawNodes: RawNode[] = Array.isArray(props.nodes) ? (props.nodes as RawNode[]) : [];
  const filter = props.filter as TreeFilter | undefined;
  const title = props.title ? asString(props.title) : undefined;
  const emptyMessage = props.emptyMessage ? asString(props.emptyMessage) : 'No nodes match the current filter.';

  const timeData = useMemo(() => collectTimeData(rawNodes), [rawNodes]);
  // The selected window; reset to the full range whenever the bounds change.
  const [range, setRange] = useState<[number, number] | undefined>(undefined);
  useEffect(() => {
    setRange(timeData ? [timeData.min, timeData.max] : undefined);
  }, [timeData]);

  const visibleNodes = useMemo(() => {
    const window = timeData && range ? { t0: range[0], t1: range[1] } : undefined;
    return filterNodes(rawNodes, filter, window);
  }, [rawNodes, filter, timeData, range]);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {title && (
        <Typography.Text bold size="lg">
          {title}
        </Typography.Text>
      )}
      {timeData && range && <TimeWindowSlider timeData={timeData} range={range} onChange={setRange} />}
      {visibleNodes.length === 0 ? (
        <Typography.Text color="secondary">{emptyMessage}</Typography.Text>
      ) : (
        <div css={{ maxHeight: 360, overflowY: 'auto' }}>
          {visibleNodes.map((node, index) => (
            <TreeNodeRow key={index} node={node} depth={0} />
          ))}
        </div>
      )}
    </div>
  );
});
