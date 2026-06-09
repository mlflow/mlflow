import { useMemo, useRef, useState } from 'react';

import {
  Button,
  ChevronDownIcon,
  ChevronUpIcon,
  CloseIcon,
  Empty,
  PlusIcon,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import {
  Catalog,
  MessageProcessor,
  type A2uiMessage,
  type SurfaceModel,
} from '@a2ui/web_core/v0_9';
import { BASIC_FUNCTIONS } from '@a2ui/web_core/v0_9/basic_catalog';
import { A2uiSurface, Row, Text, type ReactComponentImplementation } from '@a2ui/react/v0_9';

import type { ModelTrace, ModelTraceInfo, ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';
import { getSpanExceptionEvents, getTotalTokens, isV3ModelTraceInfo } from '../ModelTraceExplorer.utils';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { spanTimeFormatter } from '../timeline-tree/TimelineTree.utils';
import { DataTable } from './DataTable';
import { StatCard } from './StatCard';
import { TimelineChart } from './TimelineChart';

// Must match the `catalogId` declared in `catalog.json`.
const CUSTOM_VIEW_CATALOG_ID = 'https://mlflow.org/model-trace-explorer/custom-view/catalog.json';

const formatLatencyMs = (ms: number): string => (ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`);

// Extracts the metrics we can derive from `modelTraceInfo` alone, normalizing
// across the V3 and legacy/notebook trace-info shapes.
const getMetricsFromTraceInfo = (info: ModelTrace['info']) => {
  if (isV3ModelTraceInfo(info)) {
    const totalTokens = getTotalTokens(info);
    return {
      status: info.state ?? 'STATE_UNSPECIFIED',
      latency: info.execution_duration ?? 'N/A',
      totalTokens: totalTokens != null ? totalTokens.toLocaleString() : 'N/A',
      assessments: String(info.assessments?.length ?? 0),
    };
  }

  const legacy = info as ModelTraceInfo | undefined;
  return {
    status: legacy?.status ?? 'UNKNOWN',
    latency: typeof legacy?.execution_time_ms === 'number' ? formatLatencyMs(legacy.execution_time_ms) : 'N/A',
    totalTokens: 'N/A',
    assessments: '0',
  };
};

type TraceMetrics = ReturnType<typeof getMetricsFromTraceInfo>;

// A single row for the generic DataTable: cells are positional (aligned to the
// table's columns by index), with an optional color for the leading dot.
type TableRow = { color?: string; cells: string[] };

// A single row for the generic TimelineChart: a labeled bar spanning [start, end]
// in milliseconds (relative to the trace start), with an indentation depth.
type TimelineRow = { label: string; start: number; end: number; depth: number };

// Everything a message set might need to render. Trace-level metrics come from
// `modelTraceInfo`; per-tool rows and the timeline are derived from the parsed
// spans (nodeMap / topLevelNodes).
type CustomViewData = {
  metrics: TraceMetrics;
  toolRows: TableRow[];
  timelineRows: TimelineRow[];
};

// Categorical palette for the per-tool indicator dots. Kept local so the shared
// trace-explorer code doesn't depend on the experiment-overview chart utils.
const TOOL_ROW_COLORS = ['#077A9D', '#00A972', '#FFAB00', '#E65B77', '#8A63BF', '#3986E3'];

// Aggregates TOOL-type spans (from the parsed span tree) into per-tool rows:
// call count, success rate, and average latency. Success is derived from the
// absence of exception events on the span, since the node doesn't carry a status
// code directly.
const getToolRowsFromNodeMap = (nodeMap: Record<string, ModelTraceSpanNode>): TableRow[] => {
  const statsByTool = new Map<string, { total: number; success: number; durationUs: number }>();

  for (const node of Object.values(nodeMap)) {
    if (node.type !== ModelSpanType.TOOL) {
      continue;
    }
    const toolName = typeof node.title === 'string' ? node.title : String(node.title ?? 'unknown');
    const existing = statsByTool.get(toolName) ?? { total: 0, success: 0, durationUs: 0 };
    existing.total += 1;
    if (getSpanExceptionEvents(node).length === 0) {
      existing.success += 1;
    }
    existing.durationUs += Math.max(node.end - node.start, 0);
    statsByTool.set(toolName, existing);
  }

  return Array.from(statsByTool.entries())
    .sort((a, b) => b[1].total - a[1].total)
    .map(([toolName, stats], index) => {
      const successRate = stats.total > 0 ? (stats.success / stats.total) * 100 : 0;
      const avgDurationUs = stats.total > 0 ? stats.durationUs / stats.total : 0;
      return {
        color: TOOL_ROW_COLORS[index % TOOL_ROW_COLORS.length],
        cells: [toolName, String(stats.total), `${successRate.toFixed(2)}%`, spanTimeFormatter(avgDurationUs)],
      };
    });
};

// Flattens the span tree into ordered timeline rows (depth-first, preserving the
// tree's display order), converting absolute span timestamps (microseconds) into
// offsets in milliseconds relative to the earliest span. Works for any number of
// spans / nesting depth.
const getTimelineRowsFromNodes = (topLevelNodes: ModelTraceSpanNode[]): TimelineRow[] => {
  if (topLevelNodes.length === 0) {
    return [];
  }

  const traceStartUs = Math.min(...topLevelNodes.map((node) => node.start));
  const rows: TimelineRow[] = [];

  const visit = (node: ModelTraceSpanNode, depth: number) => {
    rows.push({
      label: typeof node.title === 'string' ? node.title : String(node.title ?? 'unknown'),
      start: (node.start - traceStartUs) / 1000,
      end: (node.end - traceStartUs) / 1000,
      depth,
    });
    for (const child of node.children ?? []) {
      visit(child, depth + 1);
    }
  };

  for (const node of topLevelNodes) {
    visit(node, 0);
  }

  return rows;
};

// A message set is a named, self-contained group of A2UI messages that renders
// one block into its own surface. Add a new entry to MESSAGE_SETS to offer
// another option in the dropdown. `build` receives the target surfaceId so the
// same set can be appended multiple times into independent surfaces.
type MessageSet = {
  id: string;
  label: string;
  build: (surfaceId: string, data: CustomViewData) => A2uiMessage[];
};

const createSurfaceMessage = (surfaceId: string): A2uiMessage => ({
  version: 'v0.9',
  createSurface: {
    surfaceId,
    catalogId: CUSTOM_VIEW_CATALOG_ID,
    sendDataModel: true,
  },
});

// Trace-level summary statistics derived from `modelTraceInfo`. The StatCard
// values are bound to data-model paths populated by the `updateDataModel`
// message built from the real metrics.
const buildTraceSummaryMessages = (surfaceId: string, { metrics }: CustomViewData): A2uiMessage[] => [
  createSurfaceMessage(surfaceId),
  {
    version: 'v0.9',
    updateComponents: {
      surfaceId,
      components: [
        {
          id: 'root',
          component: 'Row',
          children: ['stat-status', 'stat-latency', 'stat-tokens', 'stat-assessments'],
          align: 'stretch',
        },
        {
          id: 'stat-status',
          component: 'StatCard',
          value: { path: '/status' },
          label: 'Status',
          icon: 'checkCircle',
          tone: 'success',
        },
        {
          id: 'stat-latency',
          component: 'StatCard',
          value: { path: '/latency' },
          label: 'Latency',
          icon: 'clock',
          tone: 'warning',
        },
        {
          id: 'stat-tokens',
          component: 'StatCard',
          value: { path: '/totalTokens' },
          label: 'Total Tokens',
          icon: 'hash',
          tone: 'info',
        },
        {
          id: 'stat-assessments',
          component: 'StatCard',
          value: { path: '/assessments' },
          label: 'Assessments',
          icon: 'checklist',
          tone: 'success',
        },
      ],
    },
  },
  {
    version: 'v0.9',
    updateDataModel: {
      surfaceId,
      value: metrics,
    },
  },
];

// Per-tool performance table derived from the trace's TOOL spans. Rows are
// inlined directly (already computed in JS) rather than bound via the data model.
const buildToolPerformanceMessages = (surfaceId: string, { toolRows }: CustomViewData): A2uiMessage[] => [
  createSurfaceMessage(surfaceId),
  {
    version: 'v0.9',
    updateComponents: {
      surfaceId,
      components: [
        {
          id: 'root',
          component: 'DataTable',
          title: 'Tool Performance Summary',
          icon: 'wrench',
          columns: [
            { label: 'Tool', align: 'left' },
            { label: 'Calls', align: 'center' },
            { label: 'Success', align: 'center' },
            { label: 'Latency (AVG)', align: 'center' },
          ],
          rows: toolRows,
          emptyMessage: 'No tool calls in this trace.',
        },
      ],
    },
  },
];

// Gantt-style breakdown of the trace's spans, inlined directly (already computed
// in JS). The TimelineChart handles axis/scaling for any number of rows.
const buildTraceBreakdownMessages = (surfaceId: string, { timelineRows }: CustomViewData): A2uiMessage[] => [
  createSurfaceMessage(surfaceId),
  {
    version: 'v0.9',
    updateComponents: {
      surfaceId,
      components: [
        {
          id: 'root',
          component: 'TimelineChart',
          title: 'Trace Breakdown',
          icon: 'clock',
          rows: timelineRows,
          emptyMessage: 'No spans in this trace.',
        },
      ],
    },
  },
];

const MESSAGE_SETS: MessageSet[] = [
  { id: 'trace-summary', label: 'Trace summary', build: buildTraceSummaryMessages },
  { id: 'tool-performance', label: 'Tool performance', build: buildToolPerformanceMessages },
  { id: 'trace-breakdown', label: 'Trace breakdown', build: buildTraceBreakdownMessages },
];

// An appended dashboard block, each backed by its own A2UI surface.
type DashboardBlock = {
  surfaceId: string;
  label: string;
};

export const ModelTraceExplorerCustomView = ({ modelTraceInfo }: { modelTraceInfo: ModelTrace['info'] }) => {
  const { theme } = useDesignSystemTheme();

  // Span data comes from the shared view-state context (the same source the
  // Summary tab uses), not from props. `nodeMap` holds every parsed span;
  // `topLevelNodes` preserves the hierarchy/order for the timeline.
  const { nodeMap, topLevelNodes } = useModelTraceExplorerViewState();

  // The catalog is the React equivalent of `catalog.json`: it maps component
  // type names to their implementations (basic Text/Row + our custom
  // StatCard/DataTable/TimelineChart).
  const catalog = useMemo(
    () =>
      new Catalog<ReactComponentImplementation>(
        CUSTOM_VIEW_CATALOG_ID,
        [Text, Row, StatCard, DataTable, TimelineChart],
        BASIC_FUNCTIONS,
      ),
    [],
  );

  // A single long-lived processor holds the state for every appended surface.
  const [processor] = useState(() => new MessageProcessor<ReactComponentImplementation>([catalog]));

  const metrics = useMemo(() => getMetricsFromTraceInfo(modelTraceInfo), [modelTraceInfo]);
  const toolRows = useMemo(() => getToolRowsFromNodeMap(nodeMap), [nodeMap]);
  const timelineRows = useMemo(() => getTimelineRowsFromNodes(topLevelNodes), [topLevelNodes]);
  const viewData = useMemo<CustomViewData>(
    () => ({ metrics, toolRows, timelineRows }),
    [metrics, toolRows, timelineRows],
  );

  const [selectedSetId, setSelectedSetId] = useState(MESSAGE_SETS[0].id);
  const [blocks, setBlocks] = useState<DashboardBlock[]>([]);
  const blockCounter = useRef(0);

  const handleAddBlock = () => {
    const messageSet = MESSAGE_SETS.find((set) => set.id === selectedSetId) ?? MESSAGE_SETS[0];
    blockCounter.current += 1;
    const surfaceId = `custom-view-${messageSet.id}-${blockCounter.current}`;

    processor.processMessages(messageSet.build(surfaceId, viewData));
    setBlocks((prev) => [...prev, { surfaceId, label: messageSet.label }]);
  };

  const handleRemoveBlock = (surfaceId: string) => {
    // Remove the block via the A2UI renderer's deleteSurface message.
    processor.processMessages([{ version: 'v0.9', deleteSurface: { surfaceId } }]);
    setBlocks((prev) => prev.filter((block) => block.surfaceId !== surfaceId));
  };

  const handleClearAll = () => {
    processor.processMessages(blocks.map((block) => ({ version: 'v0.9', deleteSurface: { surfaceId: block.surfaceId } })));
    setBlocks([]);
  };

  // Reordering only affects render order, so we just swap entries in state; the
  // underlying surfaces are untouched (React reuses them via the surfaceId key).
  const handleMoveBlock = (index: number, direction: -1 | 1) => {
    setBlocks((prev) => {
      const targetIndex = index + direction;
      if (targetIndex < 0 || targetIndex >= prev.length) {
        return prev;
      }
      const next = [...prev];
      [next[index], next[targetIndex]] = [next[targetIndex], next[index]];
      return next;
    });
  };

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        gap: theme.spacing.md,
        padding: theme.spacing.md,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'flex-end', gap: theme.spacing.sm, flexShrink: 0 }}>
        <div css={{ width: 240 }}>
          <SimpleSelect
            componentId="shared.model-trace-explorer.custom-view.message-set-select"
            id="model-trace-explorer-custom-view-message-set-select"
            label="View"
            value={selectedSetId}
            onChange={(event) => setSelectedSetId(event.target.value)}
          >
            {MESSAGE_SETS.map((set) => (
              <SimpleSelectOption key={set.id} value={set.id}>
                {set.label}
              </SimpleSelectOption>
            ))}
          </SimpleSelect>
        </div>
        <Button
          componentId="shared.model-trace-explorer.custom-view.add-block"
          icon={<PlusIcon />}
          onClick={handleAddBlock}
        >
          Add to dashboard
        </Button>
        <Button
          componentId="shared.model-trace-explorer.custom-view.clear-all"
          onClick={handleClearAll}
          disabled={blocks.length === 0}
        >
          Clear all
        </Button>
      </div>

      <div css={{ flex: 1, minHeight: 0, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {blocks.length === 0 ? (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              minHeight: 240,
              width: '100%',
              '& > div': {
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
              },
            }}
          >
            <Empty description="Select a view and click “Add to dashboard” to start building." />
          </div>
        ) : (
          blocks.map((block, index) => {
            const surface = processor.model.getSurface(block.surfaceId);
            return (
              <div
                key={block.surfaceId}
                css={{
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.borders.borderRadiusMd,
                  backgroundColor: theme.colors.backgroundPrimary,
                }}
              >
                <div
                  css={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: theme.spacing.sm,
                    padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                    borderBottom: `1px solid ${theme.colors.border}`,
                  }}
                >
                  <Typography.Text bold>{block.label}</Typography.Text>
                  <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                    <Button
                      componentId="shared.model-trace-explorer.custom-view.move-block-up"
                      size="small"
                      icon={<ChevronUpIcon />}
                      aria-label={`Move ${block.label} up`}
                      disabled={index === 0}
                      onClick={() => handleMoveBlock(index, -1)}
                    />
                    <Button
                      componentId="shared.model-trace-explorer.custom-view.move-block-down"
                      size="small"
                      icon={<ChevronDownIcon />}
                      aria-label={`Move ${block.label} down`}
                      disabled={index === blocks.length - 1}
                      onClick={() => handleMoveBlock(index, 1)}
                    />
                    <Button
                      componentId="shared.model-trace-explorer.custom-view.remove-block"
                      size="small"
                      icon={<CloseIcon />}
                      aria-label={`Remove ${block.label}`}
                      onClick={() => handleRemoveBlock(block.surfaceId)}
                    />
                  </div>
                </div>
                <div css={{ padding: theme.spacing.md }}>{surface && <A2uiSurface surface={surface} />}</div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
