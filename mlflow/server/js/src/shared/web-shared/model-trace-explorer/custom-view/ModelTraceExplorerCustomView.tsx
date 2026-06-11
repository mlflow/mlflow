import { useEffect, useMemo, useRef, useState } from 'react';

import {
  Button,
  ChevronDownIcon,
  ChevronUpIcon,
  CloseIcon,
  Empty,
  Input,
  PlusIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useEndpointsQuery } from '@mlflow/mlflow/src/gateway/hooks/useEndpointsQuery';
import {
  Catalog,
  MessageProcessor,
  type A2uiClientAction,
  type A2uiMessage,
  type SurfaceModel,
} from '@a2ui/web_core/v0_9';
import { BASIC_FUNCTIONS } from '@a2ui/web_core/v0_9/basic_catalog';
import { A2uiSurface, Column, Row, Text, type ReactComponentImplementation } from '@a2ui/react/v0_9';

import type { Assessment, Feedback, ModelTrace, ModelTraceInfo, ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';
import {
  getIconTypeForSpan,
  getSpanExceptionEvents,
  getSpanLogLevel,
  getTotalTokens,
  isV3ModelTraceInfo,
} from '../ModelTraceExplorer.utils';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { getUser } from '../../global-settings/getUser';
import { useCreateAssessment } from '../hooks/useCreateAssessment';
import { spanTimeFormatter } from '../timeline-tree/TimelineTree.utils';
import { AssessmentBoard } from './AssessmentBoard';
import { AssessmentCard } from './AssessmentCard';
import { Card } from './Card';
import { DataTable } from './DataTable';
import { DEFAULT_FEEDBACK_NAME, FEEDBACK_SUBMITTED, FeedbackButtons } from './FeedbackButtons';
import { Icon } from './Icon';
import { KeyValueViewer } from './KeyValueViewer';
import { Markdown } from './Markdown';
import { MediaRenderer } from './MediaRenderer';
import { StatCard } from './StatCard';
import { TimelineChart } from './TimelineChart';
import { type PanelItem } from './TreeSelectionContext';
import { TreeNode } from './TreeNode';
import { TREE_NODE_SELECTED, TreeView } from './TreeView';
import type { AgentAssessment, AgentNode } from './agent/buildAgentPrompt';
import { useAgentDashboard } from './agent/useAgentDashboard';

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

// A reference span-tree node handed to Agent Mode (in the data snapshot) so the
// LLM can construct TreeNode components. `attributes` is an opaque metadata bag
// (type/logLevel/duration) the model can use to select a subset of spans.
type TreeNodeData = {
  id: string;
  label: string;
  icon: string;
  hasException: boolean;
  isRootSpan: boolean;
  badge?: string;
  attributes: {
    type: string;
    hasException: boolean;
    logLevel: number;
    durationMs: number;
  };
  children: TreeNodeData[];
};

// Everything a message set might need to render. Trace-level metrics come from
// `modelTraceInfo`; per-tool rows, the timeline, and the tree are derived from
// the parsed spans (nodeMap / topLevelNodes).
// A single key/value entry (e.g. for a KeyValueViewer). `value` is JSON-encoded
// so the snippet renderer can show it as JSON (objects) or text/markdown (strings).
type ContentField = { label: string; value: string };

type AssessmentSentiment = 'positive' | 'negative' | 'neutral' | 'error';

type AssessmentBoardItem = {
  name: string;
  value?: string;
  rationale?: string;
  source?: string;
  sentiment: AssessmentSentiment;
};

// One attribute extracted from the first tool call: a key/value pair where
// `value` is a JSON-encoded string (ready for KeyValueViewer).
type FirstToolIO = {
  toolName: string;
  input?: ContentField;
  output?: ContentField;
};

type CustomViewData = {
  metrics: TraceMetrics;
  toolRows: TableRow[];
  timelineRows: TimelineRow[];
  treeNodes: TreeNodeData[];
  // The span hierarchy (roots), used by the predefined tree/trajectory builders
  // to emit TreeNode components with per-span side panels.
  treeRoots: ModelTraceSpanNode[];
  assessmentItems: AssessmentBoardItem[];
  firstToolIO?: FirstToolIO;
};

// Turns an arbitrary inputs/outputs payload into key/value fields. Objects
// become one field per top-level key (mirroring the Details view's key/value
// list); anything else becomes a single unlabeled field.
const getContentFields = (payload: unknown): ContentField[] => {
  if (payload === null || payload === undefined) {
    return [];
  }
  if (typeof payload === 'object' && !Array.isArray(payload)) {
    return Object.entries(payload as Record<string, unknown>).map(([key, value]) => ({
      label: key,
      value: JSON.stringify(value, null, 2),
    }));
  }
  return [{ label: '', value: JSON.stringify(payload, null, 2) }];
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

// Maps the span tree into generic TreeView nodes. We reuse the trace explorer's
// own icon mapping (getIconTypeForSpan) and log-level/exception helpers so the
// tree matches the Details & Timeline view, and stash the filterable fields in
// `attributes` for the TreeView's structured filter.
const getTreeNodesFromNodes = (topLevelNodes: ModelTraceSpanNode[]): TreeNodeData[] => {
  const toTreeNode = (node: ModelTraceSpanNode): TreeNodeData => {
    const hasException = getSpanExceptionEvents(node).length > 0;
    const assessmentCount = node.assessments?.length ?? 0;
    return {
      id: String(node.key),
      label: typeof node.title === 'string' ? node.title : String(node.title ?? 'unknown'),
      icon: getIconTypeForSpan(node.type ?? ModelSpanType.UNKNOWN),
      hasException,
      isRootSpan: !node.parentId,
      badge: assessmentCount > 0 ? String(assessmentCount) : undefined,
      attributes: {
        type: node.type ?? ModelSpanType.UNKNOWN,
        hasException,
        logLevel: getSpanLogLevel(node),
        durationMs: Math.max(node.end - node.start, 0) / 1000,
      },
      children: (node.children ?? []).map(toTreeNode),
    };
  };

  return topLevelNodes.map(toTreeNode);
};

// Extracts the displayable value + error from any assessment variant
// (feedback / expectation), so the agent receives real judge/feedback results.
const getAssessmentValueAndError = (assessment: Assessment): { value: unknown; error?: string } => {
  if ('feedback' in assessment && assessment.feedback) {
    const err = assessment.feedback.error ?? assessment.error;
    return {
      value: assessment.feedback.value,
      error: err ? err.error_message ?? err.error_code : undefined,
    };
  }
  if ('expectation' in assessment && assessment.expectation) {
    const expectation = assessment.expectation;
    if ('value' in expectation) {
      return { value: expectation.value };
    }
    if ('serialized_value' in expectation) {
      return { value: expectation.serialized_value.value };
    }
  }
  return { value: undefined, error: assessment.error?.error_message ?? assessment.error?.error_code };
};

// Collects the trace's real assessments (trace-level + span-level), deduped by
// id and skipping invalidated ones, into the flat shape the agent prompt uses.
const getAgentAssessments = (
  info: ModelTrace['info'],
  nodeMap: Record<string, ModelTraceSpanNode>,
): AgentAssessment[] => {
  const byId = new Map<string, AgentAssessment>();
  const add = (assessment: Assessment) => {
    if (assessment.valid === false || byId.has(assessment.assessment_id)) {
      return;
    }
    const { value, error } = getAssessmentValueAndError(assessment);
    byId.set(assessment.assessment_id, {
      name: assessment.assessment_name,
      value,
      rationale: assessment.rationale,
      source: assessment.source?.source_type ?? 'SOURCE_TYPE_UNSPECIFIED',
      spanId: assessment.span_id,
      error,
    });
  };

  const traceAssessments = (info as { assessments?: Assessment[] } | undefined)?.assessments ?? [];
  for (const assessment of traceAssessments) {
    add(assessment);
  }
  for (const node of Object.values(nodeMap)) {
    for (const assessment of node.assessments ?? []) {
      add(assessment);
    }
  }
  return Array.from(byId.values());
};

// Maps an assessment's raw value to a verdict polarity for coloring. Affirmative
// values (yes/true/pass/correct) are positive (green); negatives (no/false/fail)
// are negative (red); an error overrides everything; otherwise neutral.
const POSITIVE_VALUES = new Set(['yes', 'true', 'pass', 'passed', 'correct', 'good', 'success']);
const NEGATIVE_VALUES = new Set(['no', 'false', 'fail', 'failed', 'incorrect', 'bad', 'failure']);

const getAssessmentSentiment = ({ value, error }: AgentAssessment): AssessmentSentiment => {
  if (error) {
    return 'error';
  }
  if (typeof value === 'boolean') {
    return value ? 'positive' : 'negative';
  }
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (POSITIVE_VALUES.has(normalized)) {
      return 'positive';
    }
    if (NEGATIVE_VALUES.has(normalized)) {
      return 'negative';
    }
  }
  return 'neutral';
};

// Shapes the trace's real assessments into AssessmentBoard items for the
// predefined "LLM-as-a-judge" view: category header, verdict value, rationale,
// and a derived sentiment that drives the green/red coloring.
const getAssessmentBoardItems = (assessments: AgentAssessment[]): AssessmentBoardItem[] =>
  assessments.map((assessment) => {
    const hasError = Boolean(assessment.error);
    const hasValue = assessment.value !== undefined && assessment.value !== null;
    return {
      name: assessment.name,
      // Keep the badge short: errors show "Error" and surface the message in the
      // body, otherwise a long value would blow out the badge/card layout.
      value: hasError ? 'Error' : hasValue ? String(assessment.value) : undefined,
      rationale: assessment.rationale ?? (hasError ? assessment.error : undefined),
      source: assessment.source,
      sentiment: getAssessmentSentiment(assessment),
    };
  });

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

// Same trace-summary stats, but grouped inside a basic `Card`. Demonstrates the
// Card primitive's single-child rule: the card's `child` is a Column, which in
// turn holds a heading Text and a Row of StatCards. StatCard values are bound to
// data-model paths populated by the trailing updateDataModel message.
const buildTraceSummaryCardMessages = (surfaceId: string, { metrics }: CustomViewData): A2uiMessage[] => [
  createSurfaceMessage(surfaceId),
  {
    version: 'v0.9',
    updateComponents: {
      surfaceId,
      components: [
        // Card wraps exactly one child — a Column that groups everything.
        { id: 'root', component: 'Card', child: 'card-body' },
        { id: 'card-body', component: 'Column', children: ['card-heading', 'card-stats'] },
        { id: 'card-heading', component: 'Text', text: 'Trace Summary', variant: 'h4' },
        {
          id: 'card-stats',
          component: 'Row',
          children: ['card-stat-status', 'card-stat-latency', 'card-stat-tokens', 'card-stat-assessments'],
          align: 'stretch',
        },
        {
          id: 'card-stat-status',
          component: 'StatCard',
          value: { path: '/status' },
          label: 'Status',
          icon: 'checkCircle',
          tone: 'success',
        },
        {
          id: 'card-stat-latency',
          component: 'StatCard',
          value: { path: '/latency' },
          label: 'Latency',
          icon: 'clock',
          tone: 'warning',
        },
        {
          id: 'card-stat-tokens',
          component: 'StatCard',
          value: { path: '/totalTokens' },
          label: 'Total Tokens',
          icon: 'hash',
          tone: 'info',
        },
        {
          id: 'card-stat-assessments',
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

// Demonstrates the custom MediaRenderer component: an image rendered from a
// direct, public URL with a caption, grouped in a Card. Swap DEMO_IMAGE_URL for
// any direct image link. NOTE: Google Drive *share* links are NOT direct image
// URLs (they serve an HTML sign-in/preview page, not raw image bytes), so they
// won't render here — use a link that ends in the image bytes (e.g. .png/.jpg).
// The same component also accepts an `mlflow-attachment://` URI, which it
// fetches from the trace artifact store and renders as an image/audio/PDF blob.
const DEMO_IMAGE_URL = 'https://cdn.britannica.com/77/170477-050-1C747EE3/Laptop-computer.jpg';

const buildMediaDemoMessages = (surfaceId: string): A2uiMessage[] => [
  createSurfaceMessage(surfaceId),
  {
    version: 'v0.9',
    updateComponents: {
      surfaceId,
      components: [
        { id: 'root', component: 'Card', child: 'image-col' },
        { id: 'image-col', component: 'Column', children: ['demo-image', 'image-caption'] },
        {
          id: 'demo-image',
          component: 'MediaRenderer',
          url: DEMO_IMAGE_URL,
          alt: 'A2UI MediaRenderer component demo',
        },
        {
          id: 'image-caption',
          component: 'Text',
          text: 'Rendered with the custom A2UI MediaRenderer component (URL or mlflow-attachment:// blob).',
        },
      ],
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

// Returns the span's "real" (non-`mlflow.`-prefixed) attributes, mirroring the
// Details & Timeline Attributes tab.
const getSpanAttributes = (span?: ModelTraceSpanNode): Record<string, unknown> => {
  if (!span?.attributes) {
    return {};
  }
  return Object.fromEntries(Object.entries(span.attributes).filter(([key]) => !key.startsWith('mlflow.')));
};

// Builds the side-panel subtree for a selected TreeNode from its lightweight
// `panelItems` directives plus the span's real data (from `nodeMap`). The author
// /LLM only emits the directives; the host materializes the heavy components
// (KeyValueViewer / Markdown / FeedbackButtons) here, keyed off the deterministic
// `${nodeId}__panel` id the TreeView renders. Runs in the action handler in
// response to a TREE_NODE_SELECTED action.
const buildSpanPanelComponents = (
  nodeId: string,
  spanId: string | undefined,
  panelItems: PanelItem[],
  nodeMap: Record<string, ModelTraceSpanNode>,
): Record<string, unknown>[] => {
  const panelRootId = `${nodeId}__panel`;
  const span = spanId ? nodeMap[spanId] : undefined;
  const childIds: string[] = [];
  const components: Record<string, unknown>[] = [];

  panelItems.forEach((item, index) => {
    const itemId = `${panelRootId}-item-${index}`;
    switch (item.type) {
      case 'input':
      case 'output':
      case 'attributes': {
        const value =
          item.type === 'input' ? span?.inputs : item.type === 'output' ? span?.outputs : getSpanAttributes(span);
        const defaultLabel = item.type === 'input' ? 'Inputs' : item.type === 'output' ? 'Outputs' : 'Attributes';
        childIds.push(itemId);
        components.push({
          id: itemId,
          component: 'KeyValueViewer',
          label: item.title || defaultLabel,
          value: JSON.stringify(value ?? null),
          initialFormat: 'json',
        });
        break;
      }
      case 'markdown': {
        childIds.push(itemId);
        components.push({
          id: itemId,
          component: 'Markdown',
          text: item.text ?? '',
          ...(item.title ? { title: item.title } : {}),
        });
        break;
      }
      case 'feedback': {
        childIds.push(itemId);
        components.push({
          id: itemId,
          component: 'FeedbackButtons',
          label: item.label || 'Was this span helpful?',
          name: item.name || 'Span helpfulness',
          ...(spanId ? { spanId } : {}),
          value: { path: `/feedback/${nodeId}` },
        });
        break;
      }
      default:
        break;
    }
  });

  return [{ id: panelRootId, component: 'Column', children: childIds }, ...components];
};

// A span tree built from first-class TreeNode components. Each node maps 1:1 to
// a span and carries `panelItems` directives (input / output / feedback). The
// host builds the actual side panel from the span's data when the node is
// selected. Demonstrates the host-built, author-directed side panel.
const buildTraceTreeMessages = (surfaceId: string, { treeRoots }: CustomViewData): A2uiMessage[] => {
  const components: Record<string, unknown>[] = [];
  let counter = 0;

  const toComponents = (span: ModelTraceSpanNode): string => {
    counter += 1;
    const nodeId = `tn-${counter}-node`;
    const childIds = (span.children ?? []).map(toComponents);

    const hasException = getSpanExceptionEvents(span).length > 0;
    const assessmentCount = span.assessments?.length ?? 0;
    components.push({
      id: nodeId,
      component: 'TreeNode',
      label: typeof span.title === 'string' ? span.title : String(span.title ?? 'unknown'),
      icon: getIconTypeForSpan(span.type ?? ModelSpanType.UNKNOWN),
      hasException,
      isRootSpan: !span.parentId,
      ...(assessmentCount > 0 ? { badge: String(assessmentCount) } : {}),
      spanId: String(span.key),
      panelItems: [{ type: 'input' }, { type: 'output' }, { type: 'feedback' }],
      ...(childIds.length > 0 ? { children: childIds } : {}),
    });
    return nodeId;
  };

  const rootChildIds = treeRoots.map(toComponents);

  return [
    createSurfaceMessage(surfaceId),
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId,
        components: [
          {
            id: 'root',
            component: 'TreeView',
            title: 'Trace Tree',
            children: rootChildIds,
            emptyMessage: 'No spans to display.',
          },
          ...components,
        ],
      },
    },
  ];
};

// Demonstrates the milestone/trajectory use case: each TreeNode is a high-level
// step whose side panel is a markdown summary that deeplinks to its span via
// [text](#span:<id>), plus span-scoped feedback. Selecting a milestone (or
// following its deeplink) asks the host to build that panel. Built from the real
// top-level spans (no fabricated narrative).
const buildTrajectoryDemoMessages = (surfaceId: string, { treeRoots }: CustomViewData): A2uiMessage[] => {
  const milestones = treeRoots.slice(0, 6);
  if (milestones.length === 0) {
    return [
      createSurfaceMessage(surfaceId),
      {
        version: 'v0.9',
        updateComponents: {
          surfaceId,
          components: [{ id: 'root', component: 'Text', text: 'No spans to summarize in this trace.' }],
        },
      },
    ];
  }

  const components: Record<string, unknown>[] = [];
  const milestoneIds: string[] = [];

  milestones.forEach((span, index) => {
    const nodeId = `ms-${index + 1}-node`;
    milestoneIds.push(nodeId);

    const spanName = typeof span.title === 'string' ? span.title : String(span.title ?? 'span');
    const spanType = String(span.type ?? ModelSpanType.UNKNOWN);
    const spanKey = String(span.key);

    components.push({
      id: nodeId,
      component: 'TreeNode',
      title: `Step ${index + 1}: ${spanName}`,
      icon: getIconTypeForSpan(span.type ?? ModelSpanType.UNKNOWN),
      isRootSpan: !span.parentId,
      spanId: spanKey,
      panelItems: [
        {
          type: 'markdown',
          title: `Step ${index + 1}`,
          text: `Milestone covering the \`${spanType}\` span **${spanName}**. Jump to the span [here](#span:${spanKey}).`,
        },
        { type: 'feedback' },
      ],
    });
  });

  return [
    createSurfaceMessage(surfaceId),
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId,
        components: [
          {
            id: 'root',
            component: 'TreeView',
            title: 'Agent Trajectory',
            children: milestoneIds,
            emptyMessage: 'No spans to summarize.',
          },
          ...components,
        ],
      },
    },
  ];
};

// One AssessmentCard per LLM-as-a-judge / human assessment, laid out in a
// wrapping AssessmentBoard. Each card is a reusable catalog primitive, so more
// assessments simply mean more cards appended to the board's children.
const buildAssessmentsMessages = (surfaceId: string, { assessmentItems }: CustomViewData): A2uiMessage[] => {
  const cardIds = assessmentItems.map((_, index) => `assessment-${index}`);
  return [
    createSurfaceMessage(surfaceId),
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId,
        components: [
          {
            id: 'root',
            component: 'AssessmentBoard',
            title: 'LLM-as-a-Judge Assessments',
            icon: 'checklist',
            children: cardIds,
            emptyMessage: 'No assessments on this trace.',
          },
          ...assessmentItems.map((item, index) => ({
            id: cardIds[index],
            component: 'AssessmentCard',
            name: item.name,
            ...(item.value !== undefined ? { value: item.value } : {}),
            ...(item.rationale !== undefined ? { rationale: item.rationale } : {}),
            ...(item.source !== undefined ? { source: item.source } : {}),
            sentiment: item.sentiment,
          })),
        ],
      },
    },
  ];
};

// Two KeyValueViewers side by side (in a Row): one input attribute and one
// output attribute from the first tool call. Demonstrates the single-value
// primitive and side-by-side layout. Each value is a JSON-encoded string.
const buildFirstToolIOMessages = (surfaceId: string, { firstToolIO }: CustomViewData): A2uiMessage[] => {
  if (!firstToolIO || (!firstToolIO.input && !firstToolIO.output)) {
    return [
      createSurfaceMessage(surfaceId),
      {
        version: 'v0.9',
        updateComponents: {
          surfaceId,
          components: [{ id: 'root', component: 'Text', text: 'No tool calls with inputs/outputs in this trace.' }],
        },
      },
    ];
  }

  const components: Record<string, unknown>[] = [];
  const children: string[] = [];
  if (firstToolIO.input) {
    children.push('tool-input');
    components.push({
      id: 'tool-input',
      component: 'KeyValueViewer',
      label: `Input · ${firstToolIO.input.label || 'value'}`,
      value: firstToolIO.input.value,
    });
  }
  if (firstToolIO.output) {
    children.push('tool-output');
    components.push({
      id: 'tool-output',
      component: 'KeyValueViewer',
      label: `Output · ${firstToolIO.output.label || 'value'}`,
      value: firstToolIO.output.value,
    });
  }

  return [
    createSurfaceMessage(surfaceId),
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId,
        components: [{ id: 'root', component: 'Row', children, align: 'start' }, ...components],
      },
    },
  ];
};

// Demonstrates the interactive FeedbackButtons primitive: a labeled thumbs
// up/down control bound to `/feedback`. Clicking highlights the choice and logs
// an MLflow feedback assessment (handled by the host's action handler).
const buildFeedbackDemoMessages = (surfaceId: string): A2uiMessage[] => [
  createSurfaceMessage(surfaceId),
  {
    version: 'v0.9',
    updateDataModel: { surfaceId, path: '/feedback', value: null },
  },
  {
    version: 'v0.9',
    updateComponents: {
      surfaceId,
      components: [
        { id: 'root', component: 'Card', child: 'feedback-buttons' },
        {
          id: 'feedback-buttons',
          component: 'FeedbackButtons',
          label: 'Was this trace helpful?',
          name: 'Trace helpfulness',
          value: { path: '/feedback' },
        },
      ],
    },
  },
];

const MESSAGE_SETS: MessageSet[] = [
  { id: 'trace-summary', label: 'Show me the high level summary of this trace', build: buildTraceSummaryMessages },
  { id: 'trace-summary-card', label: 'Show the trace summary grouped in a card', build: buildTraceSummaryCardMessages },
  { id: 'image-demo', label: 'Show an image (MediaRenderer component demo)', build: buildMediaDemoMessages },
  { id: 'feedback-demo', label: 'Collect thumbs up/down feedback', build: buildFeedbackDemoMessages },
  { id: 'tool-performance', label: 'List performance summary for all tools', build: buildToolPerformanceMessages },
  { id: 'trace-breakdown', label: 'Give me a timeline of all spans calls', build: buildTraceBreakdownMessages },
  { id: 'trace-tree', label: 'Show me the span calls in a tree view', build: buildTraceTreeMessages },
  { id: 'trajectory-demo', label: 'Summarize the agent trajectory as milestones', build: buildTrajectoryDemoMessages },
  { id: 'assessments', label: 'Show the LLM-as-a-judge assessments', build: buildAssessmentsMessages },
  {
    id: 'first-tool-io',
    label: "Compare the first tool call's input and output",
    build: buildFirstToolIOMessages,
  },
];

// An appended dashboard block, each backed by its own A2UI surface. `setId`
// identifies which message set produced it. (Span detail / side panels are now
// owned by the TreeView component itself, in-surface.)
type DashboardBlock = {
  surfaceId: string;
  label: string;
  setId: string;
};

export const ModelTraceExplorerCustomView = ({ modelTraceInfo }: { modelTraceInfo: ModelTrace['info'] }) => {
  const { theme } = useDesignSystemTheme();

  // Span data comes from the shared view-state context (the same source the
  // Summary tab uses), not from props. `nodeMap` holds every parsed span;
  // `topLevelNodes` preserves the hierarchy/order for the timeline.
  const { nodeMap, topLevelNodes } = useModelTraceExplorerViewState();

  // The catalog is the React equivalent of `catalog.json`: it maps component
  // type names to their implementations (basic Text/Row/Column + our custom
  // MediaRenderer/Card/Icon/StatCard/DataTable/TimelineChart/TreeView/TreeNode/Markdown/KeyValueViewer/FeedbackButtons/...).
  const catalog = useMemo(
    () =>
      new Catalog<ReactComponentImplementation>(
        CUSTOM_VIEW_CATALOG_ID,
        [
          Text,
          Row,
          Column,
          MediaRenderer,
          Card,
          Icon,
          StatCard,
          DataTable,
          TimelineChart,
          TreeView,
          TreeNode,
          Markdown,
          AssessmentBoard,
          AssessmentCard,
          KeyValueViewer,
          FeedbackButtons,
        ],
        BASIC_FUNCTIONS,
      ),
    [],
  );

  // Persist thumbs up/down clicks as real MLflow feedback assessments. The
  // processor's action handler is created once, so we route through a ref that
  // always points at the latest mutation/traceId.
  const traceId = useMemo(
    () => (isV3ModelTraceInfo(modelTraceInfo) ? modelTraceInfo.trace_id : (modelTraceInfo.request_id ?? '')),
    [modelTraceInfo],
  );
  const { createAssessmentMutation } = useCreateAssessment({ traceId });

  // The processor's action handler is created once, so we route through a ref
  // that always points at the latest mutation / traceId / nodeMap.
  const actionHandlerRef = useRef<(action: A2uiClientAction) => void>(() => {});

  const handleFeedbackAction = (action: A2uiClientAction) => {
    const context = action.context ?? {};
    const value = context.value;
    if (typeof value !== 'boolean') {
      return;
    }
    const name = typeof context.name === 'string' && context.name ? context.name : DEFAULT_FEEDBACK_NAME;
    const spanId = typeof context.spanId === 'string' && context.spanId ? context.spanId : undefined;
    // Spread a typed value object (matching AssessmentCreateForm) so the literal
    // narrows to the FeedbackAssessment member of the Assessment union.
    const feedbackValue: { feedback: Feedback } = { feedback: { value } };
    createAssessmentMutation({
      assessment: {
        assessment_name: name,
        trace_id: traceId,
        source: { source_type: 'HUMAN', source_id: getUser() ?? '' },
        ...(spanId ? { span_id: spanId } : {}),
        ...feedbackValue,
      },
    });
  };

  // When a TreeNode is selected, build its side panel from the node's panelItems
  // directives + the span's real data, and inject it into the same surface as a
  // Column at the deterministic `${nodeId}__panel` id the TreeView renders.
  const handleTreeNodeSelected = (action: A2uiClientAction) => {
    const context = action.context ?? {};
    const nodeId = typeof context.nodeId === 'string' && context.nodeId ? context.nodeId : undefined;
    if (!nodeId) {
      return;
    }
    const spanId = typeof context.spanId === 'string' && context.spanId ? context.spanId : undefined;
    const panelItems = Array.isArray(context.panelItems) ? (context.panelItems as PanelItem[]) : [];
    const components = buildSpanPanelComponents(nodeId, spanId, panelItems, nodeMap);
    processor.processMessages([{ version: 'v0.9', updateComponents: { surfaceId: action.surfaceId, components } }]);
  };

  actionHandlerRef.current = (action: A2uiClientAction) => {
    if (action.name === FEEDBACK_SUBMITTED) {
      handleFeedbackAction(action);
    } else if (action.name === TREE_NODE_SELECTED) {
      handleTreeNodeSelected(action);
    }
  };

  // A single long-lived processor holds the state for every appended surface.
  const [processor] = useState(
    () => new MessageProcessor<ReactComponentImplementation>([catalog], (action) => actionHandlerRef.current(action)),
  );

  // The tree starts one layer below the trace root (e.g. omit the top-level
  // `chat_agent` agent span), since that wrapper span is rarely useful. Falls
  // back to the top-level nodes if the root has no children.
  const treeRoots = useMemo(() => {
    const children = topLevelNodes.flatMap((node) => node.children ?? []);
    return children.length > 0 ? children : topLevelNodes;
  }, [topLevelNodes]);

  const metrics = useMemo(() => getMetricsFromTraceInfo(modelTraceInfo), [modelTraceInfo]);
  const toolRows = useMemo(() => getToolRowsFromNodeMap(nodeMap), [nodeMap]);
  const timelineRows = useMemo(() => getTimelineRowsFromNodes(topLevelNodes), [topLevelNodes]);
  const treeNodes = useMemo(() => getTreeNodesFromNodes(treeRoots), [treeRoots]);

  // Real assessments (LLM-judge / human feedback), used by both the predefined
  // "LLM-as-a-judge" board and Agent Mode (so the model shows real results).
  const agentAssessments = useMemo(() => getAgentAssessments(modelTraceInfo, nodeMap), [modelTraceInfo, nodeMap]);
  const assessmentItems = useMemo(() => getAssessmentBoardItems(agentAssessments), [agentAssessments]);

  // The first tool call's first input attribute + first output attribute, for
  // the KeyValueViewer side-by-side demo. Values are JSON-encoded by
  // getContentFields, ready for KeyValueViewer.
  const firstToolIO = useMemo<FirstToolIO | undefined>(() => {
    const toolNodes = Object.values(nodeMap)
      .filter((node) => node.type === ModelSpanType.TOOL)
      .sort((a, b) => a.start - b.start);
    if (toolNodes.length === 0) {
      return undefined;
    }
    const tool = toolNodes[0];
    return {
      toolName: typeof tool.title === 'string' ? tool.title : String(tool.title ?? 'tool'),
      input: getContentFields(tool.inputs)[0],
      output: getContentFields(tool.outputs)[0],
    };
  }, [nodeMap]);

  const viewData = useMemo<CustomViewData>(
    () => ({ metrics, toolRows, timelineRows, treeNodes, treeRoots, assessmentItems, firstToolIO }),
    [metrics, toolRows, timelineRows, treeNodes, treeRoots, assessmentItems, firstToolIO],
  );

  // The trace's nodeMap as plain JSON (keyed by span id) for Agent Mode. The LLM
  // parses this to extract the data it needs (including span inputs/outputs) and
  // binds it into components via the A2UI data model.
  const agentNodeMap = useMemo(() => {
    const nodes = Object.values(nodeMap);
    if (nodes.length === 0) {
      return {};
    }
    const traceStartUs = Math.min(...nodes.map((node) => node.start));
    const json: Record<string, AgentNode> = {};
    for (const node of nodes) {
      json[String(node.key)] = {
        name: typeof node.title === 'string' ? node.title : String(node.title ?? 'unknown'),
        type: node.type ?? ModelSpanType.UNKNOWN,
        startMs: Math.max(node.start - traceStartUs, 0) / 1000,
        endMs: Math.max(node.end - traceStartUs, 0) / 1000,
        durationMs: Math.max(node.end - node.start, 0) / 1000,
        parentId: node.parentId ? String(node.parentId) : undefined,
        inputs: node.inputs,
        outputs: node.outputs,
      };
    }
    return json;
  }, [nodeMap]);

  const [selectedSetId, setSelectedSetId] = useState(MESSAGE_SETS[0].id);
  const [blocks, setBlocks] = useState<DashboardBlock[]>([]);
  const blockCounter = useRef(0);

  // 'predefined' appends a canned message set from the dropdown; 'agent' asks an
  // LLM (via a gateway endpoint) to generate the A2UI message stream.
  const [viewMode, setViewMode] = useState<'predefined' | 'agent'>('predefined');
  const { data: endpoints, isLoading: endpointsLoading } = useEndpointsQuery();
  const [selectedEndpoint, setSelectedEndpoint] = useState('');
  const [instruction, setInstruction] = useState('');
  const { generate, isLoading: agentLoading, error: agentError, reset: resetAgent } = useAgentDashboard();

  // Default to the first available endpoint once the list loads.
  useEffect(() => {
    if (!selectedEndpoint && endpoints.length > 0) {
      setSelectedEndpoint(endpoints[0].name);
    }
  }, [endpoints, selectedEndpoint]);

  const handleAddBlock = () => {
    const messageSet = MESSAGE_SETS.find((set) => set.id === selectedSetId) ?? MESSAGE_SETS[0];
    blockCounter.current += 1;
    const surfaceId = `custom-view-${messageSet.id}-${blockCounter.current}`;

    processor.processMessages(messageSet.build(surfaceId, viewData));
    setBlocks((prev) => [
      ...prev,
      {
        surfaceId,
        label: messageSet.label,
        setId: messageSet.id,
      },
    ]);
  };

  // Generates a dashboard block from the user's instruction via the LLM. The
  // returned messages are already validated + normalized to our surface id, so
  // we can hand them straight to the processor.
  const handleGenerateAgentBlock = async () => {
    const endpointName = selectedEndpoint;
    const prompt = instruction.trim();
    if (!endpointName || !prompt) {
      return;
    }
    blockCounter.current += 1;
    const surfaceId = `custom-view-agent-${blockCounter.current}`;
    try {
      const messages = await generate({
        instruction: prompt,
        endpointName,
        surfaceId,
        catalogId: CUSTOM_VIEW_CATALOG_ID,
        data: { ...viewData, nodeMap: agentNodeMap, assessments: agentAssessments },
      });
      processor.processMessages(messages);
      setBlocks((prev) => [...prev, { surfaceId, label: prompt, setId: 'agent' }]);
    } catch {
      // The failure is surfaced via `agentError` below; the block is not added.
    }
  };

  const handleRemoveBlock = (block: DashboardBlock) => {
    // Remove the block via the A2UI renderer's deleteSurface message.
    processor.processMessages([{ version: 'v0.9', deleteSurface: { surfaceId: block.surfaceId } }]);
    setBlocks((prev) => prev.filter((entry) => entry.surfaceId !== block.surfaceId));
  };

  const handleClearAll = () => {
    processor.processMessages(
      blocks.map((block) => ({ version: 'v0.9', deleteSurface: { surfaceId: block.surfaceId } })),
    );
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
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, flexShrink: 0 }}>
        <div css={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: theme.spacing.sm }}>
          <SegmentedControlGroup
            name="custom-view-mode"
            componentId="shared.model-trace-explorer.custom-view.mode-toggle"
            value={viewMode}
            onChange={(event) => {
              setViewMode(event.target.value);
              resetAgent();
            }}
          >
            <SegmentedControlButton value="predefined">Predefined Prompts</SegmentedControlButton>
            <SegmentedControlButton value="agent">Agent Mode</SegmentedControlButton>
          </SegmentedControlGroup>
          <Button
            componentId="shared.model-trace-explorer.custom-view.clear-all"
            onClick={handleClearAll}
            disabled={blocks.length === 0}
          >
            Clear all
          </Button>
        </div>

        {viewMode === 'predefined' ? (
          <div css={{ display: 'flex', alignItems: 'flex-end', gap: theme.spacing.sm }}>
            <div css={{ width: 380 }}>
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
          </div>
        ) : (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {endpoints.length === 0 ? (
              <Typography.Text color="secondary">
                {endpointsLoading
                  ? 'Loading AI gateway endpoints…'
                  : 'No AI gateway endpoints are configured. Add one to use Agent Mode.'}
              </Typography.Text>
            ) : (
              <>
                <div css={{ display: 'flex', alignItems: 'flex-end', gap: theme.spacing.sm }}>
                  <div css={{ width: 240 }}>
                    <SimpleSelect
                      componentId="shared.model-trace-explorer.custom-view.agent-endpoint-select"
                      id="model-trace-explorer-custom-view-agent-endpoint-select"
                      label="AI endpoint"
                      value={selectedEndpoint}
                      onChange={(event) => setSelectedEndpoint(event.target.value)}
                    >
                      {endpoints.map((endpoint) => (
                        <SimpleSelectOption key={endpoint.name} value={endpoint.name}>
                          {endpoint.name}
                        </SimpleSelectOption>
                      ))}
                    </SimpleSelect>
                  </div>
                  <Button
                    componentId="shared.model-trace-explorer.custom-view.agent-generate"
                    icon={<PlusIcon />}
                    loading={agentLoading}
                    disabled={!selectedEndpoint || !instruction.trim() || agentLoading}
                    onClick={handleGenerateAgentBlock}
                  >
                    Generate
                  </Button>
                </div>
                <Input.TextArea
                  componentId="shared.model-trace-explorer.custom-view.agent-instruction"
                  placeholder="Describe the dashboard to generate, e.g. “Show a table of tool latencies and a timeline of all spans”."
                  value={instruction}
                  autoSize={{ minRows: 2, maxRows: 5 }}
                  onKeyDown={(event) => event.stopPropagation()}
                  onChange={(event) => setInstruction(event.target.value)}
                  disabled={agentLoading}
                />
                {agentError && (
                  <Typography.Text size="sm" css={{ color: theme.colors.textValidationDanger }}>
                    {agentError.message}
                  </Typography.Text>
                )}
              </>
            )}
          </div>
        )}
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
                  <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
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
                      onClick={() => handleRemoveBlock(block)}
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
