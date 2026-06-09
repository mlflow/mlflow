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
  Switch,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useEndpointsQuery } from '@mlflow/mlflow/src/gateway/hooks/useEndpointsQuery';
import {
  Catalog,
  MessageProcessor,
  type A2uiMessage,
  type SurfaceModel,
} from '@a2ui/web_core/v0_9';
import { BASIC_FUNCTIONS } from '@a2ui/web_core/v0_9/basic_catalog';
import { A2uiSurface, Column, Row, Text, type ReactComponentImplementation } from '@a2ui/react/v0_9';

import type { Assessment, ModelTrace, ModelTraceInfo, ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';
import {
  getIconTypeForSpan,
  getSpanExceptionEvents,
  getSpanLogLevel,
  getTotalTokens,
  isV3ModelTraceInfo,
} from '../ModelTraceExplorer.utils';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { spanTimeFormatter } from '../timeline-tree/TimelineTree.utils';
import { AssessmentBoard } from './AssessmentBoard';
import { AssessmentCard } from './AssessmentCard';
import { Carousel } from './Carousel';
import { ContentViewer } from './ContentViewer';
import { DataTable } from './DataTable';
import { FeedbackForm } from './FeedbackForm';
import { NodeSelectionProvider } from './NodeSelectionContext';
import { TreeCheckProvider } from './TreeCheckContext';
import { StatCard } from './StatCard';
import { TimelineChart } from './TimelineChart';
import { TreeView } from './TreeView';
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

// A node for the generic TreeView. `attributes` is the opaque, filterable
// metadata bag the TreeView's structured filter matches against.
type TreeNode = {
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
    startOffsetMs: number;
    endOffsetMs: number;
  };
  children: TreeNode[];
};

// Everything a message set might need to render. Trace-level metrics come from
// `modelTraceInfo`; per-tool rows, the timeline, and the tree are derived from
// the parsed spans (nodeMap / topLevelNodes).
// A single key/value entry for the ContentViewer. `value` is JSON-encoded so the
// snippet renderer can show it as JSON (objects) or text/markdown (strings).
type ContentField = { label: string; value: string };

type AssessmentSentiment = 'positive' | 'negative' | 'neutral' | 'error';

type AssessmentBoardItem = {
  name: string;
  value?: string;
  rationale?: string;
  source?: string;
  sentiment: AssessmentSentiment;
};

type CustomViewData = {
  metrics: TraceMetrics;
  toolRows: TableRow[];
  timelineRows: TimelineRow[];
  treeNodes: TreeNode[];
  assessmentItems: AssessmentBoardItem[];
};

// Turns an arbitrary inputs/outputs payload into ContentViewer fields. Objects
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
const getTreeNodesFromNodes = (topLevelNodes: ModelTraceSpanNode[], traceStartUs: number): TreeNode[] => {
  const toTreeNode = (node: ModelTraceSpanNode): TreeNode => {
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
        // Offsets from the trace start (ms), used by the TreeView's live time
        // window filter and per-span tick marks.
        startOffsetMs: Math.max(node.start - traceStartUs, 0) / 1000,
        endOffsetMs: Math.max(node.end - traceStartUs, 0) / 1000,
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

// A span in scope for feedback: the info the FeedbackForm needs to create a
// span-scoped assessment, plus its inputs/outputs for the optional I/O toggle.
type FeedbackSpan = {
  spanId: string;
  spanName: string;
  traceId: string;
  inputs: ContentField[];
  outputs: ContentField[];
};

// Builds a feedback carousel: one FeedbackForm slide per scoped span, stepped
// through by a generic Carousel. Built on demand (not a dropdown message set)
// since it depends on the scoped spans captured from a specific tree block.
const buildFeedbackCarouselMessages = (surfaceId: string, spans: FeedbackSpan[]): A2uiMessage[] => {
  const childIds = spans.map((_, index) => `feedback-slide-${index}`);
  return [
    createSurfaceMessage(surfaceId),
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId,
        components: [
          {
            id: 'root',
            component: 'Carousel',
            children: childIds,
            emptyMessage: 'No spans in scope for feedback.',
          },
          ...spans.map((span, index) => ({
            id: childIds[index],
            component: 'FeedbackForm',
            traceId: span.traceId,
            spanId: span.spanId,
            spanName: span.spanName,
            inputs: span.inputs,
            outputs: span.outputs,
          })),
        ],
      },
    },
  ];
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

// Collapsible span tree (like Details & Timeline). Nodes are inlined; the
// TreeView's own timeline slider scopes the visible spans.
const buildTraceTreeMessages = (surfaceId: string, { treeNodes }: CustomViewData): A2uiMessage[] => [
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
          nodes: treeNodes,
          emptyMessage: 'No spans to display.',
        },
      ],
    },
  },
];

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

// A ContentViewer surface for one specific span's inputs & outputs, rendered
// beside the tree when "open span details on select" is on. Built on demand
// (depends on the clicked span), like the feedback carousel.
const buildSpanDetailsMessages = (
  surfaceId: string,
  spanName: string,
  inputs: ContentField[],
  outputs: ContentField[],
): A2uiMessage[] => [
  createSurfaceMessage(surfaceId),
  {
    version: 'v0.9',
    updateComponents: {
      surfaceId,
      components: [
        { id: 'root', component: 'Column', children: ['inputs', 'outputs'] },
        {
          id: 'inputs',
          component: 'ContentViewer',
          title: 'Inputs',
          icon: 'list',
          fields: inputs,
          emptyMessage: `No inputs on ${spanName}.`,
        },
        {
          id: 'outputs',
          component: 'ContentViewer',
          title: 'Outputs',
          icon: 'checklist',
          fields: outputs,
          emptyMessage: `No outputs on ${spanName}.`,
        },
      ],
    },
  },
];

const MESSAGE_SETS: MessageSet[] = [
  { id: 'trace-summary', label: 'Show me the high level summary of this trace', build: buildTraceSummaryMessages },
  { id: 'tool-performance', label: 'List performance summary for all tools', build: buildToolPerformanceMessages },
  { id: 'trace-breakdown', label: 'Give me a timeline of all spans calls', build: buildTraceBreakdownMessages },
  { id: 'trace-tree', label: 'Show me the span calls in a tree view', build: buildTraceTreeMessages },
  { id: 'assessments', label: 'Show the LLM-as-a-judge assessments', build: buildAssessmentsMessages },
];

// An appended dashboard block, each backed by its own A2UI surface. `setId`
// identifies which message set produced it. `feedbackSurfaceId` is the optional
// companion surface rendered side by side inside the same block (the feedback
// review carousel).
type DashboardBlock = {
  surfaceId: string;
  label: string;
  setId: string;
  feedbackSurfaceId?: string;
  feedbackSpanCount?: number;
  // The span ids checked (via per-row checkboxes) in this Trace tree; "Add
  // feedback" is scoped to exactly these spans.
  checkedSpanIds?: string[];
  // Per-block toggle: when on, clicking a span in this Trace tree opens its
  // inputs/outputs in a companion ContentViewer surface, rendered side by side.
  openDetailsOnSelect?: boolean;
  detailsSurfaceId?: string;
  detailsSpanId?: string;
  detailsSpanName?: string;
};

export const ModelTraceExplorerCustomView = ({ modelTraceInfo }: { modelTraceInfo: ModelTrace['info'] }) => {
  const { theme } = useDesignSystemTheme();

  // Span data comes from the shared view-state context (the same source the
  // Summary tab uses), not from props. `nodeMap` holds every parsed span;
  // `topLevelNodes` preserves the hierarchy/order for the timeline.
  const { nodeMap, topLevelNodes } = useModelTraceExplorerViewState();

  // The catalog is the React equivalent of `catalog.json`: it maps component
  // type names to their implementations (basic Text/Row + our custom
  // StatCard/DataTable/TimelineChart/TreeView/Carousel/FeedbackForm).
  const catalog = useMemo(
    () =>
      new Catalog<ReactComponentImplementation>(
        CUSTOM_VIEW_CATALOG_ID,
        [
          Text,
          Row,
          Column,
          StatCard,
          DataTable,
          TimelineChart,
          TreeView,
          Carousel,
          FeedbackForm,
          ContentViewer,
          AssessmentBoard,
          AssessmentCard,
        ],
        BASIC_FUNCTIONS,
      ),
    [],
  );

  // A single long-lived processor holds the state for every appended surface.
  const [processor] = useState(() => new MessageProcessor<ReactComponentImplementation>([catalog]));

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
  const treeNodes = useMemo(() => {
    const traceStartUs = topLevelNodes.length > 0 ? Math.min(...topLevelNodes.map((node) => node.start)) : 0;
    return getTreeNodesFromNodes(treeRoots, traceStartUs);
  }, [treeRoots, topLevelNodes]);

  // Real assessments (LLM-judge / human feedback), used by both the predefined
  // "LLM-as-a-judge" board and Agent Mode (so the model shows real results).
  const agentAssessments = useMemo(() => getAgentAssessments(modelTraceInfo, nodeMap), [modelTraceInfo, nodeMap]);
  const assessmentItems = useMemo(() => getAssessmentBoardItems(agentAssessments), [agentAssessments]);

  const viewData = useMemo<CustomViewData>(
    () => ({ metrics, toolRows, timelineRows, treeNodes, assessmentItems }),
    [metrics, toolRows, timelineRows, treeNodes, assessmentItems],
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

  // Toggles a span's checkbox within a Trace tree block. The checked set scopes
  // "Add feedback".
  const handleToggleCheck = (sourceBlock: DashboardBlock, spanId: string, checked: boolean) => {
    setBlocks((prev) =>
      prev.map((block) => {
        if (block.surfaceId !== sourceBlock.surfaceId) {
          return block;
        }
        const current = new Set(block.checkedSpanIds ?? []);
        if (checked) {
          current.add(spanId);
        } else {
          current.delete(spanId);
        }
        return { ...block, checkedSpanIds: Array.from(current) };
      }),
    );
  };

  // Opens a feedback review carousel scoped to the spans checked in the source
  // tree block, rendered side by side inside that same block. Reuses the real
  // assessment-creation flow via FeedbackForm. Re-clicking refreshes the scoped
  // set (the existing companion surface is replaced).
  const handleAddFeedback = (sourceBlock: DashboardBlock) => {
    const spans: FeedbackSpan[] = (sourceBlock.checkedSpanIds ?? [])
      .map((spanId) => nodeMap[spanId])
      .filter((node): node is ModelTraceSpanNode => Boolean(node))
      .map((node) => ({
        spanId: String(node.key),
        spanName: typeof node.title === 'string' ? node.title : String(node.title ?? 'unknown'),
        traceId: node.traceId,
        inputs: getContentFields(node.inputs),
        outputs: getContentFields(node.outputs),
      }));
    blockCounter.current += 1;
    const feedbackSurfaceId = `custom-view-feedback-${blockCounter.current}`;

    const messages: A2uiMessage[] = [];
    if (sourceBlock.feedbackSurfaceId) {
      messages.push({ version: 'v0.9', deleteSurface: { surfaceId: sourceBlock.feedbackSurfaceId } });
    }
    messages.push(...buildFeedbackCarouselMessages(feedbackSurfaceId, spans));
    processor.processMessages(messages);

    setBlocks((prev) =>
      prev.map((block) =>
        block.surfaceId === sourceBlock.surfaceId
          ? { ...block, feedbackSurfaceId, feedbackSpanCount: spans.length }
          : block,
      ),
    );
  };

  // Closes the companion feedback surface without removing the tree block.
  const handleCloseFeedback = (block: DashboardBlock) => {
    if (!block.feedbackSurfaceId) {
      return;
    }
    processor.processMessages([{ version: 'v0.9', deleteSurface: { surfaceId: block.feedbackSurfaceId } }]);
    setBlocks((prev) =>
      prev.map((entry) =>
        entry.surfaceId === block.surfaceId
          ? { ...entry, feedbackSurfaceId: undefined, feedbackSpanCount: undefined }
          : entry,
      ),
    );
  };

  // Opens (or refreshes) the selected span's inputs/outputs in a ContentViewer
  // beside the source tree block. Driven by the TreeView's node-selection
  // context when the "open span details on select" toggle is on.
  const handleSelectSpan = (sourceBlock: DashboardBlock, spanId: string) => {
    const span = nodeMap[spanId];
    if (!span) {
      return;
    }
    const spanName = span.title ? String(span.title) : spanId;
    blockCounter.current += 1;
    const detailsSurfaceId = `custom-view-details-${blockCounter.current}`;

    const messages: A2uiMessage[] = [];
    if (sourceBlock.detailsSurfaceId) {
      messages.push({ version: 'v0.9', deleteSurface: { surfaceId: sourceBlock.detailsSurfaceId } });
    }
    messages.push(
      ...buildSpanDetailsMessages(detailsSurfaceId, spanName, getContentFields(span.inputs), getContentFields(span.outputs)),
    );
    processor.processMessages(messages);

    setBlocks((prev) =>
      prev.map((block) =>
        block.surfaceId === sourceBlock.surfaceId
          ? { ...block, detailsSurfaceId, detailsSpanId: spanId, detailsSpanName: spanName }
          : block,
      ),
    );
  };

  const handleCloseDetails = (block: DashboardBlock) => {
    if (!block.detailsSurfaceId) {
      return;
    }
    processor.processMessages([{ version: 'v0.9', deleteSurface: { surfaceId: block.detailsSurfaceId } }]);
    setBlocks((prev) =>
      prev.map((entry) =>
        entry.surfaceId === block.surfaceId
          ? { ...entry, detailsSurfaceId: undefined, detailsSpanId: undefined, detailsSpanName: undefined }
          : entry,
      ),
    );
  };

  // Per-block toggle. Turning it off for a block tears down that block's open
  // details surface.
  const handleToggleOpenDetails = (sourceBlock: DashboardBlock, next: boolean) => {
    if (!next && sourceBlock.detailsSurfaceId) {
      processor.processMessages([{ version: 'v0.9', deleteSurface: { surfaceId: sourceBlock.detailsSurfaceId } }]);
    }
    setBlocks((prev) =>
      prev.map((block) =>
        block.surfaceId === sourceBlock.surfaceId
          ? {
              ...block,
              openDetailsOnSelect: next,
              ...(next
                ? {}
                : { detailsSurfaceId: undefined, detailsSpanId: undefined, detailsSpanName: undefined }),
            }
          : block,
      ),
    );
  };

  const handleRemoveBlock = (block: DashboardBlock) => {
    // Remove the block (and any companion feedback/details surface) via the
    // A2UI renderer's deleteSurface message.
    const surfaceIds = [block.surfaceId, block.feedbackSurfaceId, block.detailsSurfaceId].filter(
      (id): id is string => Boolean(id),
    );
    processor.processMessages(surfaceIds.map((surfaceId) => ({ version: 'v0.9', deleteSurface: { surfaceId } })));
    setBlocks((prev) => prev.filter((entry) => entry.surfaceId !== block.surfaceId));
  };

  const handleClearAll = () => {
    const surfaceIds = blocks.flatMap((block) =>
      [block.surfaceId, block.feedbackSurfaceId, block.detailsSurfaceId].filter((id): id is string => Boolean(id)),
    );
    processor.processMessages(surfaceIds.map((surfaceId) => ({ version: 'v0.9', deleteSurface: { surfaceId } })));
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
            const feedbackSurface = block.feedbackSurfaceId
              ? processor.model.getSurface(block.feedbackSurfaceId)
              : undefined;
            const detailsSurface = block.detailsSurfaceId
              ? processor.model.getSurface(block.detailsSurfaceId)
              : undefined;
            const isTraceTree = block.setId === 'trace-tree';
            const selectionEnabled = Boolean(block.openDetailsOnSelect) && isTraceTree;
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
                    {isTraceTree && (
                      <Switch
                        componentId="shared.model-trace-explorer.custom-view.open-details-toggle"
                        label="Open span details on select"
                        checked={Boolean(block.openDetailsOnSelect)}
                        onChange={(next) => handleToggleOpenDetails(block, next)}
                      />
                    )}
                    {isTraceTree && (
                      <Button
                        componentId="shared.model-trace-explorer.custom-view.add-feedback"
                        size="small"
                        icon={<PlusIcon />}
                        disabled={(block.checkedSpanIds?.length ?? 0) === 0}
                        onClick={() => handleAddFeedback(block)}
                      >
                        {block.feedbackSurfaceId ? 'Refresh feedback' : 'Add feedback'}
                        {(block.checkedSpanIds?.length ?? 0) > 0 ? ` (${block.checkedSpanIds?.length})` : ''}
                      </Button>
                    )}
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
                <div
                  css={{
                    display: 'flex',
                    alignItems: 'stretch',
                    gap: theme.spacing.md,
                    padding: theme.spacing.md,
                  }}
                >
                  <div css={{ flex: 1, minWidth: 0 }}>
                    {surface && (
                      <TreeCheckProvider
                        value={{
                          enabled: isTraceTree,
                          checkedIds: new Set(block.checkedSpanIds ?? []),
                          onToggle: (spanId, checked) => handleToggleCheck(block, spanId, checked),
                        }}
                      >
                        <NodeSelectionProvider
                          value={{
                            enabled: selectionEnabled,
                            selectedId: block.detailsSpanId,
                            onSelect: (spanId) => handleSelectSpan(block, spanId),
                          }}
                        >
                          <A2uiSurface surface={surface} />
                        </NodeSelectionProvider>
                      </TreeCheckProvider>
                    )}
                  </div>
                  {detailsSurface && (
                    <div
                      css={{
                        flex: 1,
                        minWidth: 0,
                        display: 'flex',
                        flexDirection: 'column',
                        gap: theme.spacing.sm,
                        borderLeft: `1px solid ${theme.colors.border}`,
                        paddingLeft: theme.spacing.md,
                      }}
                    >
                      <div
                        css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: theme.spacing.sm }}
                      >
                        <Typography.Text bold css={{ fontFamily: 'monospace' }}>
                          {block.detailsSpanName ?? 'Span details'}
                        </Typography.Text>
                        <Button
                          componentId="shared.model-trace-explorer.custom-view.close-details"
                          size="small"
                          icon={<CloseIcon />}
                          aria-label="Close span details"
                          onClick={() => handleCloseDetails(block)}
                        />
                      </div>
                      <A2uiSurface surface={detailsSurface} />
                    </div>
                  )}
                  {feedbackSurface && (
                    <div
                      css={{
                        flex: 1,
                        minWidth: 0,
                        display: 'flex',
                        flexDirection: 'column',
                        gap: theme.spacing.sm,
                        borderLeft: `1px solid ${theme.colors.border}`,
                        paddingLeft: theme.spacing.md,
                      }}
                    >
                      <div
                        css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: theme.spacing.sm }}
                      >
                        <Typography.Text bold>
                          Feedback{typeof block.feedbackSpanCount === 'number' ? ` (${block.feedbackSpanCount} spans)` : ''}
                        </Typography.Text>
                        <Button
                          componentId="shared.model-trace-explorer.custom-view.close-feedback"
                          size="small"
                          icon={<CloseIcon />}
                          aria-label="Close feedback"
                          onClick={() => handleCloseFeedback(block)}
                        />
                      </div>
                      <A2uiSurface surface={feedbackSurface} />
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
