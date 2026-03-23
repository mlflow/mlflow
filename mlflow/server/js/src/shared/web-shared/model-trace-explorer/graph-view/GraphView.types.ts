import type { Node, Edge } from '@xyflow/react';

import type { ModelTraceSpanNode } from '../ModelTrace.types';

/**
 * Framework-agnostic graph schema describing node-and-edge topology.
 * Populated from the `mlflow.trace.graphSchema` trace tag (see TraceTagKey.GRAPH_SCHEMA).
 * Currently extracted by LangGraph autologging via CompiledGraph.get_graph().to_json();
 * other frameworks can populate the same tag to enable the graph view.
 */
export interface GraphSchema {
  nodes: Array<{
    id: string;
    type?: string;
    data?: { id?: string[]; name?: string };
  }>;
  edges: Array<{
    source: string;
    target: string;
    data?: string;
    conditional?: boolean;
  }>;
}

/**
 * Data payload for workflow nodes
 */
export interface WorkflowNodeData extends Record<string, unknown> {
  displayName: string;
  nodeType: string;
  count: number;
  spans: ModelTraceSpanNode[];
  isSelected: boolean;
  isOnHighlightedPath: boolean;
  onViewSpanDetails: (span: ModelTraceSpanNode) => void;
  nodeWidth: number;
  nodeHeight: number;
  isStructural?: boolean;
  isExecuted?: boolean;
  executionOrder?: number[];
  orientation?: GraphOrientation;
}

/**
 * React Flow node type for workflow view
 */
export type WorkflowFlowNode = Node<WorkflowNodeData>;

/**
 * Data payload for workflow edges
 */
export interface WorkflowEdgeData extends Record<string, unknown> {
  count: number;
  isBackEdge: boolean;
  isNestedCall: boolean;
  isHighlighted: boolean;
  isConditional?: boolean;
  isExecuted?: boolean;
  conditionLabel?: string;
  stepSequence?: number[];
  isReturnEdge?: boolean;
  orientation?: GraphOrientation;
}

/**
 * React Flow edge type for workflow view
 */
export type WorkflowFlowEdge = Edge<WorkflowEdgeData>;

/**
 * Flow direction for the graph layout.
 * 'TB' = top-to-bottom (vertical), 'LR' = left-to-right (horizontal).
 */
export type GraphOrientation = 'TB' | 'LR';

/**
 * Layout configuration options
 */
export interface GraphLayoutConfig {
  nodeWidth: number;
  nodeHeight: number;
  horizontalSpacing: number;
  verticalSpacing: number;
  padding: number;
  orientation: GraphOrientation;
}

/**
 * Aggregated node representing all spans of a given name.
 */
export interface WorkflowNode {
  id: string;
  displayName: string;
  nodeType: string;
  count: number;
  spans: ModelTraceSpanNode[];
  x: number;
  y: number;
  width: number;
  height: number;
  isStructural?: boolean;
  isExecuted?: boolean;
  /** Step numbers showing when this node was visited during execution (1-indexed). */
  executionOrder?: number[];
}

/**
 * Edge representing a transition between workflow nodes.
 * Created based on parent-child hierarchy in the span tree.
 */
export interface WorkflowEdge {
  sourceId: string;
  targetId: string;
  count: number;
  isBackEdge: boolean;
  isNestedCall?: boolean;
  condition?: string;
  isConditional?: boolean;
  isExecuted?: boolean;
  /** Step numbers showing when this edge was traversed during execution (1-indexed). */
  stepSequence?: number[];
  /** True for loop-back return edges (e.g. tool→agent). Rendered subtly. */
  isReturnEdge?: boolean;
}

/**
 * Complete workflow graph layout with aggregated nodes and transition edges
 */
export interface WorkflowLayout {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  width: number;
  height: number;
}

/**
 * Default workflow layout configuration
 */
export const DEFAULT_WORKFLOW_LAYOUT_CONFIG: GraphLayoutConfig = {
  nodeWidth: 160,
  nodeHeight: 56,
  horizontalSpacing: 60,
  verticalSpacing: 80,
  padding: 40,
  orientation: 'TB',
};

/**
 * Expanded workflow layout configuration — used when the graph is in full-expand mode.
 * Larger nodes and generous spacing fill the expanded canvas area.
 */
export const EXPANDED_WORKFLOW_LAYOUT_CONFIG: GraphLayoutConfig = {
  nodeWidth: 220,
  nodeHeight: 72,
  horizontalSpacing: 80,
  verticalSpacing: 100,
  padding: 50,
  orientation: 'TB',
};
