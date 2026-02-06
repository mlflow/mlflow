import type { Node, Edge } from '@xyflow/react';

import type { ModelTraceSpanNode } from '../ModelTrace.types';

/**
 * View modes for the graph visualization
 */
export type GraphViewMode = 'all_spans' | 'logical_workflow';

/**
 * Data payload for span tree nodes (All Spans view)
 */
export interface SpanNodeData extends Record<string, unknown> {
  spanNode: ModelTraceSpanNode;
  label: string;
  isSelected: boolean;
  isOnHighlightedPath: boolean;
  onSelect: () => void;
  onViewDetails: () => void;
}

/**
 * React Flow node type for span tree view
 */
export type SpanFlowNode = Node<SpanNodeData>;

/**
 * Data payload for workflow nodes (Workflow view)
 */
export interface WorkflowNodeData extends Record<string, unknown> {
  displayName: string;
  nodeType: string;
  count: number;
  spans: ModelTraceSpanNode[];
  isSelected: boolean;
  isOnHighlightedPath: boolean;
  onSelect: () => void;
  onViewSpanDetails: (span: ModelTraceSpanNode) => void;
}

/**
 * React Flow node type for workflow view
 */
export type WorkflowFlowNode = Node<WorkflowNodeData>;

/**
 * Data payload for span tree edges
 */
export interface SpanEdgeData extends Record<string, unknown> {
  isHighlighted: boolean;
}

/**
 * React Flow edge type for span tree view
 */
export type SpanFlowEdge = Edge<SpanEdgeData>;

/**
 * Data payload for workflow edges
 */
export interface WorkflowEdgeData extends Record<string, unknown> {
  count: number;
  isBackEdge: boolean;
  isNestedCall: boolean;
  isHighlighted: boolean;
}

/**
 * React Flow edge type for workflow view
 */
export type WorkflowFlowEdge = Edge<WorkflowEdgeData>;

/**
 * Represents a node in the graph layout
 */
export interface GraphNode {
  id: string;
  spanNode: ModelTraceSpanNode;
  layer: number;
  orderIndex: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Represents an edge connecting two nodes
 */
export interface GraphEdge {
  sourceId: string;
  targetId: string;
  sourceNode: GraphNode;
  targetNode: GraphNode;
}

/**
 * Complete graph layout with nodes, edges, and dimensions
 */
export interface GraphLayout {
  nodes: GraphNode[];
  edges: GraphEdge[];
  width: number;
  height: number;
}

/**
 * Layout configuration options
 */
export interface GraphLayoutConfig {
  nodeWidth: number;
  nodeHeight: number;
  horizontalSpacing: number;
  verticalSpacing: number;
  padding: number;
}

/**
 * Default layout configuration
 */
export const DEFAULT_GRAPH_LAYOUT_CONFIG: GraphLayoutConfig = {
  nodeWidth: 140,
  nodeHeight: 48,
  horizontalSpacing: 40,
  verticalSpacing: 60,
  padding: 40,
};

/**
 * Aggregated node representing all spans of a given name.
 * Unlike GraphNode which is 1:1 with spans, WorkflowNode aggregates multiple spans.
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
};
