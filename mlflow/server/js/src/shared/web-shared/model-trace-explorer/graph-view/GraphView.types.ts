import type { Node, Edge } from '@xyflow/react';

import type { ModelTraceSpanNode } from '../ModelTrace.types';

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
  onSelect: () => void;
  onViewSpanDetails: (span: ModelTraceSpanNode) => void;
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
}

/**
 * React Flow edge type for workflow view
 */
export type WorkflowFlowEdge = Edge<WorkflowEdgeData>;

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
