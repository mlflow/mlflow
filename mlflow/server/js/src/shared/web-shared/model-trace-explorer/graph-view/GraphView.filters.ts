import type { ModelTraceSpanNode } from '../ModelTrace.types';

/**
 * Graph node attribute keys used for logical workflow visualization.
 * These match the Python constants in mlflow/tracing/constant.py
 */
export const GRAPH_NODE_ATTRIBUTE_KEYS = {
  GRAPH_NODE_ID: 'mlflow.graph.node.id',
  GRAPH_NODE_PARENT_ID: 'mlflow.graph.node.parentId',
  GRAPH_NODE_TYPE: 'mlflow.graph.node.type',
  GRAPH_NODE_DISPLAY_NAME: 'mlflow.graph.node.displayName',
  GRAPH_EDGE_CONDITION: 'mlflow.graph.edge.condition',
} as const;

/**
 * Safely get an attribute value from a span.
 * Handles both Record<string, any> (v2/v3) and array format (v4).
 */
export function getSpanAttribute(span: ModelTraceSpanNode, key: string): unknown {
  const attrs = span.attributes;
  if (!attrs) {
    return undefined;
  }
  // Handle array format (v4)
  if (Array.isArray(attrs)) {
    const entry = attrs.find((a) => a.key === key);
    if (!entry) return undefined;
    const val = entry.value;
    return val.string_value ?? val.int_value ?? val.bool_value;
  }
  // Handle object format (v2/v3)
  return attrs[key];
}

/**
 * Check if a span has graph node attributes set
 */
export function hasGraphNodeAttributes(span: ModelTraceSpanNode): boolean {
  return getSpanAttribute(span, GRAPH_NODE_ATTRIBUTE_KEYS.GRAPH_NODE_TYPE) !== undefined;
}

/**
 * Get the graph node type from span attributes
 */
export function getGraphNodeType(span: ModelTraceSpanNode): string | undefined {
  return getSpanAttribute(span, GRAPH_NODE_ATTRIBUTE_KEYS.GRAPH_NODE_TYPE) as string | undefined;
}

/**
 * Get the graph node display name from span attributes
 */
export function getGraphNodeDisplayName(span: ModelTraceSpanNode): string | undefined {
  return getSpanAttribute(span, GRAPH_NODE_ATTRIBUTE_KEYS.GRAPH_NODE_DISPLAY_NAME) as string | undefined;
}

/**
 * Check if any span in the tree has graph node attributes
 */
export function hasAnyGraphNodeAttributes(rootNode: ModelTraceSpanNode): boolean {
  if (hasGraphNodeAttributes(rootNode)) {
    return true;
  }

  if (rootNode.children) {
    return rootNode.children.some((child) => hasAnyGraphNodeAttributes(child));
  }

  return false;
}

/**
 * Filter the span tree to only include spans with graph node attributes.
 * Preserves the tree structure by keeping parent spans if any descendant has graph attributes.
 *
 * @param node The root node to filter
 * @returns A new tree containing only logical workflow nodes, or null if no nodes match
 */
export function filterForLogicalWorkflow(node: ModelTraceSpanNode): ModelTraceSpanNode | null {
  const hasGraphAttrs = hasGraphNodeAttributes(node);

  // If no children, only include this node if it has graph attributes
  if (!node.children || node.children.length === 0) {
    return hasGraphAttrs ? { ...node, children: [] } : null;
  }

  // Recursively filter children
  const filteredChildren = node.children
    .map((child) => filterForLogicalWorkflow(child))
    .filter((child): child is ModelTraceSpanNode => child !== null);

  // Include this node if:
  // 1. It has graph attributes, OR
  // 2. Any of its descendants have graph attributes (to preserve structure)
  if (hasGraphAttrs || filteredChildren.length > 0) {
    return {
      ...node,
      children: filteredChildren,
    };
  }

  return null;
}

/**
 * Count the number of spans with graph node attributes in the tree
 */
export function countGraphNodes(node: ModelTraceSpanNode): number {
  let count = hasGraphNodeAttributes(node) ? 1 : 0;

  if (node.children) {
    count += node.children.reduce((sum, child) => sum + countGraphNodes(child), 0);
  }

  return count;
}
