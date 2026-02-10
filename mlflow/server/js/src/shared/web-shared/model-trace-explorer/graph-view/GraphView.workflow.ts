import type { ModelTraceSpanNode } from '../ModelTrace.types';
import type { WorkflowNode, WorkflowEdge, WorkflowLayout, GraphLayoutConfig } from './GraphView.types';
import { DEFAULT_WORKFLOW_LAYOUT_CONFIG } from './GraphView.types';

/**
 * Flattens the span tree into a list of all spans.
 */
function flattenSpans(node: ModelTraceSpanNode): ModelTraceSpanNode[] {
  const result: ModelTraceSpanNode[] = [node];

  if (node.children) {
    for (const child of node.children) {
      result.push(...flattenSpans(child));
    }
  }

  return result;
}

/**
 * Gets the aggregation key for a span using its name.
 */
function getAggregationKey(span: ModelTraceSpanNode): string {
  return String(span.title ?? 'Unknown');
}

/**
 * Custom layered layout algorithm for workflow graphs.
 * Assigns nodes to layers based on dependencies, then positions them.
 * Handles cycles by detecting back-edges and treating them specially.
 */
function applyLayeredLayout(
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  config: GraphLayoutConfig,
): { width: number; height: number } {
  if (nodes.length === 0) {
    return { width: 0, height: 0 };
  }

  // Build adjacency maps
  const outgoing = new Map<string, string[]>(); // node -> targets
  const incoming = new Map<string, string[]>(); // node -> sources

  for (const node of nodes) {
    outgoing.set(node.id, []);
    incoming.set(node.id, []);
  }

  for (const edge of edges) {
    if (!edge.isBackEdge) {
      outgoing.get(edge.sourceId)?.push(edge.targetId);
      incoming.get(edge.targetId)?.push(edge.sourceId);
    }
  }

  // Assign layers using topological sort with BFS
  // Nodes with no incoming edges (excluding back-edges) go to layer 0
  const layers = new Map<string, number>();
  const queue: string[] = [];

  // Find root nodes (no incoming edges)
  for (const node of nodes) {
    const incomingEdges = incoming.get(node.id) ?? [];
    if (incomingEdges.length === 0) {
      layers.set(node.id, 0);
      queue.push(node.id);
    }
  }

  // If no root nodes found (all nodes in cycle), start from first node
  if (queue.length === 0 && nodes.length > 0) {
    layers.set(nodes[0].id, 0);
    queue.push(nodes[0].id);
  }

  // BFS to assign layers
  while (queue.length > 0) {
    const nodeId = queue.shift()!;
    const currentLayer = layers.get(nodeId) ?? 0;
    const targets = outgoing.get(nodeId) ?? [];

    for (const targetId of targets) {
      const existingLayer = layers.get(targetId);
      const newLayer = currentLayer + 1;

      if (existingLayer === undefined || newLayer > existingLayer) {
        layers.set(targetId, newLayer);
        // Only add to queue if not already processed at this or higher layer
        if (!queue.includes(targetId)) {
          queue.push(targetId);
        }
      }
    }
  }

  // Handle any remaining unassigned nodes (disconnected or in pure cycles)
  let maxLayer = 0;
  for (const layer of layers.values()) {
    maxLayer = Math.max(maxLayer, layer);
  }
  for (const node of nodes) {
    if (!layers.has(node.id)) {
      maxLayer++;
      layers.set(node.id, maxLayer);
    }
  }

  // Group nodes by layer
  const layerGroups = new Map<number, WorkflowNode[]>();
  for (const node of nodes) {
    const layer = layers.get(node.id) ?? 0;
    if (!layerGroups.has(layer)) {
      layerGroups.set(layer, []);
    }
    layerGroups.get(layer)!.push(node);
  }

  // Position nodes in each layer
  const sortedLayers = Array.from(layerGroups.keys()).sort((a, b) => a - b);
  let totalWidth = 0;

  for (const layerIndex of sortedLayers) {
    const layerNodes = layerGroups.get(layerIndex)!;
    const layerWidth =
      layerNodes.reduce((sum, n) => sum + n.width, 0) + (layerNodes.length - 1) * config.horizontalSpacing;
    totalWidth = Math.max(totalWidth, layerWidth);
  }

  // Center nodes within each layer
  for (const layerIndex of sortedLayers) {
    const layerNodes = layerGroups.get(layerIndex)!;
    const layerWidth =
      layerNodes.reduce((sum, n) => sum + n.width, 0) + (layerNodes.length - 1) * config.horizontalSpacing;
    let x = config.padding + (totalWidth - layerWidth) / 2;
    const y = config.padding + layerIndex * (config.nodeHeight + config.verticalSpacing);

    for (const node of layerNodes) {
      node.x = x;
      node.y = y;
      x += node.width + config.horizontalSpacing;
    }
  }

  // Mark back-edges (edges that go from lower layer to higher layer numbers,
  // i.e., from a node "below" to a node "above" in the visual hierarchy)
  for (const edge of edges) {
    const sourceLayer = layers.get(edge.sourceId) ?? 0;
    const targetLayer = layers.get(edge.targetId) ?? 0;
    if (sourceLayer >= targetLayer && !edge.isNestedCall) {
      edge.isBackEdge = true;
    }
  }

  return {
    width: totalWidth + config.padding * 2,
    height:
      sortedLayers.length * (config.nodeHeight + config.verticalSpacing) - config.verticalSpacing + config.padding * 2,
  };
}

/**
 * Internal function to build workflow nodes and edges from span data.
 * Groups spans by name and uses the span's type for visual styling.
 */
function buildWorkflowGraph(
  rootNode: ModelTraceSpanNode,
  config: GraphLayoutConfig,
): { nodes: WorkflowNode[]; edges: WorkflowEdge[] } | null {
  // Flatten all spans
  const spans = flattenSpans(rootNode);

  if (spans.length === 0) {
    return null;
  }

  // Sort by start time to get execution order
  spans.sort((a, b) => a.start - b.start);

  // Group spans by name (aggregation key)
  const nodeGroups = new Map<string, ModelTraceSpanNode[]>();
  for (const span of spans) {
    const key = getAggregationKey(span);
    if (!nodeGroups.has(key)) {
      nodeGroups.set(key, []);
    }
    nodeGroups.get(key)!.push(span);
  }

  // Create WorkflowNodes using span name and type
  const workflowNodes: WorkflowNode[] = [];
  for (const [key, groupSpans] of nodeGroups) {
    const firstSpan = groupSpans[0];
    const nodeType = firstSpan.type ?? 'UNKNOWN';
    const displayName = String(firstSpan.title ?? 'Unknown');

    workflowNodes.push({
      id: key,
      displayName,
      nodeType,
      count: groupSpans.length,
      spans: groupSpans,
      x: 0,
      y: 0,
      width: config.nodeWidth,
      height: config.nodeHeight,
    });
  }

  // Build edges from parent-child relationships
  const edgeMap = new Map<string, WorkflowEdge>();

  function buildEdgesFromHierarchy(node: ModelTraceSpanNode, parentKey: string | null): void {
    const currentKey = getAggregationKey(node);

    if (parentKey !== null && currentKey !== parentKey) {
      const edgeId = `${parentKey}->${currentKey}`;
      if (!edgeMap.has(edgeId)) {
        edgeMap.set(edgeId, {
          sourceId: parentKey,
          targetId: currentKey,
          count: 0,
          isBackEdge: false,
          isNestedCall: false,
        });
      }
      edgeMap.get(edgeId)!.count++;
    }

    if (node.children) {
      for (const child of node.children) {
        buildEdgesFromHierarchy(child, currentKey);
      }
    }
  }

  buildEdgesFromHierarchy(rootNode, null);

  // Detect nested call edges (bidirectional relationships)
  const toolLikeTypes = new Set(['TOOL', 'GUARDRAIL', 'FUNCTION', 'RETRIEVER']);
  for (const edge of edgeMap.values()) {
    const reverseEdgeId = `${edge.targetId}->${edge.sourceId}`;
    if (edgeMap.has(reverseEdgeId)) {
      const reverseEdge = edgeMap.get(reverseEdgeId)!;
      // Check if source node is a tool-like type
      const sourceNode = workflowNodes.find((n) => n.id === edge.sourceId);
      const reverseSourceNode = workflowNodes.find((n) => n.id === reverseEdge.sourceId);
      if (sourceNode && toolLikeTypes.has(sourceNode.nodeType.toUpperCase())) {
        edge.isNestedCall = true;
      } else if (reverseSourceNode && toolLikeTypes.has(reverseSourceNode.nodeType.toUpperCase())) {
        reverseEdge.isNestedCall = true;
      }
    }
  }

  return {
    nodes: workflowNodes,
    edges: Array.from(edgeMap.values()),
  };
}

/**
 * Computes an aggregated workflow graph from span data.
 * Groups spans by name and uses span type for visual styling.
 *
 * Algorithm:
 * 1. Flatten all spans in the tree
 * 2. Sort by start time to get execution order
 * 3. Group spans by name
 * 4. Build edges based on parent-child hierarchy
 * 5. Apply custom layered layout algorithm
 *
 * @param rootNode - The root span node of the trace
 * @param config - Layout configuration
 * @returns WorkflowLayout with positioned nodes and edges
 */
export function computeWorkflowLayout(
  rootNode: ModelTraceSpanNode | null,
  config: GraphLayoutConfig = DEFAULT_WORKFLOW_LAYOUT_CONFIG,
): WorkflowLayout {
  if (!rootNode) {
    return { nodes: [], edges: [], width: 0, height: 0 };
  }

  const graph = buildWorkflowGraph(rootNode, config);
  if (!graph) {
    return { nodes: [], edges: [], width: 0, height: 0 };
  }

  // Apply custom layered layout
  const dimensions = applyLayeredLayout(graph.nodes, graph.edges, config);

  return {
    nodes: graph.nodes,
    edges: graph.edges,
    width: dimensions.width,
    height: dimensions.height,
  };
}
