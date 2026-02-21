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
 * Gets the aggregation key for a span using its type and name.
 * Including type prevents unrelated operations with the same name from merging.
 */
function getAggregationKey(span: ModelTraceSpanNode): string {
  return `${span.type ?? 'UNKNOWN'}::${span.title ?? 'Unknown'}`;
}

/**
 * Detects cycles via iterative DFS and marks cycle-causing edges as back-edges.
 * Uses the standard white/gray/black coloring: an edge to a gray (in-stack)
 * node is a back-edge. After this runs, the remaining forward edges form a DAG.
 */
function detectAndMarkBackEdges(nodes: WorkflowNode[], edges: WorkflowEdge[]): void {
  const WHITE = 0;
  const GRAY = 1;
  const BLACK = 2;

  const color = new Map<string, number>();
  for (const node of nodes) {
    color.set(node.id, WHITE);
  }

  const outgoing = new Map<string, WorkflowEdge[]>();
  for (const node of nodes) {
    outgoing.set(node.id, []);
  }
  for (const edge of edges) {
    outgoing.get(edge.sourceId)?.push(edge);
  }

  for (const node of nodes) {
    if (color.get(node.id) !== WHITE) {
      continue;
    }

    // Iterative DFS: stack entries are [nodeId, index into outgoing edges]
    const stack: [string, number][] = [[node.id, 0]];
    color.set(node.id, GRAY);

    while (stack.length > 0) {
      const top = stack[stack.length - 1];
      const nodeEdges = outgoing.get(top[0]) ?? [];

      if (top[1] >= nodeEdges.length) {
        color.set(top[0], BLACK);
        stack.pop();
        continue;
      }

      const edge = nodeEdges[top[1]];
      top[1]++;

      const targetColor = color.get(edge.targetId) ?? WHITE;
      if (targetColor === GRAY) {
        edge.isBackEdge = true;
      } else if (targetColor === WHITE) {
        color.set(edge.targetId, GRAY);
        stack.push([edge.targetId, 0]);
      }
    }
  }
}

/**
 * Custom layered layout algorithm for workflow graphs.
 * Assigns nodes to layers based on dependencies, then positions them.
 * Cycles are broken beforehand by detectAndMarkBackEdges so the BFS
 * operates on a guaranteed DAG.
 */
function applyLayeredLayout(
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  rootNodeId: string | null,
  config: GraphLayoutConfig,
): { width: number; height: number } | null {
  if (nodes.length === 0) {
    return null;
  }

  // Break cycles first — marks cycle-causing edges as back-edges
  detectAndMarkBackEdges(nodes, edges);

  // Build adjacency maps from forward edges only (back-edges excluded)
  const outgoing = new Map<string, string[]>();
  const incoming = new Map<string, string[]>();

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

  // Assign layers via longest-path BFS on the DAG
  const layers = new Map<string, number>();
  const queue: string[] = [];
  const inQueue = new Set<string>();
  let head = 0;

  // Always seed the root span node at layer 0
  if (rootNodeId && incoming.has(rootNodeId)) {
    layers.set(rootNodeId, 0);
    queue.push(rootNodeId);
    inQueue.add(rootNodeId);
  }

  // Seed remaining root nodes (no incoming forward edges)
  for (const node of nodes) {
    if (!layers.has(node.id) && (incoming.get(node.id) ?? []).length === 0) {
      layers.set(node.id, 0);
      queue.push(node.id);
      inQueue.add(node.id);
    }
  }

  // If no root nodes were found, the graph is likely broken
  if (queue.length === 0) {
    return null;
  }

  while (head < queue.length) {
    const nodeId = queue[head++];
    inQueue.delete(nodeId);

    const currentLayer = layers.get(nodeId) ?? 0;

    for (const targetId of outgoing.get(nodeId) ?? []) {
      const newLayer = currentLayer + 1;

      if ((layers.get(targetId) ?? -1) < newLayer) {
        layers.set(targetId, newLayer);
        if (!inQueue.has(targetId)) {
          queue.push(targetId);
          inQueue.add(targetId);
        }
      }
    }
  }

  // Place any remaining unassigned nodes (disconnected components) at the top layer
  for (const node of nodes) {
    if (!layers.has(node.id)) {
      layers.set(node.id, 0);
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

  // Mark any remaining same-layer or upward edges as back-edges
  for (const edge of edges) {
    if (edge.isBackEdge || edge.isNestedCall) {
      continue;
    }
    const sourceLayer = layers.get(edge.sourceId) ?? 0;
    const targetLayer = layers.get(edge.targetId) ?? 0;
    if (sourceLayer >= targetLayer) {
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
): { nodes: WorkflowNode[]; edges: WorkflowEdge[]; rootNodeId: string } | null {
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

  // Detect nested call edges (bidirectional relationships).
  // When both A->B and B->A edges exist, mark the second one as a nested call.
  const processedPairs = new Set<string>();
  for (const edge of edgeMap.values()) {
    const pairKey = [edge.sourceId, edge.targetId].sort().join('<->');
    if (processedPairs.has(pairKey)) {
      continue;
    }
    const reverseEdgeId = `${edge.targetId}->${edge.sourceId}`;
    if (edgeMap.has(reverseEdgeId)) {
      const reverseEdge = edgeMap.get(reverseEdgeId)!;
      // Keep the first edge as the primary direction, mark the reverse as nested
      reverseEdge.isNestedCall = true;
      processedPairs.add(pairKey);
    }
  }

  return {
    nodes: workflowNodes,
    edges: Array.from(edgeMap.values()),
    rootNodeId: getAggregationKey(rootNode),
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
  const dimensions = applyLayeredLayout(graph.nodes, graph.edges, graph.rootNodeId, config);
  if (!dimensions) {
    return { nodes: [], edges: [], width: 0, height: 0 };
  }

  return {
    nodes: graph.nodes,
    edges: graph.edges,
    width: dimensions.width,
    height: dimensions.height,
  };
}
