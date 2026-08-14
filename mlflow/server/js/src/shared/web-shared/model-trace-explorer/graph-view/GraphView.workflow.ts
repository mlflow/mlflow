import type { ModelTraceSpanNode } from '../ModelTrace.types';
import type { WorkflowNode, WorkflowEdge, WorkflowLayout, GraphLayoutConfig } from './GraphView.types';
import { DEFAULT_WORKFLOW_LAYOUT_CONFIG } from './GraphView.types';

/**
 * Flattens the span tree into a list of all spans using an iterative DFS.
 */
function flattenSpans(root: ModelTraceSpanNode): ModelTraceSpanNode[] {
  const result: ModelTraceSpanNode[] = [];
  const stack: ModelTraceSpanNode[] = [root];

  while (stack.length > 0) {
    const node = stack.pop();
    if (!node) {
      continue;
    }
    result.push(node);
    if (node.children) {
      for (let i = node.children.length - 1; i >= 0; i--) {
        stack.push(node.children[i]);
      }
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
 * Collects all descendant node IDs in a subtree via BFS.
 */
function getSubtreeIds(rootId: string, childrenMap: Map<string, string[]>): string[] {
  const result: string[] = [];
  const queue = childrenMap.get(rootId) ?? [];
  let head = 0;
  while (head < queue.length) {
    const id = queue[head++];
    result.push(id);
    for (const childId of childrenMap.get(id) ?? []) {
      queue.push(childId);
    }
  }
  return result;
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
    layerGroups.get(layer)?.push(node);
  }

  // Position nodes using parent-centered layout (Reingold-Tilford style)
  const sortedLayers = Array.from(layerGroups.keys()).sort((a, b) => a - b);

  // Build parent→children map from forward edges
  const childrenMap = new Map<string, string[]>();
  for (const edge of edges) {
    if (!edge.isBackEdge) {
      if (!childrenMap.has(edge.sourceId)) {
        childrenMap.set(edge.sourceId, []);
      }
      childrenMap.get(edge.sourceId)?.push(edge.targetId);
    }
  }

  // Node lookup
  const nodeById = new Map<string, WorkflowNode>();
  for (const node of nodes) {
    nodeById.set(node.id, node);
  }

  // Track which nodes have been positioned
  const positioned = new Set<string>();

  // Assign Y positions for all layers
  for (const layerIndex of sortedLayers) {
    const y = config.padding + layerIndex * (config.nodeHeight + config.verticalSpacing);
    for (const node of layerGroups.get(layerIndex) ?? []) {
      node.y = y;
    }
  }

  // Bottom-up pass: deepest layer first
  const reversedLayers = [...sortedLayers].reverse();

  for (const layerIndex of reversedLayers) {
    const layerNodes = layerGroups.get(layerIndex) ?? [];

    // Step 1: Position leaf nodes (no children) sequentially
    let nextLeafX = config.padding;
    for (const node of layerNodes) {
      const childIds = (childrenMap.get(node.id) ?? []).filter((id) => positioned.has(id));
      if (childIds.length === 0) {
        node.x = nextLeafX;
        nextLeafX += node.width + config.horizontalSpacing;
        positioned.add(node.id);
      }
    }

    // Step 2: Center parent nodes above their positioned children
    for (const node of layerNodes) {
      if (positioned.has(node.id)) {
        continue;
      }
      const childIds = (childrenMap.get(node.id) ?? []).filter((id) => positioned.has(id));
      if (childIds.length > 0) {
        const children = childIds.map((id) => nodeById.get(id) as WorkflowNode);
        const childLeft = Math.min(...children.map((c) => c.x));
        const childRight = Math.max(...children.map((c) => c.x + c.width));
        node.x = (childLeft + childRight) / 2 - node.width / 2;
        positioned.add(node.id);
      }
    }

    // Step 3: Position any remaining unpositioned nodes in this layer
    for (const node of layerNodes) {
      if (!positioned.has(node.id)) {
        node.x = nextLeafX;
        nextLeafX += node.width + config.horizontalSpacing;
        positioned.add(node.id);
      }
    }

    // Step 4: Resolve overlaps — sort by x, push right as needed
    layerNodes.sort((a, b) => a.x - b.x);
    for (let i = 1; i < layerNodes.length; i++) {
      const prev = layerNodes[i - 1];
      const current = layerNodes[i];
      const minX = prev.x + prev.width + config.horizontalSpacing;
      if (current.x >= minX) continue;

      const delta = minX - current.x;
      current.x = minX;

      // Also shift this node's subtree to maintain centering
      const subtreeIds = getSubtreeIds(current.id, childrenMap);
      for (const id of subtreeIds) {
        const descendant = nodeById.get(id);
        if (!descendant) continue;
        descendant.x += delta;
      }
    }
  }

  // Compute bounding box and shift everything to start at padding
  let minX = Infinity;
  let maxX = -Infinity;
  for (const node of nodes) {
    minX = Math.min(minX, node.x);
    maxX = Math.max(maxX, node.x + node.width);
  }
  const totalWidth = maxX - minX;
  const shift = config.padding - minX;
  if (shift !== 0) {
    for (const node of nodes) {
      node.x += shift;
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
    nodeGroups.get(key)?.push(span);
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
      const edge = edgeMap.get(edgeId);
      if (edge) {
        edge.count++;
      }
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
      const reverseEdge = edgeMap.get(reverseEdgeId);
      // Keep the first edge as the primary direction, mark the reverse as nested
      if (reverseEdge) {
        reverseEdge.isNestedCall = true;
      }
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
