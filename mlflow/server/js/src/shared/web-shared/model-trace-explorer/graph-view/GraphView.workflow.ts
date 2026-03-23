import type { ModelTraceSpanNode } from '../ModelTrace.types';
import type { GraphSchema, WorkflowNode, WorkflowEdge, WorkflowLayout, GraphLayoutConfig } from './GraphView.types';
import { DEFAULT_WORKFLOW_LAYOUT_CONFIG } from './GraphView.types';

/**
 * Flattens the span tree into a list of all spans using an iterative DFS.
 */
function flattenSpans(root: ModelTraceSpanNode): ModelTraceSpanNode[] {
  const result: ModelTraceSpanNode[] = [];
  const stack: ModelTraceSpanNode[] = [root];

  while (stack.length > 0) {
    const node = stack.pop();
    if (!node) break;
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
 * Resolves overlapping nodes within a single layer along the cross axis.
 * Sorts nodes by their cross-axis position and pushes overlapping nodes
 * (and their subtrees) apart by the required spacing.
 */
function resolveLayerOverlaps(
  layerNodes: WorkflowNode[],
  spacing: number,
  isHorizontal: boolean,
  childrenMap: Map<string, string[]>,
  nodeById: Map<string, WorkflowNode>,
): void {
  if (isHorizontal) {
    layerNodes.sort((a, b) => a.y - b.y);
  } else {
    layerNodes.sort((a, b) => a.x - b.x);
  }

  for (let i = 1; i < layerNodes.length; i++) {
    const prev = layerNodes[i - 1];
    const current = layerNodes[i];
    const prevEnd = isHorizontal ? prev.y + prev.height : prev.x + prev.width;
    const currentStart = isHorizontal ? current.y : current.x;

    if (currentStart >= prevEnd + spacing) continue;

    const delta = prevEnd + spacing - currentStart;
    if (isHorizontal) {
      current.y += delta;
    } else {
      current.x += delta;
    }

    for (const id of getSubtreeIds(current.id, childrenMap)) {
      const descendant = nodeById.get(id);
      if (!descendant) continue;
      if (isHorizontal) {
        descendant.y += delta;
      } else {
        descendant.x += delta;
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

  const isHorizontal = config.orientation === 'LR';

  // In TB mode: layers advance along Y, siblings spread along X.
  // In LR mode: layers advance along X, siblings spread along Y.
  const layerNodeSize = isHorizontal ? config.nodeWidth : config.nodeHeight;
  const layerSpacing = isHorizontal ? config.horizontalSpacing : config.verticalSpacing;
  const crossNodeSize = isHorizontal ? config.nodeHeight : config.nodeWidth;
  const crossSpacing = isHorizontal ? config.verticalSpacing : config.horizontalSpacing;

  // Assign primary-axis positions for all layers
  for (const layerIndex of sortedLayers) {
    const primary = config.padding + layerIndex * (layerNodeSize + layerSpacing);
    for (const node of layerGroups.get(layerIndex) ?? []) {
      if (isHorizontal) {
        node.x = primary;
      } else {
        node.y = primary;
      }
    }
  }

  // Bottom-up pass: deepest layer first — position along the cross axis
  const reversedLayers = [...sortedLayers].reverse();

  for (const layerIndex of reversedLayers) {
    const layerNodes = layerGroups.get(layerIndex) ?? [];

    // Step 1: Position leaf nodes (no children) sequentially
    let nextLeafCross = config.padding;
    for (const node of layerNodes) {
      const childIds = (childrenMap.get(node.id) ?? []).filter((id) => positioned.has(id));
      if (childIds.length === 0) {
        if (isHorizontal) {
          node.y = nextLeafCross;
        } else {
          node.x = nextLeafCross;
        }
        nextLeafCross += crossNodeSize + crossSpacing;
        positioned.add(node.id);
      }
    }

    // Step 2: Center parent nodes above/beside their positioned children
    for (const node of layerNodes) {
      if (positioned.has(node.id)) {
        continue;
      }
      const childIds = (childrenMap.get(node.id) ?? []).filter((id) => positioned.has(id));
      if (childIds.length > 0) {
        const children = childIds.map((id) => nodeById.get(id)).filter(Boolean) as WorkflowNode[];
        if (isHorizontal) {
          const childTop = Math.min(...children.map((c) => c.y));
          const childBottom = Math.max(...children.map((c) => c.y + c.height));
          node.y = (childTop + childBottom) / 2 - node.height / 2;
        } else {
          const childLeft = Math.min(...children.map((c) => c.x));
          const childRight = Math.max(...children.map((c) => c.x + c.width));
          node.x = (childLeft + childRight) / 2 - node.width / 2;
        }
        positioned.add(node.id);
      }
    }

    // Step 3: Position any remaining unpositioned nodes in this layer
    for (const node of layerNodes) {
      if (!positioned.has(node.id)) {
        if (isHorizontal) {
          node.y = nextLeafCross;
        } else {
          node.x = nextLeafCross;
        }
        nextLeafCross += crossNodeSize + crossSpacing;
        positioned.add(node.id);
      }
    }

    // Step 4: Resolve overlaps along the cross axis
    resolveLayerOverlaps(layerNodes, crossSpacing, isHorizontal, childrenMap, nodeById);
  }

  // Compute bounding box and shift everything to start at padding
  if (isHorizontal) {
    let minY = Infinity;
    for (const node of nodes) {
      minY = Math.min(minY, node.y);
    }
    const shift = config.padding - minY;
    if (shift !== 0) {
      for (const node of nodes) {
        node.y += shift;
      }
    }
  } else {
    let minX = Infinity;
    for (const node of nodes) {
      minX = Math.min(minX, node.x);
    }
    const shift = config.padding - minX;
    if (shift !== 0) {
      for (const node of nodes) {
        node.x += shift;
      }
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

  // Compute final bounding dimensions
  let totalWidth = 0;
  let totalHeight = 0;
  for (const node of nodes) {
    totalWidth = Math.max(totalWidth, node.x + node.width);
    totalHeight = Math.max(totalHeight, node.y + node.height);
  }

  return {
    width: totalWidth + config.padding,
    height: totalHeight + config.padding,
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
      const existingEdge = edgeMap.get(edgeId);
      if (existingEdge) existingEdge.count++;
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
    const reverseEdge = edgeMap.get(reverseEdgeId);
    if (reverseEdge) {
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

/**
 * Safely access metadata from span attributes (may be Record or array form).
 */
function getSpanMetadata(span: ModelTraceSpanNode): Record<string, unknown> | undefined {
  const attrs = span.attributes;
  if (!attrs || Array.isArray(attrs)) return undefined;
  const meta = attrs['metadata'];
  if (meta && typeof meta === 'object' && !Array.isArray(meta)) {
    return meta as Record<string, unknown>;
  }
  return undefined;
}

/**
 * Resolves the graph node ID that a span belongs to.
 *
 * Strategy (in priority order):
 *  1. Framework-specific metadata (e.g. LangGraph's `langgraph_node`)
 *  2. Span name matching against known schema node IDs
 *
 * This keeps span-to-node mapping framework-agnostic: any framework
 * whose span names correspond to graph node IDs will work out of the box.
 */
function resolveGraphNodeId(span: ModelTraceSpanNode, schemaNodeIds: Set<string>): string | undefined {
  const meta = getSpanMetadata(span);
  if (meta) {
    // LangGraph convention
    if (typeof meta['langgraph_node'] === 'string') return meta['langgraph_node'] as string;
  }

  // Universal fallback: match span name against schema node IDs
  const title = typeof span.title === 'string' ? span.title : undefined;
  if (title && schemaNodeIds.has(title)) return title;

  return undefined;
}

/**
 * Builds an ordered list of graph-node transitions from span data.
 *
 * Strategy (in priority order):
 *  1. Framework-specific step metadata (e.g. LangGraph's `langgraph_step`)
 *  2. Span start timestamps (universal — works for any framework)
 *
 * Each returned entry pairs a graph node ID with a monotonic step number.
 */
function buildStepEntries(
  allSpans: ModelTraceSpanNode[],
  schemaNodeIds: Set<string>,
  parentNodeMap: Map<string, string | undefined>,
): Array<{ node: string; step: number }> {
  // Try framework-specific step metadata first
  const metadataEntries: Array<{ node: string; step: number }> = [];
  const seenSteps = new Set<number>();

  for (const span of allSpans) {
    const graphNode = resolveGraphNodeId(span, schemaNodeIds);
    if (!graphNode) continue;

    const parentGraphNode = parentNodeMap.get(String(span.key));
    if (parentGraphNode === graphNode) continue;

    const meta = getSpanMetadata(span);
    const lgStep = meta?.['langgraph_step'];
    if (typeof lgStep === 'number' && !seenSteps.has(lgStep)) {
      seenSteps.add(lgStep);
      metadataEntries.push({ node: graphNode, step: lgStep });
    }
  }

  if (metadataEntries.length > 0) {
    metadataEntries.sort((a, b) => a.step - b.step);
    return metadataEntries;
  }

  // Universal fallback: derive execution order from span start timestamps
  const timestampEntries: Array<{ node: string; start: number }> = [];
  for (const span of allSpans) {
    const graphNode = resolveGraphNodeId(span, schemaNodeIds);
    if (!graphNode) continue;
    const parentGraphNode = parentNodeMap.get(String(span.key));
    if (parentGraphNode === graphNode) continue;
    timestampEntries.push({ node: graphNode, start: span.start });
  }

  timestampEntries.sort((a, b) => a.start - b.start);

  // Deduplicate consecutive same-node entries and assign synthetic step numbers
  const result: Array<{ node: string; step: number }> = [];
  let stepCounter = 0;
  let prevNode: string | undefined;
  for (const entry of timestampEntries) {
    if (entry.node !== prevNode) {
      result.push({ node: entry.node, step: stepCounter++ });
      prevNode = entry.node;
    }
  }
  return result;
}

/**
 * Computes a logical flow layout from a graph schema and span data.
 * Instead of inferring the graph from parent-child span hierarchy, this uses
 * the framework's native graph topology (nodes + edges + conditional edges).
 */
export function computeLogicalFlowLayout(
  schema: GraphSchema,
  rootNode: ModelTraceSpanNode | null,
  config: GraphLayoutConfig = DEFAULT_WORKFLOW_LAYOUT_CONFIG,
): WorkflowLayout {
  if (!schema.nodes || schema.nodes.length === 0) {
    return { nodes: [], edges: [], width: 0, height: 0 };
  }

  const allSpans = rootNode ? flattenSpans(rootNode) : [];
  const schemaNodeIds = new Set(schema.nodes.map((n) => n.id));

  // Build a map from span key to its parent's graph node assignment.
  // This lets us identify "entry" spans for each node (where the parent belongs
  // to a different node or has no node), filtering out internal child spans
  // that the framework tags with the same graph node as their parent.
  const parentNodeMap = new Map<string, string | undefined>();
  function buildParentNodeMap(node: ModelTraceSpanNode, parentGraphNode?: string) {
    parentNodeMap.set(String(node.key), parentGraphNode);
    for (const child of node.children ?? []) {
      const thisGraphNode = resolveGraphNodeId(node, schemaNodeIds);
      buildParentNodeMap(child, thisGraphNode ?? parentGraphNode);
    }
  }
  if (rootNode) buildParentNodeMap(rootNode);

  // Group spans by their graph node, only counting "entry" spans
  // where the span's node differs from its parent's node.
  const spansByNodeName = new Map<string, ModelTraceSpanNode[]>();
  for (const span of allSpans) {
    const graphNode = resolveGraphNodeId(span, schemaNodeIds);
    if (graphNode) {
      const parentGraphNode = parentNodeMap.get(String(span.key));
      if (parentGraphNode === graphNode) continue;
      if (!spansByNodeName.has(graphNode)) {
        spansByNodeName.set(graphNode, []);
      }
      spansByNodeName.get(graphNode)?.push(span);
    }
  }

  // Build step-order transitions to determine executed edges.
  const stepEntries = buildStepEntries(allSpans, schemaNodeIds, parentNodeMap);

  // Build ordered transition list with step numbers for edge annotations.
  // Each transition gets a 1-indexed sequence number representing the order
  // of execution through the graph.
  const observedTransitions = new Set<string>();
  const edgeStepSequences = new Map<string, number[]>();
  let transitionCounter = 1;

  if (stepEntries.length > 0) {
    const startEdge = `__start__->${stepEntries[0].node}`;
    observedTransitions.add(startEdge);
    edgeStepSequences.set(startEdge, [transitionCounter++]);
  }
  for (let i = 1; i < stepEntries.length; i++) {
    if (stepEntries[i].node !== stepEntries[i - 1].node) {
      const key = `${stepEntries[i - 1].node}->${stepEntries[i].node}`;
      observedTransitions.add(key);
      if (!edgeStepSequences.has(key)) {
        edgeStepSequences.set(key, []);
      }
      edgeStepSequences.get(key)?.push(transitionCounter++);
    }
  }
  if (stepEntries.length > 0) {
    const endEdge = `${stepEntries[stepEntries.length - 1].node}->__end__`;
    observedTransitions.add(endEdge);
    edgeStepSequences.set(endEdge, [transitionCounter++]);
  }

  // Compute node visit order — each step entry is a node visit, numbered 1,2,3...
  const nodeExecutionOrder = new Map<string, number[]>();
  let visitCounter = 1;
  for (const entry of stepEntries) {
    if (!nodeExecutionOrder.has(entry.node)) {
      nodeExecutionOrder.set(entry.node, []);
    }
    nodeExecutionOrder.get(entry.node)?.push(visitCounter++);
  }

  // Create WorkflowNodes from schema
  const workflowNodes: WorkflowNode[] = [];
  for (const schemaNode of schema.nodes) {
    const isStructural = schemaNode.id === '__start__' || schemaNode.id === '__end__';
    const matchedSpans = spansByNodeName.get(schemaNode.id) ?? [];
    const displayName = isStructural
      ? schemaNode.id === '__start__'
        ? 'START'
        : 'END'
      : (schemaNode.data?.name ?? schemaNode.id);
    const nodeType = isStructural ? 'CHAIN' : matchedSpans.length > 0 ? (matchedSpans[0].type ?? 'CHAIN') : 'CHAIN';

    workflowNodes.push({
      id: schemaNode.id,
      displayName,
      nodeType,
      count: matchedSpans.length,
      spans: matchedSpans,
      x: 0,
      y: 0,
      width: isStructural ? Math.round(config.nodeWidth * 0.6) : config.nodeWidth,
      height: isStructural ? Math.round(config.nodeHeight * 0.7) : config.nodeHeight,
      isStructural,
      isExecuted: isStructural ? true : matchedSpans.length > 0,
      executionOrder: nodeExecutionOrder.get(schemaNode.id),
    });
  }

  // Create WorkflowEdges from schema
  const workflowEdges: WorkflowEdge[] = [];
  for (const schemaEdge of schema.edges) {
    const transitionKey = `${schemaEdge.source}->${schemaEdge.target}`;
    const isExecuted = observedTransitions.has(transitionKey);
    const steps = edgeStepSequences.get(transitionKey);
    workflowEdges.push({
      sourceId: schemaEdge.source,
      targetId: schemaEdge.target,
      count: steps?.length ?? 0,
      isBackEdge: false,
      isConditional: schemaEdge.conditional ?? false,
      condition: schemaEdge.data ?? undefined,
      isExecuted,
      stepSequence: steps,
    });
  }

  // Mark return edges: if both A→B and B→A exist, the non-conditional one going
  // back toward the earlier node is the return edge. These render subtly.
  const edgeSet = new Set(workflowEdges.map((e) => `${e.sourceId}->${e.targetId}`));
  for (const edge of workflowEdges) {
    const reverseKey = `${edge.targetId}->${edge.sourceId}`;
    if (edgeSet.has(reverseKey) && !edge.isConditional) {
      edge.isReturnEdge = true;
    }
  }

  // Reorder nodes so loop-back tool nodes appear before terminal-path nodes.
  // The layout algorithm positions leaf nodes left-to-right in array order,
  // so putting __end__ and its predecessors last places the exit path on the right.
  const endPredecessors = new Set<string>();
  for (const edge of workflowEdges) {
    if (edge.targetId === '__end__') endPredecessors.add(edge.sourceId);
  }
  workflowNodes.sort((a, b) => {
    // __start__ always first, __end__ always last
    if (a.id === '__start__') return -1;
    if (b.id === '__start__') return 1;
    if (a.id === '__end__') return 1;
    if (b.id === '__end__') return -1;
    // Nodes that lead to __end__ go after nodes that don't
    const aToEnd = endPredecessors.has(a.id) ? 1 : 0;
    const bToEnd = endPredecessors.has(b.id) ? 1 : 0;
    return aToEnd - bToEnd;
  });

  const startNodeId = schema.nodes.find((n) => n.id === '__start__')?.id ?? workflowNodes[0]?.id ?? null;
  const dimensions = applyLayeredLayout(workflowNodes, workflowEdges, startNodeId, config);
  if (!dimensions) {
    return { nodes: [], edges: [], width: 0, height: 0 };
  }

  // applyLayeredLayout marks cycle-causing edges as back-edges for layout purposes.
  // In a schema-based graph all edges are intentional, so reset the back-edge flag
  // to avoid alarming red styling. The layout has already been computed correctly.
  for (const edge of workflowEdges) {
    edge.isBackEdge = false;
  }

  // Post-layout: move terminal-path nodes (those leading to __end__) to the right
  // of loop-back nodes so the exit path sits on the right side of the graph.
  const nodeById = new Map(workflowNodes.map((n) => [n.id, n]));
  for (const predId of endPredecessors) {
    const predNode = nodeById.get(predId);
    const endNode = nodeById.get('__end__');
    if (!predNode || !endNode) continue;

    // Find all nodes in the same layer as the predecessor
    const sameLayerNodes = workflowNodes.filter((n) => n.y === predNode.y && n.id !== predNode.id && !n.isStructural);
    if (sameLayerNodes.length === 0) continue;

    // If the predecessor is already to the right of all siblings, nothing to do
    const maxSiblingRight = Math.max(...sameLayerNodes.map((n) => n.x + n.width));
    if (predNode.x >= maxSiblingRight) continue;

    // Move predecessor to the right of the rightmost sibling
    const newX = maxSiblingRight + config.horizontalSpacing;
    const dx = newX - predNode.x;
    predNode.x = newX;

    // Also shift __end__ by the same amount to stay centered below
    endNode.x += dx;

    // Recalculate total width
    dimensions.width = Math.max(
      dimensions.width,
      newX + predNode.width + config.padding,
      endNode.x + endNode.width + config.padding,
    );
  }

  return {
    nodes: workflowNodes,
    edges: workflowEdges,
    width: dimensions.width,
    height: dimensions.height,
  };
}
