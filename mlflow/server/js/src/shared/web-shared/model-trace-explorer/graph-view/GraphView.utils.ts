import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType, GraphNodeType } from '../ModelTrace.types';
import type { GraphNode, GraphEdge, GraphLayout, GraphLayoutConfig } from './GraphView.types';
import { DEFAULT_GRAPH_LAYOUT_CONFIG } from './GraphView.types';

/**
 * Computes a tree layout for the given span tree.
 * Uses a bottom-up approach to calculate subtree widths, then positions nodes top-down.
 * Each subtree gets its own horizontal space, creating clear visual separation.
 */
export function computeGraphLayout(
  rootNode: ModelTraceSpanNode | null,
  config: GraphLayoutConfig = DEFAULT_GRAPH_LAYOUT_CONFIG,
): GraphLayout {
  if (!rootNode) {
    return { nodes: [], edges: [], width: 0, height: 0 };
  }

  const { nodeWidth, nodeHeight, horizontalSpacing, verticalSpacing, padding } = config;

  const nodeMap = new Map<string, GraphNode>();
  const edges: GraphEdge[] = [];
  const subtreeWidths = new Map<string, number>();

  // Build graph nodes and calculate subtree widths (bottom-up)
  function calculateSubtreeWidth(spanNode: ModelTraceSpanNode, layer: number): number {
    const id = String(spanNode.key);
    const children = spanNode.children ?? [];

    // Create graph node
    const graphNode: GraphNode = {
      id,
      spanNode,
      layer,
      orderIndex: 0,
      x: 0,
      y: 0,
      width: nodeWidth,
      height: nodeHeight,
    };
    nodeMap.set(id, graphNode);

    // If no children, subtree width is just the node width
    if (children.length === 0) {
      subtreeWidths.set(id, nodeWidth);
      return nodeWidth;
    }

    // Calculate children subtree widths and create edges
    let totalChildrenWidth = 0;
    for (let i = 0; i < children.length; i++) {
      const child = children[i];
      const childWidth = calculateSubtreeWidth(child, layer + 1);
      totalChildrenWidth += childWidth;

      // Add spacing between siblings
      if (i > 0) {
        totalChildrenWidth += horizontalSpacing;
      }

      // Create edge from parent to child
      const childId = String(child.key);
      const childNode = nodeMap.get(childId);
      if (childNode) {
        edges.push({
          sourceId: id,
          targetId: childId,
          sourceNode: graphNode,
          targetNode: childNode,
        });
      }
    }

    // Subtree width is the max of node width and total children width
    const subtreeWidth = Math.max(nodeWidth, totalChildrenWidth);
    subtreeWidths.set(id, subtreeWidth);
    return subtreeWidth;
  }

  calculateSubtreeWidth(rootNode, 0);

  // Position nodes (top-down)
  // Parent is centered above its children; children are spread within parent's subtree space
  function positionNode(spanNode: ModelTraceSpanNode, layer: number, leftX: number): void {
    const id = String(spanNode.key);
    const graphNode = nodeMap.get(id)!;
    const subtreeWidth = subtreeWidths.get(id) ?? nodeWidth;
    const children = spanNode.children ?? [];

    // Y coordinate based on layer
    graphNode.y = padding + layer * (nodeHeight + verticalSpacing);

    if (children.length === 0) {
      // Leaf node: center within its allocated space
      graphNode.x = leftX + (subtreeWidth - nodeWidth) / 2;
    } else {
      // Parent node: position children first, then center parent above them
      let childLeftX = leftX;

      // Calculate total children width to center them within parent's subtree space
      let totalChildrenWidth = 0;
      for (let i = 0; i < children.length; i++) {
        const childId = String(children[i].key);
        totalChildrenWidth += subtreeWidths.get(childId) ?? nodeWidth;
        if (i > 0) {
          totalChildrenWidth += horizontalSpacing;
        }
      }

      // Start children centered within parent's subtree space
      childLeftX = leftX + (subtreeWidth - totalChildrenWidth) / 2;

      // Position each child
      for (let i = 0; i < children.length; i++) {
        const child = children[i];
        const childId = String(child.key);
        const childSubtreeWidth = subtreeWidths.get(childId) ?? nodeWidth;

        positionNode(child, layer + 1, childLeftX);

        childLeftX += childSubtreeWidth + horizontalSpacing;
      }

      // Center parent above its children
      const firstChild = nodeMap.get(String(children[0].key))!;
      const lastChild = nodeMap.get(String(children[children.length - 1].key))!;
      const childrenCenterX = (firstChild.x + lastChild.x + nodeWidth) / 2;
      graphNode.x = childrenCenterX - nodeWidth / 2;
    }
  }

  const rootSubtreeWidth = subtreeWidths.get(String(rootNode.key)) ?? nodeWidth;
  positionNode(rootNode, 0, padding);

  // Calculate total dimensions
  let maxLayer = 0;
  nodeMap.forEach((node) => {
    maxLayer = Math.max(maxLayer, node.layer);
  });

  const totalWidth = rootSubtreeWidth + 2 * padding;
  const totalHeight = (maxLayer + 1) * (nodeHeight + verticalSpacing) - verticalSpacing + 2 * padding;

  // Update edge references with final node positions
  edges.forEach((edge) => {
    edge.sourceNode = nodeMap.get(edge.sourceId)!;
    edge.targetNode = nodeMap.get(edge.targetId)!;
  });

  return {
    nodes: Array.from(nodeMap.values()),
    edges,
    width: totalWidth,
    height: totalHeight,
  };
}

/**
 * Gets the background color for a node based on its span type.
 * Colors are consistent with ModelTraceExplorerIcon.tsx
 * Supports both ModelSpanType enum values and graph node type strings (HANDOFF, GUARDRAIL, etc.)
 */
export function getNodeBackgroundColor(
  spanType: ModelSpanType | string | undefined,
  theme: { colors: any; isDarkMode: boolean },
): string {
  switch (spanType) {
    case ModelSpanType.LLM:
    case ModelSpanType.CHAT_MODEL:
      // Blue for LLM/Chat
      return theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100;

    case ModelSpanType.RETRIEVER:
      // Green for Retriever
      return theme.isDarkMode ? theme.colors.green800 : theme.colors.green100;

    case ModelSpanType.TOOL:
      // Red for Tool
      return theme.isDarkMode ? theme.colors.red800 : theme.colors.red100;

    case ModelSpanType.AGENT:
    case GraphNodeType.TEAM:
      // Purple for Agent/Team (not in theme, using hex)
      return theme.isDarkMode ? '#3d2b5a' : '#f3e8ff';

    case ModelSpanType.CHAIN:
      // Indigo for Chain (not in theme, using hex)
      return theme.isDarkMode ? '#2d3458' : '#eef2ff';

    case ModelSpanType.EMBEDDING:
    case ModelSpanType.RERANKER:
      // Teal for embedding/reranker (not in theme, using hex)
      return theme.isDarkMode ? '#134e4a' : '#f0fdfa';

    // Graph node types from OpenAI agent tracer
    case GraphNodeType.HANDOFF:
      // Orange for Handoff (not in theme, using hex)
      return theme.isDarkMode ? '#7c2d12' : '#fff7ed';

    case GraphNodeType.GUARDRAIL:
      // Yellow for Guardrail
      return theme.isDarkMode ? theme.colors.yellow800 : theme.colors.yellow100;

    case GraphNodeType.CUSTOM:
      // Slate/Gray for Custom (not in theme, using hex)
      return theme.isDarkMode ? '#334155' : '#f1f5f9';

    case ModelSpanType.FUNCTION:
    case ModelSpanType.PARSER:
    case ModelSpanType.MEMORY:
    case ModelSpanType.UNKNOWN:
    default:
      // Default gray
      return theme.colors.backgroundSecondary;
  }
}

/**
 * Gets the border color for a node based on its span type.
 * Supports both ModelSpanType enum values and graph node type strings.
 */
export function getNodeBorderColor(
  spanType: ModelSpanType | string | undefined,
  theme: { colors: any; isDarkMode: boolean },
): string {
  switch (spanType) {
    case ModelSpanType.LLM:
    case ModelSpanType.CHAT_MODEL:
      return theme.isDarkMode ? theme.colors.blue500 : theme.colors.blue600;

    case ModelSpanType.RETRIEVER:
      return theme.isDarkMode ? theme.colors.green500 : theme.colors.green600;

    case ModelSpanType.TOOL:
      return theme.isDarkMode ? theme.colors.red500 : theme.colors.red600;

    case ModelSpanType.AGENT:
    case GraphNodeType.TEAM:
      // Purple
      return theme.isDarkMode ? '#a855f7' : '#9333ea';

    case ModelSpanType.CHAIN:
      // Indigo
      return theme.isDarkMode ? '#6366f1' : '#4f46e5';

    case ModelSpanType.EMBEDDING:
    case ModelSpanType.RERANKER:
      // Teal
      return theme.isDarkMode ? '#14b8a6' : '#0d9488';

    // Graph node types from OpenAI agent tracer
    case GraphNodeType.HANDOFF:
      // Orange
      return theme.isDarkMode ? '#f97316' : '#ea580c';

    case GraphNodeType.GUARDRAIL:
      return theme.isDarkMode ? theme.colors.yellow500 : theme.colors.yellow600;

    case GraphNodeType.CUSTOM:
      // Slate
      return theme.isDarkMode ? '#64748b' : '#475569';

    default:
      return theme.colors.border;
  }
}

/**
 * Gets the text color for a node based on its span type.
 * Supports both ModelSpanType enum values and graph node type strings.
 */
export function getNodeTextColor(
  spanType: ModelSpanType | string | undefined,
  theme: { colors: any; isDarkMode: boolean },
): string {
  switch (spanType) {
    case ModelSpanType.LLM:
    case ModelSpanType.CHAT_MODEL:
      return theme.isDarkMode ? theme.colors.blue300 : theme.colors.blue700;

    case ModelSpanType.RETRIEVER:
      return theme.isDarkMode ? theme.colors.green300 : theme.colors.green700;

    case ModelSpanType.TOOL:
      return theme.isDarkMode ? theme.colors.red300 : theme.colors.red700;

    case ModelSpanType.AGENT:
    case GraphNodeType.TEAM:
      // Purple
      return theme.isDarkMode ? '#c4b5fd' : '#7c3aed';

    case ModelSpanType.CHAIN:
      // Indigo
      return theme.isDarkMode ? '#a5b4fc' : '#4338ca';

    case ModelSpanType.EMBEDDING:
      // Teal
      return theme.isDarkMode ? '#5eead4' : '#0f766e';

    // Graph node types from OpenAI agent tracer
    case GraphNodeType.HANDOFF:
      // Orange
      return theme.isDarkMode ? '#fdba74' : '#c2410c';

    case GraphNodeType.GUARDRAIL:
      return theme.isDarkMode ? theme.colors.yellow300 : theme.colors.yellow700;

    case GraphNodeType.CUSTOM:
      // Slate
      return theme.isDarkMode ? '#cbd5e1' : '#334155';

    default:
      return theme.colors.textPrimary;
  }
}

/**
 * Truncates text to a maximum length with ellipsis
 */
export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) {
    return text;
  }
  return text.substring(0, maxLength - 3) + '...';
}
