import { ModelSpanType } from '../ModelTrace.types';
import type { WorkflowLayout } from './GraphView.types';

/**
 * Gets the background color for a node based on its span type.
 * Colors are consistent with ModelTraceExplorerIcon.tsx
 */
export function getNodeBackgroundColor(
  spanType: ModelSpanType | string | undefined,
  theme: { colors: any; isDarkMode: boolean },
): string {
  switch (spanType) {
    case ModelSpanType.LLM:
    case ModelSpanType.CHAT_MODEL:
      return theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100;
    case ModelSpanType.RETRIEVER:
      return theme.isDarkMode ? theme.colors.green800 : theme.colors.green100;
    case ModelSpanType.TOOL:
      return theme.isDarkMode ? theme.colors.red800 : theme.colors.red100;
    default:
      return theme.colors.backgroundSecondary;
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

/**
 * Computes the path from a workflow node to root nodes via BFS over incoming edges.
 */
export function computeWorkflowPathToRoot(
  nodeId: string | null,
  layout: WorkflowLayout | null,
): { nodeIds: Set<string>; edgeIds: Set<string> } {
  const nodeIds = new Set<string>();
  const edgeIds = new Set<string>();

  if (!nodeId || !layout || layout.nodes.length === 0) {
    return { nodeIds, edgeIds };
  }

  nodeIds.add(nodeId);

  const incomingEdgesMap = new Map<string, typeof layout.edges>();
  for (const edge of layout.edges) {
    if (!incomingEdgesMap.has(edge.targetId)) {
      incomingEdgesMap.set(edge.targetId, []);
    }
    incomingEdgesMap.get(edge.targetId)!.push(edge);
  }

  const toProcess = [nodeId];
  const visited = new Set<string>([nodeId]);
  let head = 0;

  while (head < toProcess.length) {
    const currentNodeId = toProcess[head++];
    const incomingEdges = incomingEdgesMap.get(currentNodeId) || [];

    for (const edge of incomingEdges) {
      const parentId = edge.sourceId;
      edgeIds.add(`${edge.sourceId}->${edge.targetId}`);

      if (!visited.has(parentId)) {
        visited.add(parentId);
        nodeIds.add(parentId);
        toProcess.push(parentId);
      }
    }
  }

  return { nodeIds, edgeIds };
}
