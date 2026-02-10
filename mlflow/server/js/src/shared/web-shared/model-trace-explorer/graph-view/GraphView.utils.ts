import { ModelSpanType } from '../ModelTrace.types';

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
      // Blue for LLM/Chat
      return theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100;

    case ModelSpanType.RETRIEVER:
      // Green for Retriever
      return theme.isDarkMode ? theme.colors.green800 : theme.colors.green100;

    case ModelSpanType.TOOL:
      // Red for Tool
      return theme.isDarkMode ? theme.colors.red800 : theme.colors.red100;

    case ModelSpanType.AGENT:
      // Purple for Agent
      return theme.isDarkMode ? '#3d2b5a' : '#f3e8ff';

    case ModelSpanType.CHAIN:
      // Indigo for Chain
      return theme.isDarkMode ? '#2d3458' : '#eef2ff';

    case ModelSpanType.EMBEDDING:
    case ModelSpanType.RERANKER:
      // Teal for embedding/reranker
      return theme.isDarkMode ? '#134e4a' : '#f0fdfa';

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
      // Purple
      return theme.isDarkMode ? '#a855f7' : '#9333ea';

    case ModelSpanType.CHAIN:
      // Indigo
      return theme.isDarkMode ? '#6366f1' : '#4f46e5';

    case ModelSpanType.EMBEDDING:
    case ModelSpanType.RERANKER:
      // Teal
      return theme.isDarkMode ? '#14b8a6' : '#0d9488';

    default:
      return theme.colors.border;
  }
}

/**
 * Gets the text color for a node based on its span type.
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
      // Purple
      return theme.isDarkMode ? '#c4b5fd' : '#7c3aed';

    case ModelSpanType.CHAIN:
      // Indigo
      return theme.isDarkMode ? '#a5b4fc' : '#4338ca';

    case ModelSpanType.EMBEDDING:
      // Teal
      return theme.isDarkMode ? '#5eead4' : '#0f766e';

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
