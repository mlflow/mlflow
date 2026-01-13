/**
 * Assistant Agent integration for MLflow UI.
 *
 * This module provides components for AI-powered assistant experience.
 */
export { AssistantProvider, useAssistant } from './AssistantContext';
export { AssistantChatPanel } from './AssistantChatPanel';
export { AssistantButton } from './AssistantButton';

export * from './AssistantService';

export type {
  ChatMessage,
  AssistantAgentState,
  AssistantAgentActions,
  AssistantAgentContextType,
  MessageRequest,
} from './types';
