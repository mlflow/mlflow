/**
 * Assistant Agent integration for MLflow UI.
 *
 * This module provides components for AI-powered assistant experience.
 */
export { AssistantProvider, useAssistant } from './AssistantContext';
export { AssistantChatPanel } from './AssistantChatPanel';
export {
  AssistantRouteContextProvider,
  useRegisterAssistantContext,
  useAssistantPageContext,
  useAssistantPageContextActions,
  useRegisterSelectedIds,
} from './AssistantPageContext';

export * from './AssistantService';

export type {
  ChatMessage,
  AssistantAgentState,
  AssistantAgentActions,
  AssistantAgentContextType,
  MessageRequest,
  KnownAssistantContext,
  AssistantContextKey,
} from './types';
