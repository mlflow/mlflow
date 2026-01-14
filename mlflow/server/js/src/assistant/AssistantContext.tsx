/**
 * React Context for Assistant Agent.
 * Provides Assistant functionality accessible from anywhere in MLflow.
 */

import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from 'react';

import type { AssistantAgentContextType, ChatMessage } from './types';
import { sendMessageStream } from './AssistantService';

const AssistantReactContext = createContext<AssistantAgentContextType | null>(null);

const generateMessageId = (): string => {
  return `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

export const AssistantProvider = ({ children }: { children: ReactNode }) => {
  // Panel state
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  // Chat state
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentStatus, setCurrentStatus] = useState<string | null>(null);

  // Use ref to track current streaming message
  const streamingMessageRef = useRef<string>('');

  const appendToStreamingMessage = useCallback((text: string) => {
    streamingMessageRef.current += text;
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1];
      if (lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming) {
        return [...prev.slice(0, -1), { ...lastMessage, content: streamingMessageRef.current }];
      }
      return prev;
    });
  }, []);

  const finalizeStreamingMessage = useCallback(() => {
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1];
      if (lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming) {
        return [...prev.slice(0, -1), { ...lastMessage, isStreaming: false }];
      }
      return prev;
    });
    streamingMessageRef.current = '';
    setIsStreaming(false);
    setCurrentStatus(null);
  }, []);

  const handleStatus = useCallback((status: string) => {
    setCurrentStatus(status);
  }, []);

  const handleSessionId = useCallback((newSessionId: string) => {
    setSessionId(newSessionId);
  }, []);

  const handleStreamError = useCallback((errorMsg: string) => {
    setError(errorMsg);
    setIsStreaming(false);
    setCurrentStatus(null);
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1];
      if (lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming) {
        return [...prev.slice(0, -1), { ...lastMessage, content: `Error: ${errorMsg}`, isStreaming: false }];
      }
      return prev;
    });
  }, []);

  // Actions
  const openPanel = useCallback(() => {
    setIsPanelOpen(true);
    setError(null);
  }, []);

  const closePanel = useCallback(() => {
    setIsPanelOpen(false);
  }, []);

  const reset = useCallback(() => {
    setSessionId(null);
    setMessages([]);
    setIsStreaming(false);
    setError(null);
    streamingMessageRef.current = '';
  }, []);

  const startChat = useCallback(
    async (prompt?: string) => {
      setError(null);
      setIsStreaming(true);

      // Add user message if prompt provided
      if (prompt) {
        setMessages((prev) => [
          ...prev,
          {
            id: generateMessageId(),
            role: 'user',
            content: prompt,
            timestamp: new Date(),
          },
        ]);
      }

      // Add streaming assistant message placeholder
      streamingMessageRef.current = '';
      setMessages((prev) => [
        ...prev,
        {
          id: generateMessageId(),
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          isStreaming: true,
        },
      ]);

      try {
        await sendMessageStream(
          {
            message: prompt || '',
            session_id: sessionId ?? undefined,
          },
          appendToStreamingMessage,
          handleStreamError,
          finalizeStreamingMessage,
          handleStatus,
          handleSessionId,
        );
      } catch (err) {
        handleStreamError(err instanceof Error ? err.message : 'Failed to start chat');
      }
    },
    [sessionId, appendToStreamingMessage, handleStreamError, finalizeStreamingMessage, handleStatus, handleSessionId],
  );

  const handleSendMessage = useCallback(
    (message: string) => {
      if (!sessionId) {
        startChat(message);
        return;
      }

      setError(null);
      setIsStreaming(true);

      // Add user message
      setMessages((prev) => [
        ...prev,
        {
          id: generateMessageId(),
          role: 'user',
          content: message,
          timestamp: new Date(),
        },
      ]);

      // Add streaming assistant message placeholder
      streamingMessageRef.current = '';
      setMessages((prev) => [
        ...prev,
        {
          id: generateMessageId(),
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          isStreaming: true,
        },
      ]);

      // Send message and stream response
      sendMessageStream(
        { session_id: sessionId, message },
        appendToStreamingMessage,
        handleStreamError,
        finalizeStreamingMessage,
        handleStatus,
        handleSessionId,
      );
    },
    [sessionId, appendToStreamingMessage, handleStreamError, finalizeStreamingMessage, handleStatus, handleSessionId],
  );

  const value: AssistantAgentContextType = {
    // State
    isPanelOpen,
    sessionId,
    messages,
    isStreaming,
    error,
    currentStatus,
    // Actions
    openPanel,
    closePanel,
    sendMessage: handleSendMessage,
    reset,
  };

  return <AssistantReactContext.Provider value={value}>{children}</AssistantReactContext.Provider>;
};

// Default disabled state when no provider is present
const disabledAssistantContext: AssistantAgentContextType = {
  isPanelOpen: false,
  sessionId: null,
  messages: [],
  isStreaming: false,
  error: null,
  currentStatus: null,
  openPanel: () => {},
  closePanel: () => {},
  sendMessage: () => {},
  reset: () => {},
};

/**
 * Hook to access the Assistant context.
 */
export const useAssistant = (): AssistantAgentContextType => {
  const context = useContext(AssistantReactContext);
  return context ?? disabledAssistantContext;
};
