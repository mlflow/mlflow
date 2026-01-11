/**
 * React Context for Assistant Agent.
 * Provides Assistant functionality accessible from anywhere in MLflow.
 */

import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from 'react';

import type { AssistantAgentContextType, ChatMessage } from './types';
import { cancelSession as cancelSessionApi, sendMessageStream } from './AssistantService';
import { useAssistantPageContextActions } from './AssistantPageContext';

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

  // NB: Using the actions hook to avoid re-rendering the component when the context changes.
  const { getContext: getPageContext } = useAssistantPageContextActions();

  // Use ref to track active EventSource for cancellation
  const eventSourceRef = useRef<EventSource | null>(null);

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
    eventSourceRef.current = null;
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
    eventSourceRef.current = null;
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1];
      if (lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming) {
        return [...prev.slice(0, -1), { ...lastMessage, content: `Error: ${errorMsg}`, isStreaming: false }];
      }
      return prev;
    });
  }, []);

  const handleInterrupted = useCallback(() => {
    setIsStreaming(false);
    setCurrentStatus(null);
    eventSourceRef.current = null;
    streamingMessageRef.current = '';
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1];
      if (lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming) {
        return [...prev.slice(0, -1), { ...lastMessage, isStreaming: false, isInterrupted: true }];
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
        const pageContext = getPageContext();
        const result = await sendMessageStream(
          {
            message: prompt || '',
            session_id: sessionId ?? undefined,
            experiment_id: pageContext['experimentId'] as string | undefined,
            context: pageContext,
          },
          {
            onMessage: appendToStreamingMessage,
            onError: handleStreamError,
            onDone: finalizeStreamingMessage,
            onStatus: handleStatus,
            onSessionId: handleSessionId,
            onInterrupted: handleInterrupted,
          },
        );
        eventSourceRef.current = result.eventSource;
      } catch (err) {
        handleStreamError(err instanceof Error ? err.message : 'Failed to start chat');
      }
    },
    [
      sessionId,
      getPageContext,
      appendToStreamingMessage,
      handleStreamError,
      finalizeStreamingMessage,
      handleStatus,
      handleSessionId,
      handleInterrupted,
    ],
  );

  const handleSendMessage = useCallback(
    async (message: string) => {
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
      const pageContext = getPageContext();
      const result = await sendMessageStream(
        {
          session_id: sessionId,
          message,
          experiment_id: pageContext['experimentId'] as string | undefined,
          context: pageContext,
        },
        {
          onMessage: appendToStreamingMessage,
          onError: handleStreamError,
          onDone: finalizeStreamingMessage,
          onStatus: handleStatus,
          onSessionId: handleSessionId,
          onInterrupted: handleInterrupted,
        },
      );
      eventSourceRef.current = result.eventSource;
    },
    [
      sessionId,
      getPageContext,
      startChat,
      appendToStreamingMessage,
      handleStreamError,
      finalizeStreamingMessage,
      handleStatus,
      handleSessionId,
      handleInterrupted,
    ],
  );

  const handleCancelSession = useCallback(() => {
    if (!sessionId || !isStreaming) return;

    // Close EventSource immediately to stop receiving data
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    // Send cancel request to backend
    cancelSessionApi(sessionId).catch((err) => {
      console.error('Failed to cancel session:', err);
    });

    // Mark the current streaming message as interrupted
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1];
      if (lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming) {
        return [...prev.slice(0, -1), { ...lastMessage, isStreaming: false, isInterrupted: true }];
      }
      return prev;
    });

    setIsStreaming(false);
    setCurrentStatus(null);
    streamingMessageRef.current = '';
  }, [sessionId, isStreaming]);

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
    cancelSession: handleCancelSession,
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
  cancelSession: () => {},
};

/**
 * Hook to access the Assistant context.
 */
export const useAssistant = (): AssistantAgentContextType => {
  const context = useContext(AssistantReactContext);
  return context ?? disabledAssistantContext;
};
