/**
 * React Context for Assistant Agent.
 * Provides Assistant functionality accessible from anywhere in MLflow.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';

import type { AssistantAgentContextType, ChatMessage, ToolUseInfo } from './types';
import { cancelSession as cancelSessionApi, sendMessageStream, getConfig } from './AssistantService';
import { useLocalStorage } from '../shared/web-shared/hooks/useLocalStorage';
import { useAssistantPageContextActions } from './AssistantPageContext';

const AssistantReactContext = createContext<AssistantAgentContextType | null>(null);

/**
 * Check if the server is running locally (localhost or 127.0.0.1).
 */
const checkIsLocalServer = (): boolean => {
  const hostname = window.location.hostname;
  return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1';
};

const generateMessageId = (): string => {
  return `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

export const AssistantProvider = ({ children }: { children: ReactNode }) => {
  // Detect if server is local - memoized since hostname doesn't change
  const isLocalServer = useMemo(() => checkIsLocalServer(), []);

  // Panel state - persisted to localStorage
  // Only open by default on first visit if server is local
  const [isPanelOpen, setIsPanelOpen] = useLocalStorage({
    key: 'mlflow.assistant.panelOpen',
    version: 1,
    initialValue: isLocalServer,
  });

  // Chat state
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentStatus, setCurrentStatus] = useState<string | null>(null);
  const [activeTools, setActiveTools] = useState<ToolUseInfo[]>([]);

  // Setup state
  const [setupComplete, setSetupComplete] = useState(false);
  const [isLoadingConfig, setIsLoadingConfig] = useState(true);

  // Use ref to track current streaming message
  const streamingMessageRef = useRef<string>('');

  // NB: Using the actions hook to avoid re-rendering the component when the context changes.
  const { getContext: getPageContext } = useAssistantPageContextActions();

  // Use ref to track active EventSource for cancellation
  const eventSourceRef = useRef<EventSource | null>(null);

  const appendToStreamingMessage = useCallback((text: string) => {
    // Add newline separator if there's already content (e.g. reasoning)
    if (streamingMessageRef.current && !streamingMessageRef.current.endsWith('\n') && !text.startsWith('\n')) {
      streamingMessageRef.current += '\n\n';
    }
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
    setActiveTools([]);
  }, []);

  const handleStatus = useCallback((status: string) => {
    setCurrentStatus(status);
  }, []);

  const handleSessionId = useCallback((newSessionId: string) => {
    setSessionId(newSessionId);
  }, []);

  const handleToolUse = useCallback((tools: ToolUseInfo[]) => {
    setActiveTools(tools);
  }, []);

  // Setup actions
  const refreshConfig = useCallback(async () => {
    setIsLoadingConfig(true);
    try {
      const config = await getConfig();
      // Setup is complete if claude_code provider is selected
      const isComplete = config.providers?.['claude_code']?.selected === true;
      setSetupComplete(isComplete);
    } catch {
      // On error, assume setup is not complete
      setSetupComplete(false);
    } finally {
      setIsLoadingConfig(false);
    }
  }, []);

  const completeSetup = useCallback(() => {
    // Refresh config after setup completes to update the UI
    refreshConfig();
  }, [refreshConfig]);

  // Fetch config on mount
  useEffect(() => {
    refreshConfig();
  }, [refreshConfig]);

  const handleStreamError = useCallback((errorMsg: string) => {
    setError(errorMsg);
    setIsStreaming(false);
    setCurrentStatus(null);
    eventSourceRef.current = null;
    setActiveTools([]);
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
    // Refresh config when panel opens (intentionally not awaited)
    refreshConfig();
  }, [refreshConfig, setIsPanelOpen]);

  const closePanel = useCallback(() => {
    setIsPanelOpen(false);
  }, [setIsPanelOpen]);

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
            onToolUse: handleToolUse,
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
      handleToolUse,
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
          onToolUse: handleToolUse,
          onInterrupted: handleInterrupted,
        },
      );
      eventSourceRef.current = result.eventSource;
    },
    [
      sessionId,
      startChat,
      getPageContext,
      startChat,
      appendToStreamingMessage,
      handleStreamError,
      finalizeStreamingMessage,
      handleStatus,
      handleSessionId,
      handleToolUse,
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

  const regenerateLastMessage = useCallback(() => {
    // Prevent regeneration while already streaming
    if (isStreaming) {
      return;
    }

    // Find the last user message from current state
    const lastUserMessageIndex = messages.findLastIndex((msg) => msg.role === 'user');
    if (lastUserMessageIndex === -1) {
      return; // No user message to regenerate from
    }

    const userMessageContent = messages[lastUserMessageIndex].content;

    // Set streaming state BEFORE modifying messages
    setError(null);
    setIsStreaming(true);
    streamingMessageRef.current = '';

    // Remove all messages after the last user message and add streaming placeholder
    setMessages((prev) => {
      const lastUserIdx = prev.findLastIndex((msg) => msg.role === 'user');

      if (lastUserIdx === -1) {
        return prev;
      }

      // Keep messages up to and including the last user message
      const messagesUpToLastUser = prev.slice(0, lastUserIdx + 1);

      // Add the new streaming placeholder
      return [
        ...messagesUpToLastUser,
        {
          id: generateMessageId(),
          role: 'assistant' as const,
          content: '',
          timestamp: new Date(),
          isStreaming: true,
        },
      ];
    });

    // Re-send the last user message
    const pageContext = getPageContext();
    sendMessageStream(
      {
        session_id: sessionId ?? undefined,
        message: userMessageContent,
        experiment_id: pageContext['experimentId'] as string | undefined,
        context: pageContext,
      },
      {
        onMessage: appendToStreamingMessage,
        onError: handleStreamError,
        onDone: finalizeStreamingMessage,
        onStatus: handleStatus,
        onSessionId: handleSessionId,
        onToolUse: handleToolUse,
        onInterrupted: handleInterrupted,
      },
    );
  }, [
    messages,
    sessionId,
    isStreaming,
    getPageContext,
    appendToStreamingMessage,
    handleStreamError,
    finalizeStreamingMessage,
    handleStatus,
    handleSessionId,
  ]);

  const value: AssistantAgentContextType = {
    // State
    isPanelOpen,
    sessionId,
    messages,
    isStreaming,
    error,
    currentStatus,
    activeTools,
    setupComplete,
    isLoadingConfig,
    isLocalServer,
    // Actions
    openPanel,
    closePanel,
    sendMessage: handleSendMessage,
    regenerateLastMessage,
    reset,
    cancelSession: handleCancelSession,
    refreshConfig,
    completeSetup,
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
  activeTools: [],
  setupComplete: false,
  isLoadingConfig: false,
  isLocalServer: false,
  openPanel: () => {},
  closePanel: () => {},
  sendMessage: () => {},
  regenerateLastMessage: () => {},
  reset: () => {},
  cancelSession: () => {},
  refreshConfig: () => Promise.resolve(),
  completeSetup: () => {},
};

/**
 * Hook to access the Assistant context.
 */
export const useAssistant = (): AssistantAgentContextType => {
  const context = useContext(AssistantReactContext);
  return context ?? disabledAssistantContext;
};
