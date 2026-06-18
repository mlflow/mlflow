/**
 * React Context for Assistant Agent.
 * Provides Assistant functionality accessible from anywhere in MLflow.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';

import type {
  AssistantAgentContextType,
  AssistantPart,
  ChatMessage,
  ToolUseInfo,
  ToolResultInfo,
  TokenUsage,
} from './types';
import { cancelSession as cancelSessionApi, sendMessageStream, getConfig } from './AssistantService';
import { useLocalStorage, useSessionStorage } from '@databricks/web-shared/hooks';
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

/**
 * Set the current (open) text segment of an assistant turn. `text` is the full
 * segment since the last tool call, so we replace the trailing text part if there
 * is one, otherwise append a new text part (a tool call always closes the prior
 * text part, so the next text starts fresh).
 */
const setOpenTextPart = (parts: AssistantPart[], text: string): AssistantPart[] => {
  const last = parts[parts.length - 1];
  if (last?.type === 'text') {
    return [...parts.slice(0, -1), { type: 'text', text }];
  }
  return [...parts, { type: 'text', text }];
};

/** Add or update tool-call parts by `toolUseId` (they can re-stream, so upsert). */
export const upsertToolCalls = (parts: AssistantPart[], tools: ToolUseInfo[]): AssistantPart[] => {
  const next = [...parts];
  for (const tool of tools) {
    const i = next.findIndex((p) => p.type === 'toolCall' && p.toolUseId === tool.id);
    const part = { type: 'toolCall' as const, toolUseId: tool.id, name: tool.name, input: tool.input };
    if (i >= 0) {
      // Merge without clobbering an already-resolved status/result from a tool_result.
      next[i] = { ...next[i], ...part };
    } else {
      next.push({ ...part, status: 'running' });
    }
  }
  return next;
};

/** Resolve a tool call's status/result once its tool_result arrives, matched by `toolUseId`. */
export const applyToolResult = (parts: AssistantPart[], result: ToolResultInfo): AssistantPart[] =>
  parts.map((p) =>
    p.type === 'toolCall' && p.toolUseId === result.toolUseId
      ? { ...p, status: result.isError ? 'error' : 'done', result: result.content }
      : p,
  );

const partsToContent = (parts: AssistantPart[]): string =>
  parts
    .filter((p): p is Extract<AssistantPart, { type: 'text' }> => p.type === 'text')
    .map((p) => p.text)
    .join('');

const EMPTY_TOKEN_USAGE: TokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0, costUsd: null };
// Keep the persisted transcript well under the ~5 MB storage limit.
const MAX_PERSISTED_BYTES = 1_500_000;

interface PersistedChat {
  messages: ChatMessage[];
  sessionId: string | null;
  tokenUsage: TokenUsage;
}

/** `timestamp` round-trips through JSON as a string; restore it to a Date on load. */
export const reviveMessages = (messages: ChatMessage[]): ChatMessage[] =>
  messages.map((m) => ({ ...m, timestamp: new Date(m.timestamp) }));

/** Shrink a transcript to fit storage by dropping the oldest messages under a byte budget. */
export const trimForStorage = (messages: ChatMessage[], maxBytes: number = MAX_PERSISTED_BYTES): ChatMessage[] => {
  // Drop the oldest message until under budget, but never drop the last one.
  let trimmed = messages;
  while (trimmed.length > 1 && JSON.stringify(trimmed).length > maxBytes) {
    trimmed = trimmed.slice(1);
  }
  return trimmed;
};

export const AssistantProvider = ({ children }: { children: ReactNode }) => {
  // Detect if server is local - memoized since hostname doesn't change
  const isLocalServer = useMemo(() => checkIsLocalServer(), []);

  // Panel state - persisted to localStorage
  const [isPanelOpen, setIsPanelOpen] = useLocalStorage({
    key: 'mlflow.assistant.panelOpen',
    version: 1,
    initialValue: false,
  });

  // Conversation - persisted per-tab in sessionStorage so it survives reloads within a
  // tab but never collides with other tabs (unlike localStorage).
  const [persistedChat, setPersistedChat] = useSessionStorage<PersistedChat>({
    key: 'mlflow.assistant.chat',
    version: 1,
    initialValue: { messages: [], sessionId: null, tokenUsage: EMPTY_TOKEN_USAGE },
  });

  // Chat state - seeded once from this tab's persisted conversation on first mount.
  const [sessionId, setSessionId] = useState<string | null>(persistedChat.sessionId);
  const [messages, setMessages] = useState<ChatMessage[]>(() => reviveMessages(persistedChat.messages));
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentStatus, setCurrentStatus] = useState<string | null>(null);
  const [activeTools, setActiveTools] = useState<ToolUseInfo[]>([]);
  const [tokenUsage, setTokenUsage] = useState<TokenUsage>(persistedChat.tokenUsage);

  // Setup state
  const [setupComplete, setSetupComplete] = useState(false);
  const [isLoadingConfig, setIsLoadingConfig] = useState(true);

  // Use ref to track current streaming message
  const streamingMessageRef = useRef<string>('');

  // NB: Using the actions hook to avoid re-rendering the component when the context changes.
  const { getContext: getPageContext } = useAssistantPageContextActions();

  // Use ref to track active EventSource for cancellation
  const eventSourceRef = useRef<EventSource | null>(null);

  // Throttle streaming updates to avoid overwhelming React with re-renders
  const rafPendingRef = useRef<number | null>(null);

  // Apply `fn` to the last message's parts if it's the streaming assistant message,
  // keeping `content` mirrored to the text parts.
  const updateStreamingParts = useCallback((fn: (parts: AssistantPart[]) => AssistantPart[]) => {
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1];
      if (lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming) {
        const parts = fn(lastMessage.parts ?? []);
        return [...prev.slice(0, -1), { ...lastMessage, parts, content: partsToContent(parts) }];
      }
      return prev;
    });
  }, []);

  // Seal the in-flight streaming assistant message: flush any buffered text into an open
  // text part, mirror `content`, and mark it no longer streaming. Optionally append a
  // trailing text part (e.g. an error) or merge extra flags (e.g. isInterrupted). Shared
  // by the done / error / interrupt terminal paths so the message-finalize logic lives once.
  const sealStreamingMessage = useCallback((options: { appendText?: string; extra?: Partial<ChatMessage> } = {}) => {
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1];
      if (!(lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming)) {
        return prev;
      }
      const flushed = streamingMessageRef.current
        ? setOpenTextPart(lastMessage.parts ?? [], streamingMessageRef.current)
        : (lastMessage.parts ?? []);
      const parts: AssistantPart[] = options.appendText
        ? [...flushed, { type: 'text', text: options.appendText }]
        : flushed;
      return [
        ...prev.slice(0, -1),
        { ...lastMessage, parts, content: partsToContent(parts), isStreaming: false, ...options.extra },
      ];
    });
    streamingMessageRef.current = '';
  }, []);

  const flushStreamingMessage = useCallback(() => {
    rafPendingRef.current = null;
    if (!streamingMessageRef.current) {
      return;
    }
    updateStreamingParts((parts) => setOpenTextPart(parts, streamingMessageRef.current));
  }, [updateStreamingParts]);

  const appendToStreamingMessage = useCallback(
    (text: string) => {
      streamingMessageRef.current += text;
      if (rafPendingRef.current === null) {
        rafPendingRef.current = requestAnimationFrame(flushStreamingMessage);
      }
    },
    [flushStreamingMessage],
  );

  const finalizeStreamingMessage = useCallback(() => {
    // Cancel any pending RAF and do a final flush with isStreaming: false
    if (rafPendingRef.current !== null) {
      cancelAnimationFrame(rafPendingRef.current);
      rafPendingRef.current = null;
    }
    sealStreamingMessage();
    eventSourceRef.current = null;
    setIsStreaming(false);
    setCurrentStatus(null);
    setActiveTools([]);
  }, [sealStreamingMessage]);

  const handleStatus = useCallback((status: string) => {
    setCurrentStatus(status);
  }, []);

  const handleSessionId = useCallback((newSessionId: string) => {
    setSessionId(newSessionId);
  }, []);

  const handleToolUse = useCallback(
    (tools: ToolUseInfo[]) => {
      // `activeTools` drives the transient "working" indicator only.
      setActiveTools(tools);
      if (tools.length === 0) {
        return;
      }
      // Persist the calls onto the message, in order. Commit any buffered text
      // first (so the calls land after the text that preceded them) and reset the
      // buffer so subsequent text starts a new part after the tool call.
      if (rafPendingRef.current !== null) {
        cancelAnimationFrame(rafPendingRef.current);
        rafPendingRef.current = null;
      }
      updateStreamingParts((parts) => {
        const withText = streamingMessageRef.current ? setOpenTextPart(parts, streamingMessageRef.current) : parts;
        return upsertToolCalls(withText, tools);
      });
      streamingMessageRef.current = '';
    },
    [updateStreamingParts],
  );

  const handleToolResult = useCallback(
    (result: ToolResultInfo) => {
      updateStreamingParts((parts) => applyToolResult(parts, result));
    },
    [updateStreamingParts],
  );

  const handleUsage = useCallback(
    (usage: {
      prompt_tokens: number;
      completion_tokens: number;
      total_tokens: number;
      total_cost_usd?: number | null;
    }) => {
      setTokenUsage((prev) => ({
        promptTokens: prev.promptTokens + (usage.prompt_tokens ?? 0),
        completionTokens: prev.completionTokens + (usage.completion_tokens ?? 0),
        totalTokens: prev.totalTokens + (usage.total_tokens ?? 0),
        // Accumulate cost only from priced turns; stays null until the first
        // numeric estimate arrives so unpriced models render no cost at all.
        costUsd: usage.total_cost_usd == null ? prev.costUsd : (prev.costUsd ?? 0) + usage.total_cost_usd,
      }));
    },
    [],
  );

  // Setup actions
  const refreshConfig = useCallback(async () => {
    setIsLoadingConfig(true);
    try {
      const config = await getConfig();
      const isComplete = Object.values(config.providers ?? {}).some((p) => p.selected === true);
      setSetupComplete(isComplete);
    } catch {
      // On error, assume setup is not complete
      setSetupComplete(false);
    } finally {
      setIsLoadingConfig(false);
    }
  }, []);

  const completeSetup = useCallback(() => {
    setSetupComplete(true);
    refreshConfig();
  }, [refreshConfig]);

  // Fetch config on mount
  useEffect(() => {
    refreshConfig();
  }, [refreshConfig]);

  // Cancel pending RAF and close EventSource on unmount
  useEffect(() => {
    return () => {
      if (rafPendingRef.current !== null) {
        cancelAnimationFrame(rafPendingRef.current);
        rafPendingRef.current = null;
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, []);

  // Persist the conversation only once a turn has settled (never on the streaming
  // hot path, which would write to storage on every frame). `reset()` flips back to
  // the empty state here too, which clears the stored conversation.
  useEffect(() => {
    if (isStreaming) {
      return;
    }
    setPersistedChat({ messages: trimForStorage(messages), sessionId, tokenUsage });
  }, [isStreaming, messages, sessionId, tokenUsage, setPersistedChat]);

  const handleStreamError = useCallback(
    (errorMsg: string) => {
      setError(errorMsg);
      setIsStreaming(false);
      setCurrentStatus(null);
      eventSourceRef.current = null;
      setActiveTools([]);
      // Keep any tool calls / text produced so far and append the error as a text
      // part (a styled error callout is a planned follow-up).
      sealStreamingMessage({ appendText: `Error: ${errorMsg}` });
    },
    [sealStreamingMessage],
  );

  const handleInterrupted = useCallback(() => {
    setIsStreaming(false);
    setCurrentStatus(null);
    setActiveTools([]);
    eventSourceRef.current = null;
    if (rafPendingRef.current !== null) {
      cancelAnimationFrame(rafPendingRef.current);
      rafPendingRef.current = null;
    }
    // Keep whatever text/tool parts streamed before the interrupt.
    sealStreamingMessage({ extra: { isInterrupted: true } });
  }, [sealStreamingMessage]);

  // Shared SSE callback wiring for startChat and handleSendMessage.
  const streamCallbacks = useMemo(
    () => ({
      onMessage: appendToStreamingMessage,
      onError: handleStreamError,
      onDone: finalizeStreamingMessage,
      onStatus: handleStatus,
      onSessionId: handleSessionId,
      onToolUse: handleToolUse,
      onToolResult: handleToolResult,
      onInterrupted: handleInterrupted,
      onUsage: handleUsage,
    }),
    [
      appendToStreamingMessage,
      handleStreamError,
      finalizeStreamingMessage,
      handleStatus,
      handleSessionId,
      handleToolUse,
      handleToolResult,
      handleInterrupted,
      handleUsage,
    ],
  );

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
    setTokenUsage({ promptTokens: 0, completionTokens: 0, totalTokens: 0, costUsd: null });
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
          streamCallbacks,
        );
        eventSourceRef.current = result.eventSource;
      } catch (err) {
        handleStreamError(err instanceof Error ? err.message : 'Failed to start chat');
      }
    },
    [sessionId, getPageContext, streamCallbacks, handleStreamError],
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
        streamCallbacks,
      );
      eventSourceRef.current = result.eventSource;
    },
    [sessionId, startChat, getPageContext, streamCallbacks],
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
      if (err) {
        // fail silently
      }
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
    setActiveTools([]);
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
        onToolResult: handleToolResult,
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
    handleToolUse,
    handleToolResult,
    handleInterrupted,
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
    tokenUsage,
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
  tokenUsage: { promptTokens: 0, completionTokens: 0, totalTokens: 0, costUsd: null },
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
