/**
 * React Context for Assistant Agent.
 * Provides Assistant functionality accessible from anywhere in MLflow.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';

import type {
  AssistantAgentContextType,
  AssistantConfig,
  AssistantPart,
  ChatMessage,
  PermissionRequest,
  ToolUseInfo,
  ToolResultInfo,
  TokenUsage,
} from './types';
import {
  cancelSession as cancelSessionApi,
  sendMessageStream,
  getConfig,
  resumeStream,
  type SendMessageStreamCallbacks,
  type SendMessageStreamResult,
} from './AssistantService';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import { useAssistantPageContextActions } from './AssistantPageContext';
import { GatewayApi } from '../gateway/api';
import { GATEWAY_PROVIDER_ID } from './constants';

const AssistantReactContext = createContext<AssistantAgentContextType | null>(null);

// Cap the persisted transcript by JSON string length (UTF-16 code units — what localStorage counts),
// keeping it well under the ~5 MB localStorage limit.
const MAX_PERSISTED_CHARS = 1_500_000;

const CHAT_STORAGE_KEY_BASE = 'mlflow.assistant.chat';
const CHAT_STORAGE_VERSION = 1;
export const CHAT_STORAGE_KEY = buildStorageKey(CHAT_STORAGE_KEY_BASE, CHAT_STORAGE_VERSION);

const EMPTY_TOKEN_USAGE: TokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0, costUsd: null };

interface PersistedChat {
  messages: ChatMessage[];
  tokenUsage: TokenUsage;
}

/** `timestamp` round-trips through JSON as a string; restore it to a Date on load. */
export const reviveMessages = (messages: ChatMessage[]): ChatMessage[] =>
  messages.map((m) => ({ ...m, timestamp: new Date(m.timestamp) }));

/** Shrink a transcript to fit storage by dropping the oldest messages under a string-length budget. */
export const trimForStorage = (messages: ChatMessage[], maxChars: number = MAX_PERSISTED_CHARS): ChatMessage[] => {
  // Drop the oldest message until under budget, but never drop the last one.
  const lengths = messages.map((msg) => JSON.stringify(msg).length);
  let size = lengths.reduce((acc, len) => acc + len, 0); // best-effort; ignores separators
  let start = 0;
  while (start < messages.length - 1 && size > maxChars) {
    size -= lengths[start];
    start += 1;
  }
  return start === 0 ? messages : messages.slice(start);
};

/**
 * Wrap every stream callback so it no-ops once the originating send is stale (the user reset or
 * cancelled while the POST was still in flight). Guards the whole object generically rather than
 * each callback by hand, so callbacks added later are covered automatically.
 */
const withGuard = (isCurrent: () => boolean, callbacks: SendMessageStreamCallbacks): SendMessageStreamCallbacks =>
  Object.fromEntries(
    Object.entries(callbacks).map(([key, fn]) => [
      key,
      typeof fn === 'function'
        ? (...args: unknown[]) => {
            if (isCurrent()) {
              fn(...args);
            }
          }
        : fn,
    ]),
    // Object.fromEntries widens to { [k: string]: ... }; the shape is unchanged so the cast is safe.
  ) as SendMessageStreamCallbacks;

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

async function resolveSetupComplete(config: AssistantConfig): Promise<boolean> {
  const selectedProvider = Object.entries(config.providers ?? {}).find(
    ([, providerConfig]) => providerConfig.selected === true,
  );
  if (!selectedProvider) return false;

  const [providerId, providerConfig] = selectedProvider;
  if (providerId !== GATEWAY_PROVIDER_ID) {
    return true;
  }
  // The endpoint must be the same as the model name
  const { endpoints } = await GatewayApi.listEndpoints();
  return endpoints.some((endpoint) => endpoint.name === providerConfig.model);
}

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

export const AssistantProvider = ({ children }: { children: ReactNode }) => {
  // Detect if server is local - memoized since hostname doesn't change
  const isLocalServer = useMemo(() => checkIsLocalServer(), []);

  // Panel state - persisted to localStorage
  const [isPanelOpen, setIsPanelOpen] = useLocalStorage({
    key: 'mlflow.assistant.panelOpen',
    version: 1,
    initialValue: false,
  });

  // Conversation - persisted to localStorage so it survives reloads as a single conversation.
  const [persistedChat, setPersistedChat] = useLocalStorage<PersistedChat>({
    key: CHAT_STORAGE_KEY_BASE,
    version: CHAT_STORAGE_VERSION,
    initialValue: { messages: [], tokenUsage: EMPTY_TOKEN_USAGE },
  });

  // Chat state - messages/tokenUsage seeded once from the persisted conversation on first mount.
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>(() => reviveMessages(persistedChat.messages));
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentStatus, setCurrentStatus] = useState<string | null>(null);
  const [activeTools, setActiveTools] = useState<ToolUseInfo[]>([]);
  const [pendingPermission, setPendingPermission] = useState<PermissionRequest | null>(null);
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);
  const [tokenUsage, setTokenUsage] = useState<TokenUsage>(persistedChat.tokenUsage);

  // Setup state
  const [setupComplete, setSetupComplete] = useState(false);
  const [isLoadingConfig, setIsLoadingConfig] = useState(true);
  const [remoteAccessAllowed, setRemoteAccessAllowed] = useState(false);
  const canUseAssistant = isLocalServer || remoteAccessAllowed;

  // Use ref to track current streaming message
  const streamingMessageRef = useRef<string>('');

  // NB: Using the actions hook to avoid re-rendering the component when the context changes.
  const { getContext: getPageContext } = useAssistantPageContextActions();

  // Use ref to track active EventSource for cancellation
  const eventSourceRef = useRef<EventSource | null>(null);

  // Token identifying the in-flight send; reset/cancel invalidates it so a late POST's
  // guarded callbacks no-op and its stream is closed instead of leaking into new state.
  const activeRequestRef = useRef<symbol | null>(null);

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

  // Commit the in-flight streaming assistant message: flush any buffered text into an open
  // text part, mirror `content`, and mark it no longer streaming. Optionally append a
  // trailing text part (e.g. an error) or merge extra flags (e.g. isInterrupted). Shared
  // by the done / error / interrupt terminal paths so the message-finalize logic lives once.
  const commitStreamingMessage = useCallback((options: { appendText?: string; extra?: Partial<ChatMessage> } = {}) => {
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1];
      if (!(lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming)) {
        return prev;
      }
      const withBufferedText = streamingMessageRef.current
        ? setOpenTextPart(lastMessage.parts ?? [], streamingMessageRef.current)
        : (lastMessage.parts ?? []);
      const parts: AssistantPart[] = options.appendText
        ? [...withBufferedText, { type: 'text', text: options.appendText }]
        : withBufferedText;
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
    commitStreamingMessage();
    eventSourceRef.current = null;
    setIsStreaming(false);
    setCurrentStatus(null);
    setActiveTools([]);
    setPendingPermission(null);
  }, [commitStreamingMessage]);

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

  const handlePermissionRequest = useCallback((request: PermissionRequest) => {
    setPendingPermission(request);
  }, []);

  // Setup actions
  const refreshConfig = useCallback(async () => {
    setIsLoadingConfig(true);
    try {
      const config = await getConfig();
      const isComplete = await resolveSetupComplete(config);
      setSetupComplete(isComplete);
      setRemoteAccessAllowed(config.remote_access_allowed ?? false);
    } catch {
      // On error, assume setup is not complete
      setSetupComplete(false);
      setRemoteAccessAllowed(false);
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
      // Invalidate any in-flight send so any POST cleans up the stream on unmount
      activeRequestRef.current = null;
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
    setPersistedChat({ messages: trimForStorage(messages), tokenUsage });
  }, [isStreaming, messages, tokenUsage, setPersistedChat]);

  const handleStreamError = useCallback(
    (errorMsg: string) => {
      setError(errorMsg);
      setIsStreaming(false);
      setCurrentStatus(null);
      eventSourceRef.current = null;
      setActiveTools([]);
      setPendingPermission(null);
      // Keep any tool calls / text produced so far and append the error as a text
      // part (a styled error callout is a planned follow-up).
      commitStreamingMessage({ appendText: `Error: ${errorMsg}` });
    },
    [commitStreamingMessage],
  );

  const handleInterrupted = useCallback(() => {
    setIsStreaming(false);
    setCurrentStatus(null);
    setActiveTools([]);
    setPendingPermission(null);
    eventSourceRef.current = null;
    if (rafPendingRef.current !== null) {
      cancelAnimationFrame(rafPendingRef.current);
      rafPendingRef.current = null;
    }
    // Keep whatever text/tool parts streamed before the interrupt.
    commitStreamingMessage({ extra: { isInterrupted: true } });
  }, [commitStreamingMessage]);

  // Shared SSE callback wiring for startChat, handleSendMessage, respondToPermission and
  // regenerate. Each call site wraps this in `withGuard(isCurrent, streamCallbacks)` so a
  // superseded send's callbacks no-op.
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
      onPermissionRequest: handlePermissionRequest,
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
      handlePermissionRequest,
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
    // Drop any queued prompt — closing the panel is an abandon, so a stale seed shouldn't
    // inject into an unrelated chat opened later.
    setPendingPrompt(null);
  }, [setIsPanelOpen]);

  const prefillPrompt = useCallback((prompt: string) => setPendingPrompt(prompt), []);
  const clearPendingPrompt = useCallback(() => setPendingPrompt(null), []);

  const reset = useCallback(() => {
    // Invalidate any in-flight send still awaiting its POST: its captured token no longer matches,
    // so its guarded callbacks no-op and its EventSource is closed when the await resolves.
    activeRequestRef.current = null;
    // Tear down any active stream so its callbacks can't leak into the reset state
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (rafPendingRef.current !== null) {
      cancelAnimationFrame(rafPendingRef.current);
      rafPendingRef.current = null;
    }
    setSessionId(null);
    setMessages([]);
    setIsStreaming(false);
    setError(null);
    setCurrentStatus(null);
    setActiveTools([]);
    setTokenUsage({ promptTokens: 0, completionTokens: 0, totalTokens: 0, costUsd: null });
    streamingMessageRef.current = '';
    setPendingPermission(null);
  }, []);

  // Begin a new in-flight send: stamp a fresh token in closure,
  // return a checker for whether this send is
  // still the active one (i.e. not superseded by a reset/cancel that ran during its POST).
  const beginRequest = useCallback(() => {
    const token = Symbol();
    activeRequestRef.current = token;
    return () => activeRequestRef.current === token;
  }, []);

  // Store the resolved stream if its send is still current, otherwise close the orphan. Returns
  // whether it was attached
  const attachStreamIfCurrent = useCallback((isCurrent: () => boolean, result: SendMessageStreamResult): boolean => {
    if (!isCurrent()) {
      result.eventSource?.close();
      return false;
    }
    eventSourceRef.current = result.eventSource;
    return true;
  }, []);

  const startChat = useCallback(
    async (prompt?: string) => {
      const isCurrent = beginRequest();

      setError(null);
      setIsStreaming(true);
      // A new message supersedes any prompt the user was deciding on. Clearing it
      // here drops the stale Allow/Deny so it can't resume the abandoned turn; the
      // backend closes the orphaned tool call out as cancelled.
      setPendingPermission(null);

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
          withGuard(isCurrent, streamCallbacks),
        );
        if (!attachStreamIfCurrent(isCurrent, result)) {
          return;
        }
      } catch (err) {
        if (!isCurrent()) {
          return;
        }
        handleStreamError(err instanceof Error ? err.message : 'Failed to start chat');
      }
    },
    [sessionId, beginRequest, attachStreamIfCurrent, getPageContext, streamCallbacks, handleStreamError],
  );

  const respondToPermission = useCallback(
    (allow: boolean) => {
      if (!pendingPermission) {
        return;
      }
      // Target the request's originating session, not the current one, so a
      // session change while the prompt was shown can't resolve the wrong turn.
      const { sessionId: requestSessionId, requestId } = pendingPermission;
      setPendingPermission(null);
      setError(null);
      setIsStreaming(true);

      // The paused assistant placeholder keeps streaming — no new message; the
      // resume stream continues accumulating into it until done.
      const isCurrent = beginRequest();
      resumeStream(requestSessionId, requestId, allow ? 'allow' : 'deny', withGuard(isCurrent, streamCallbacks))
        .then((result) => {
          attachStreamIfCurrent(isCurrent, result);
        })
        .catch((err) => {
          if (isCurrent()) {
            handleStreamError(err instanceof Error ? err.message : 'Failed to resume');
          }
        });
    },
    [pendingPermission, beginRequest, attachStreamIfCurrent, streamCallbacks, handleStreamError],
  );

  const handleSendMessage = useCallback(
    async (message: string) => {
      if (!sessionId) {
        startChat(message);
        return;
      }

      const isCurrent = beginRequest();

      setError(null);
      setIsStreaming(true);
      setPendingPermission(null);

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
        withGuard(isCurrent, streamCallbacks),
      );
      attachStreamIfCurrent(isCurrent, result);
    },
    [sessionId, startChat, beginRequest, attachStreamIfCurrent, getPageContext, streamCallbacks],
  );

  const handleCancelSession = useCallback(() => {
    if (!sessionId || !isStreaming) return;

    // Invalidate any in-flight send so a late POST can't reopen a stream after cancel
    activeRequestRef.current = null;

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
    setPendingPermission(null);
    streamingMessageRef.current = '';
  }, [sessionId, isStreaming]);

  const regenerateLastMessage = useCallback(async () => {
    // Prevent regeneration while already streaming
    if (isStreaming) {
      return;
    }

    // Find the last user message from current state
    const lastUserMessageIndex = messages.findLastIndex((msg) => msg.role === 'user');
    if (lastUserMessageIndex === -1) {
      return; // No user message to regenerate from
    }

    const isCurrent = beginRequest();

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
    const result = await sendMessageStream(
      {
        session_id: sessionId ?? undefined,
        message: userMessageContent,
        experiment_id: pageContext['experimentId'] as string | undefined,
        context: pageContext,
      },
      withGuard(isCurrent, streamCallbacks),
    );
    attachStreamIfCurrent(isCurrent, result);
  }, [messages, sessionId, isStreaming, beginRequest, attachStreamIfCurrent, getPageContext, streamCallbacks]);

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
    pendingPrompt,
    pendingPermission,
    canUseAssistant,
    tokenUsage,
    // Actions
    openPanel,
    closePanel,
    sendMessage: handleSendMessage,
    prefillPrompt,
    clearPendingPrompt,
    regenerateLastMessage,
    reset,
    cancelSession: handleCancelSession,
    refreshConfig,
    completeSetup,
    respondToPermission,
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
  pendingPrompt: null,
  pendingPermission: null,
  canUseAssistant: false,
  tokenUsage: { promptTokens: 0, completionTokens: 0, totalTokens: 0, costUsd: null },
  openPanel: () => {},
  closePanel: () => {},
  sendMessage: () => {},
  prefillPrompt: () => {},
  clearPendingPrompt: () => {},
  regenerateLastMessage: () => {},
  reset: () => {},
  cancelSession: () => {},
  refreshConfig: () => Promise.resolve(),
  completeSetup: () => {},
  respondToPermission: () => {},
};

/**
 * Hook to access the Assistant context.
 */
export const useAssistant = (): AssistantAgentContextType => {
  const context = useContext(AssistantReactContext);
  return context ?? disabledAssistantContext;
};
