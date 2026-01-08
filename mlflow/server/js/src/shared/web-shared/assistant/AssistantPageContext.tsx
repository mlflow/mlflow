/**
 * Assistant Page Context - Collects page-specific context for the assistant.
 *
 * Features:
 * - Ref-based storage to avoid re-renders
 * - URL-based auto-extraction for experimentId and traceId
 * - Type-safe registration API
 * - Subscription API for reactive components
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useParams, useSearchParams } from '../genai-traces-table/utils/RoutingUtils';

import type { KnownAssistantContext } from './types';

// Use Record<string, unknown> for internal storage to avoid type intersection issues
type AssistantPageContextData = Record<string, unknown>;

interface AssistantPageContextValue {
  /** Get current context snapshot (for sending messages) */
  getContext: () => AssistantPageContextData;
  /** Register a known context value (type-safe) */
  registerContext: <K extends keyof KnownAssistantContext>(key: K, value: KnownAssistantContext[K]) => void;
  /** Register an extension context value (for custom keys) */
  registerExtensionContext: (key: string, value: unknown) => void;
  /** Unregister a context value */
  unregisterContext: (key: string) => void;
  /** Subscribe to context changes (for reactive components) */
  subscribe: (callback: (context: AssistantPageContextData) => void) => () => void;
}

const AssistantPageContext = createContext<AssistantPageContextValue | null>(null);

/**
 * Provider that collects context from all pages.
 * Should wrap the entire app, outside of AssistantProvider.
 */
export const AssistantPageContextProvider = ({ children }: { children: ReactNode }) => {
  // Ref-based storage - no re-renders on context changes
  const contextRef = useRef<AssistantPageContextData>({});
  const registeredKeysRef = useRef<Set<string>>(new Set());
  const subscribersRef = useRef<Set<(context: AssistantPageContextData) => void>>(new Set());

  // Extract URL-based context
  const params = useParams();
  const [searchParams] = useSearchParams();

  // Notify subscribers of context changes
  const notifySubscribers = useCallback(() => {
    const snapshot = { ...contextRef.current };
    subscribersRef.current.forEach((callback) => callback(snapshot));
  }, []);

  // Auto-extract experimentId from route params
  const experimentIdFromParams = (params as Record<string, string | undefined>)['experimentId'];
  useEffect(() => {
    if (experimentIdFromParams) {
      contextRef.current['experimentId'] = experimentIdFromParams;
    } else {
      delete contextRef.current['experimentId'];
    }
    notifySubscribers();
  }, [experimentIdFromParams, notifySubscribers]);

  // Auto-extract traceId from query params (selectedEvaluationId)
  useEffect(() => {
    const traceId = searchParams.get('selectedEvaluationId');
    // Only set from URL if not explicitly registered by a component
    if (traceId && !registeredKeysRef.current.has('traceId')) {
      contextRef.current['traceId'] = traceId;
      notifySubscribers();
    }
  }, [searchParams, notifySubscribers]);

  const registerContext = useCallback(
    <K extends keyof KnownAssistantContext>(key: K, value: KnownAssistantContext[K]) => {
      if (process.env['NODE_ENV'] === 'development' && registeredKeysRef.current.has(key as string)) {
        console.warn(`[AssistantContext] Key "${String(key)}" already registered. Overwriting.`);
      }
      registeredKeysRef.current.add(key as string);
      contextRef.current[key as string] = value;
      notifySubscribers();
    },
    [notifySubscribers],
  );

  const registerExtensionContext = useCallback(
    (key: string, value: unknown) => {
      const knownKeys: string[] = ['experimentId', 'traceId', 'selectedTraceIds'];
      if (process.env['NODE_ENV'] === 'development' && knownKeys.includes(key)) {
        console.warn(`[AssistantContext] Use registerContext for known key "${key}"`);
      }
      contextRef.current[key] = value;
      notifySubscribers();
    },
    [notifySubscribers],
  );

  const unregisterContext = useCallback(
    (key: string) => {
      registeredKeysRef.current.delete(key);
      delete contextRef.current[key];
      notifySubscribers();
    },
    [notifySubscribers],
  );

  const getContext = useCallback(() => ({ ...contextRef.current }), []);

  const subscribe = useCallback((callback: (context: AssistantPageContextData) => void) => {
    subscribersRef.current.add(callback);
    // Immediately call with current context
    callback({ ...contextRef.current });
    return () => {
      subscribersRef.current.delete(callback);
    };
  }, []);

  const value = useMemo<AssistantPageContextValue>(
    () => ({
      getContext,
      registerContext,
      registerExtensionContext,
      unregisterContext,
      subscribe,
    }),
    [getContext, registerContext, registerExtensionContext, unregisterContext, subscribe],
  );

  return <AssistantPageContext.Provider value={value}>{children}</AssistantPageContext.Provider>;
};

/**
 * Hook for pages to register known context values.
 * Type-safe for known keys. Automatically unregisters on unmount.
 *
 * @param key - The context key (e.g., 'experimentId', 'traceId')
 * @param value - The context value (undefined/null values are unregistered)
 */
export const useRegisterAssistantContext = <K extends keyof KnownAssistantContext>(
  key: K,
  value: KnownAssistantContext[K] | undefined | null,
) => {
  const ctx = useContext(AssistantPageContext);

  useEffect(() => {
    if (!ctx) return;

    if (value !== undefined && value !== null) {
      ctx.registerContext(key, value);
    } else {
      ctx.unregisterContext(key as string);
    }

    return () => ctx.unregisterContext(key as string);
  }, [ctx, key, value]);
};

/**
 * Hook to get a stable function that returns current page context.
 * Use this for non-reactive access (e.g., reading at send time).
 */
export const useAssistantPageContextGetter = (): (() => AssistantPageContextData) => {
  const ctx = useContext(AssistantPageContext);
  return ctx?.getContext ?? (() => ({}));
};

/**
 * Hook to reactively read the current page context.
 * Use this when you need to display or react to context changes.
 */
export const useAssistantPageContextValue = (): AssistantPageContextData => {
  const ctx = useContext(AssistantPageContext);
  const [contextValue, setContextValue] = useState<AssistantPageContextData>({});

  useEffect(() => {
    if (!ctx) return;
    return ctx.subscribe(setContextValue);
  }, [ctx]);

  return contextValue;
};
