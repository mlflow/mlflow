import { useEffect, useMemo } from 'react';
import { create } from '@databricks/web-shared/zustand';
import { usePageTitle, useParams, useSearchParams } from '../common/utils/RoutingUtils';
import type { RowSelectionState } from '@tanstack/react-table';

import type { AssistantContextKey, KnownAssistantContext } from './types';

type AssistantPageContextData = Record<string, unknown>;

interface AssistantPageContextStore {
  context: AssistantPageContextData;
  setContext: (key: string, value: unknown) => void;
  removeContext: (key: string) => void;
  updateRouteContext: (
    updates: Partial<Record<'experimentId' | 'runId' | 'traceId' | 'currentPage', string | null>>,
  ) => void;
  getContext: () => AssistantPageContextData;
}

const useAssistantPageContextStore = create<AssistantPageContextStore>((set, get) => ({
  context: {},
  setContext: (key, value) => set((state) => ({ context: { ...state.context, [key]: value } })),
  removeContext: (key) =>
    set((state) => {
      const { [key]: _, ...rest } = state.context;
      return { context: rest };
    }),
  updateRouteContext: (updates) =>
    set((state) => {
      const newContext = { ...state.context };
      for (const [key, value] of Object.entries(updates)) {
        if (value !== null && value !== undefined) {
          newContext[key] = value;
        } else {
          delete newContext[key];
        }
      }
      return { context: newContext };
    }),
  getContext: () => ({ ...get().context }),
}));

/**
 * Converts a URL slug to a human-readable title.
 * Examples: "tool-calls" → "Tool Calls", "model-metrics" → "Model Metrics"
 */
const slugToTitle = (slug: string): string => {
  return slug
    .split('-')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

/**
 * Extracts sub-tab name from route params.
 * Automatically converts URL slugs to readable names - no manual mapping needed.
 */
const getSubTabFromParams = (params: Record<string, string | undefined>): string | null => {
  // Check for overview sub-tab (e.g., /overview/tool-calls)
  const overviewTab = params['overviewTab'];
  if (overviewTab) {
    return slugToTitle(overviewTab);
  }

  // Check for catch-all param (used in run page tabs, e.g., /runs/:id/model-metrics)
  const catchAll = params['*'];
  if (catchAll) {
    const tabKey = catchAll.split('/')[0];
    if (tabKey) {
      return slugToTitle(tabKey);
    }
  }

  // Check for tabName param (used in logged model details)
  const tabName = params['tabName'];
  if (tabName) {
    return slugToTitle(tabName);
  }

  return null;
};

/**
 * Component that provides route parameters as assistant context.
 * Mount this once at the app root (inside Router context).
 *
 * Uses React Router's route handles (via usePageTitle) to automatically
 * detect the current page. When new routes are added with getPageTitle handlers,
 * they are automatically included in the context.
 */
export const AssistantRouteContextProvider = () => {
  const params = useParams();
  const [searchParams] = useSearchParams();
  const { updateRouteContext } = useAssistantPageContextStore();

  // Get page title from route handle (automatically works for any route with getPageTitle)
  const pageTitle = usePageTitle();

  const experimentId = (params as Record<string, string | undefined>)['experimentId'];
  const runId = (params as Record<string, string | undefined>)['runUuid'];
  const traceId = searchParams.get('selectedEvaluationId');

  // Build the current page context, including sub-tabs if present
  const currentPage = useMemo(() => {
    if (!pageTitle) return null;

    // Check if there's a sub-tab to append
    const subTab = getSubTabFromParams(params as Record<string, string | undefined>);
    if (subTab) {
      return `${pageTitle} > ${subTab}`;
    }

    return pageTitle;
  }, [pageTitle, params]);

  useEffect(() => {
    updateRouteContext({
      experimentId: experimentId ?? null,
      runId: runId ?? null,
      traceId: traceId ?? null,
      currentPage: currentPage ?? null,
    });
  }, [experimentId, runId, traceId, currentPage, updateRouteContext]);

  return null;
};

/**
 * Hook for pages to register known context values.
 * Automatically unregisters on unmount.
 */
export const useRegisterAssistantContext = <K extends AssistantContextKey>(
  key: K,
  value: KnownAssistantContext[K] | undefined | null,
) => {
  const { setContext, removeContext } = useAssistantPageContextStore();

  useEffect(() => {
    if (value !== undefined && value !== null) {
      setContext(key, value);
    } else {
      removeContext(key);
    }
    return () => removeContext(key);
  }, [key, value, setContext, removeContext]);
};

/**
 * Hook to reactively read the current page context.
 * Subscribes to changes - component will re-render when context updates.
 */
export const useAssistantPageContext = (): AssistantPageContextData => {
  return useAssistantPageContextStore((state) => state.context);
};

/**
 * Hook to get context actions (non-reactive).
 * Does not subscribe to context changes - no re-renders.
 */
export const useAssistantPageContextActions = () => {
  const store = useAssistantPageContextStore();
  return {
    setContext: <K extends AssistantContextKey>(key: K, value: KnownAssistantContext[K]) =>
      store.setContext(key, value),
    removeContext: (key: AssistantContextKey) => store.removeContext(key),
    getContext: store.getContext,
  };
};

/**
 * Hook to register selected IDs from row selection state.
 * Handles conversion from RowSelectionState to string array.
 */
export const useRegisterSelectedIds = (key: 'selectedTraceIds' | 'selectedRunIds', rowSelection: RowSelectionState) => {
  const selectedIds = useMemo(() => {
    const ids = Object.keys(rowSelection).filter((id) => rowSelection[id]);
    return ids.length > 0 ? ids : undefined;
  }, [rowSelection]);
  useRegisterAssistantContext(key, selectedIds);
};
