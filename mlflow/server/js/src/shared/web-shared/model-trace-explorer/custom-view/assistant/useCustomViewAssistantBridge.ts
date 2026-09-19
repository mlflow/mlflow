import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { buildAgentDataSnapshot, type AgentTraceData } from '../agent/buildAgentPrompt';
import { toCustomViewApplyTarget, type CustomView } from '../customViewDefinition';
import { useCustomViewAssistantConnector, type OpenCustomViewAssistantOptions } from './CustomViewAssistantConnector';
import { registerCustomViewAuthoringContext } from './customViewAuthoringContext';
import {
  CustomViewValidationError,
  registerCustomViewSpecApplier,
  type RenderCustomViewSpec,
} from './customViewSpecApplier';

export type CustomViewAssistantBridge = {
  isAvailable: boolean;
  openAssistant?: (prompt?: string, options?: OpenCustomViewAssistantOptions) => void;
  isStreaming: boolean;
  isPending: boolean;
  applyError?: string;
  // Clears a leftover apply error, e.g. when the user retries a build so the
  // building skeleton isn't immediately suppressed by the previous failure.
  clearApplyError: () => void;
};

/**
 * Bridges the Custom View host to the host application's agent (MLflow
 * Assistant). It:
 *
 * 1. publishes the current authoring context (this trace's snapshot + the
 *    active view's template) to a module-level store so the assistant's
 *    context plugin can include it in the agent prompt;
 * 2. registers an applier so native tool calls and terminal structured responses
 *    can hand the agent-produced spec back to this host's `onSpec` (validate + render);
 * 3. exposes the connector's chat opener + streaming state.
 *
 * The agent itself (tool/skill registration, panel opening) is wired at a
 * higher layer that web-shared cannot import; this hook only depends on the
 * injected connector and the two module-level registries.
 */
export const useCustomViewAssistantBridge = ({
  data,
  activeView,
  onSpec,
  enabled = true,
}: {
  data: AgentTraceData;
  // The view being authored. Both the template the agent edits and the target a
  // resulting spec applies to are derived from this single object, so they can
  // never disagree about which view a turn is about.
  activeView?: CustomView;
  onSpec: (spec: RenderCustomViewSpec) => Promise<void> | void;
  enabled?: boolean;
}): CustomViewAssistantBridge => {
  const connector = useCustomViewAssistantConnector();
  const [applyError, setApplyError] = useState<string | undefined>(undefined);

  // A stable id for this host instance. The render_custom_view tool captures the
  // session that is active when it starts and scopes its wait to it, so a
  // different Custom View host mounting mid-call can't receive this host's spec.
  const sessionIdRef = useRef(`cv-session-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`);

  // The snapshot is recomputed only when the trace data changes; the applier
  // reads the latest onSpec via a ref so registration stays stable.
  const traceSample = useMemo(() => buildAgentDataSnapshot(data), [data]);

  const onSpecRef = useRef(onSpec);
  useEffect(() => {
    onSpecRef.current = onSpec;
  }, [onSpec]);

  // Publish the authoring context for the assistant's context plugin to read.
  useEffect(() => {
    if (!enabled) {
      return;
    }
    return registerCustomViewAuthoringContext({
      currentTemplate: activeView?.template,
      traceSample,
      applyTarget: activeView ? toCustomViewApplyTarget(activeView) : undefined,
    });
  }, [activeView, enabled, traceSample]);

  // Register the applier the render_custom_view tool calls. Wraps onSpec so a
  // thrown validation/render error becomes a structured failure the agent can
  // read and the user can see.
  useEffect(() => {
    if (!enabled) {
      return;
    }
    return registerCustomViewSpecApplier(sessionIdRef.current, async (spec) => {
      try {
        await onSpecRef.current(spec);
        setApplyError(undefined);
        return { ok: true };
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to render the custom view.';
        setApplyError(message);
        return { ok: false, error: message, retryable: error instanceof CustomViewValidationError };
      }
    });
  }, [enabled]);

  const openAssistant = useMemo(() => {
    if (!enabled) {
      return undefined;
    }
    return connector.openAssistant;
  }, [connector, enabled]);

  const clearApplyError = useCallback(() => setApplyError(undefined), []);

  return {
    isAvailable: Boolean(openAssistant),
    openAssistant,
    isStreaming: enabled && Boolean(connector.isStreaming),
    isPending: enabled && Boolean(connector.isPending),
    applyError: enabled ? applyError : undefined,
    clearApplyError,
  };
};
