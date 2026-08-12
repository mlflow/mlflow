import { useEffect, useMemo, type ReactNode } from 'react';

import {
  CustomViewAssistantConnectorProvider,
  CustomViewDefinitionProvider,
  RENDER_CUSTOM_VIEW_TOOL_NAME,
  buildCustomViewAuthoringGuide,
  getCurrentApplierSessionId,
  getCustomViewAuthoringContext,
  latchDispatchedCustomViewApplyTarget,
  waitForCustomViewSpecApplier,
  type CustomViewAssistantConnector,
  type OpenCustomViewAssistantOptions,
} from '@databricks/web-shared/model-trace-explorer/custom-view';

import { useExperimentCustomViewDefinition } from './hooks/custom-view/useExperimentCustomViewDefinition';
import { useCanEditExperimentCustomViews } from './hooks/custom-view/useCanEditExperimentCustomViews';
import { useAssistant } from '../../../../../assistant/AssistantContext';
import { registerClientToolHandler } from '../../../../../assistant/clientToolHandlers';
import { registerAssistantContextProvider } from '../../../../../assistant/contextProviders';

// The empty-state prompt box hands us the user's natural-language request. This
// prompt is submitted ONLY for the initial build (handleSubmitPrompt); the edit
// flow (handleEditWithAssistant) sends nothing and just opens the panel.
const buildRenderCustomViewPrompt = (request: string): string =>
  [`Build my custom trace view: "${request}".`, '', 'Use the `render_custom_view` tool to build this view.'].join('\n');

/**
 * Wires the trace explorer's Custom View feature to the experiment's persistence
 * (one tag per view) and to MLflow Assistant. Mount this high in the traces page
 * (above the trace drawer) so saved views + in-flight edits survive opening/closing
 * the drawer and cycling between traces.
 */
export const ExperimentCustomViewProvider = ({
  experimentId,
  children,
}: {
  experimentId?: string;
  children: ReactNode;
}) => {
  const { views, isLoaded, persistView } = useExperimentCustomViewDefinition(experimentId);
  const { canEdit: canModifyPersistedViews } = useCanEditExperimentCustomViews(experimentId);
  const { openPanel, sendMessage, isStreaming, activeProvider } = useAssistant();

  const connector = useMemo<CustomViewAssistantConnector>(
    () => ({
      openAssistant: (prompt?: string, options?: OpenCustomViewAssistantOptions) => {
        const instruction = prompt?.trim();
        openPanel();
        // "Edit with assistant" (no instruction): just open/focus the panel. The
        // custom-view authoring context is already published while the tab is open
        // (see useCustomViewAssistantBridge), so the user can describe the change
        // with full context instead of us auto-triggering an unprompted rebuild.
        if (instruction) {
          // Submit the build directive immediately. A brand-new build requests a
          // fresh Assistant thread atomically so it cannot inherit an unrelated
          // conversation; prompted edits can continue the current thread.
          sendMessage(buildRenderCustomViewPrompt(instruction), options);
        }
      },
      isStreaming,
    }),
    [openPanel, sendMessage, isStreaming],
  );

  // Tier 1 (Gateway/Ollama): the assistant backend calls a real `render_custom_view`
  // client tool and pauses the turn; this handler runs it by handing the spec to
  // whichever Custom View host is currently registered (the applier), then reports
  // the result back to resume the stream. Scoped to the session that was active
  // when the call started, so a different tab remounting mid-call can't steal it.
  useEffect(() => {
    return registerClientToolHandler(RENDER_CUSTOM_VIEW_TOOL_NAME, async (toolInput) => {
      const applier = await waitForCustomViewSpecApplier(getCurrentApplierSessionId());
      if (!applier) {
        return { content: 'The custom view tab is not open, so the view could not be rendered.', isError: true };
      }
      const result = await applier({ title: toolInput['title'], messages: toolInput['messages'] });
      return result.ok
        ? { content: 'The custom view was rendered successfully.' }
        : { content: result.error, isError: true };
    });
  }, []);

  // Publishes the custom-view guide + this trace's live authoring context (see
  // useCustomViewAssistantBridge) into the assistant's page context. Pull-based (read
  // at message-send time, see contextProviders.ts) rather than pushed into React
  // state, since whether to publish at all depends on which provider is about to
  // serve the turn. Only providers that support a real client-tool call
  // (`render_custom_view`) can act on this guide today — CLI providers (Claude
  // Code, Codex) have no mid-stream client-tool channel without MCP plumbing, so
  // publishing the guide for them would be a no-op at best. Support for those is
  // tracked as a follow-up (a fenced-block convention), not yet implemented.
  useEffect(() => {
    if (!activeProvider?.supports_client_tools) {
      return undefined;
    }
    return registerAssistantContextProvider('customTraceView', () => {
      const ctx = getCustomViewAuthoringContext();
      if (!ctx) {
        return null;
      }
      // Records which view this turn's template came from, so a spec the agent
      // returns lands on the right view even if the user switches views mid-turn.
      latchDispatchedCustomViewApplyTarget(ctx.applyTarget);
      return {
        guide: buildCustomViewAuthoringGuide(),
        traceSample: ctx.traceSample,
        currentTemplate: ctx.currentTemplate,
      };
    });
  }, [activeProvider]);

  return (
    <CustomViewAssistantConnectorProvider connector={connector}>
      <CustomViewDefinitionProvider
        views={views}
        isLoaded={isLoaded}
        onPersistView={persistView}
        canModifyPersistedViews={canModifyPersistedViews}
      >
        {children}
      </CustomViewDefinitionProvider>
    </CustomViewAssistantConnectorProvider>
  );
};
