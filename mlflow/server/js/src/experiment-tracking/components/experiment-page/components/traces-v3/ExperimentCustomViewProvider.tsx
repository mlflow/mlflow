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
// The structured Custom View response format is defined and validated by the backend
// in mlflow/assistant/custom_view.py; this prompt only selects that delivery contract.
const buildRenderCustomViewPrompt = (request: string, delivery: 'tool' | 'structured'): string =>
  [
    `Build my custom trace view: "${request}".`,
    '',
    delivery === 'tool'
      ? 'Use the `render_custom_view` tool to build this view.'
      : 'Return the view using the structured Custom View response format.',
  ].join('\n');

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
  const { views, isLoaded, persistView, deleteView } = useExperimentCustomViewDefinition(experimentId);
  const { canEdit: canModifyPersistedViews } = useCanEditExperimentCustomViews(experimentId);
  const {
    openPanel,
    requestComposerFocus,
    sendMessageWhenReady,
    pendingAutomaticMessage,
    isStreaming,
    activeProvider,
  } = useAssistant();
  const clientToolDelivery = activeProvider?.client_tool_delivery;

  const connector = useMemo<CustomViewAssistantConnector>(
    () => ({
      openAssistant: (prompt?: string, options?: OpenCustomViewAssistantOptions) => {
        const instruction = prompt?.trim();
        openPanel();
        requestComposerFocus();
        // "Edit with assistant" (no instruction) only opens/focuses the panel. The
        // custom-view authoring context is already published while the tab is open,
        // so the user can describe the change without an unprompted rebuild.
        if (instruction) {
          // Submit the build directive immediately. A brand-new build requests a
          // fresh Assistant thread atomically so it cannot inherit an unrelated
          // conversation; prompted edits can continue the current thread.
          const delivery = clientToolDelivery === 'structured' ? 'structured' : 'tool';
          sendMessageWhenReady(buildRenderCustomViewPrompt(instruction, delivery), options);
        }
      },
      isStreaming,
      isPending: Boolean(pendingAutomaticMessage),
    }),
    [clientToolDelivery, openPanel, requestComposerFocus, sendMessageWhenReady, pendingAutomaticMessage, isStreaming],
  );

  // The assistant backend delivers a `render_custom_view` client action. API-based
  // providers use a native tool call; local providers translate structured final
  // output into a terminal action. This handler hands the spec to whichever Custom View host is
  // registered, scoped to the session that was active when the call started so a
  // different tab remounting mid-call can't steal it.
  useEffect(() => {
    return registerClientToolHandler(RENDER_CUSTOM_VIEW_TOOL_NAME, async (toolInput) => {
      const applier = await waitForCustomViewSpecApplier(getCurrentApplierSessionId());
      if (!applier) {
        return { content: 'The custom view tab is not open, so the view could not be rendered.', isError: true };
      }
      const result = await applier({ title: toolInput['title'], messages: toolInput['messages'] });
      return result.ok
        ? { content: 'The custom view was rendered successfully.' }
        : { content: result.error, isError: true, retryable: result.retryable };
    });
  }, []);

  // Publishes the custom-view guide + this trace's live authoring context (see
  // useCustomViewAssistantBridge) into the assistant's page context. Pull-based (read
  // at message-send time, see contextProviders.ts) rather than pushed into React
  // state, since whether to publish at all depends on which provider is about to
  // serve the turn. The guide matches the provider's native-tool or structured-output
  // delivery mode so the model never receives conflicting output instructions.
  useEffect(() => {
    if (!clientToolDelivery || clientToolDelivery === 'unsupported') {
      return undefined;
    }
    const deliveryMode = clientToolDelivery === 'structured' ? 'structured' : 'tool';
    return registerAssistantContextProvider('customTraceView', () => {
      const ctx = getCustomViewAuthoringContext();
      if (!ctx) {
        return null;
      }
      // Records which view this turn's template came from, so a spec the agent
      // returns lands on the right view even if the user switches views mid-turn.
      latchDispatchedCustomViewApplyTarget(ctx.applyTarget);
      return {
        guide: buildCustomViewAuthoringGuide(deliveryMode),
        traceSample: ctx.traceSample,
        currentTemplate: ctx.currentTemplate,
      };
    });
  }, [clientToolDelivery]);

  return (
    <CustomViewAssistantConnectorProvider connector={connector}>
      <CustomViewDefinitionProvider
        views={views}
        isLoaded={isLoaded}
        onPersistView={persistView}
        onDeleteView={deleteView}
        canModifyPersistedViews={canModifyPersistedViews}
        // Every experiment with saved views defaults to its first one on load. With
        // no saved views this is a no-op: the host's "Build a custom trace view"
        // authoring prompt (create-first-view empty state) shows unchanged.
        autoSelectFirstView
      >
        {children}
      </CustomViewDefinitionProvider>
    </CustomViewAssistantConnectorProvider>
  );
};
