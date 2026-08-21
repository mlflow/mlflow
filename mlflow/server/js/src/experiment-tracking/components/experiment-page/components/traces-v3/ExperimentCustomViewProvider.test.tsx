import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import React from 'react';

import { ExperimentCustomViewProvider } from './ExperimentCustomViewProvider';
import { useExperimentCustomViewDefinition } from './hooks/custom-view/useExperimentCustomViewDefinition';
import { useAssistant } from '../../../../../assistant/AssistantContext';
import type { ClientToolHandler } from '../../../../../assistant/clientToolHandlers';
import type { AssistantContextProvider } from '../../../../../assistant/contextProviders';
import type {
  CustomView,
  CustomViewApplyResult,
  RenderCustomViewSpec,
} from '@databricks/web-shared/model-trace-explorer/custom-view';

// `useExperimentCustomViewDefinition` / `useCanEditExperimentCustomViews` / the
// `@databricks/web-shared/model-trace-explorer/custom-view` barrel are stubbed
// (and their inputs captured) so this test is isolated to the provider's own
// wiring logic.
jest.mock('./hooks/custom-view/useExperimentCustomViewDefinition', () => ({
  useExperimentCustomViewDefinition: jest.fn(),
}));

const mockUseExperimentCustomViewDefinition = jest.mocked(useExperimentCustomViewDefinition);

let mockCanEdit = { canEdit: true, isLoading: false };
jest.mock('./hooks/custom-view/useCanEditExperimentCustomViews', () => ({
  useCanEditExperimentCustomViews: () => mockCanEdit,
}));

jest.mock('../../../../../assistant/AssistantContext', () => ({
  useAssistant: jest.fn(),
}));

let capturedToolHandler: ClientToolHandler | undefined;
const mockRegisterClientToolHandler = jest.fn((name: string, handler: ClientToolHandler) => {
  capturedToolHandler = handler;
  return () => {};
});
jest.mock('../../../../../assistant/clientToolHandlers', () => ({
  registerClientToolHandler: (name: string, handler: ClientToolHandler) => mockRegisterClientToolHandler(name, handler),
}));

let capturedContextProvider: AssistantContextProvider | undefined;
const mockRegisterAssistantContextProvider = jest.fn((key: string, provider: AssistantContextProvider) => {
  capturedContextProvider = provider;
  return () => {};
});
jest.mock('../../../../../assistant/contextProviders', () => ({
  registerAssistantContextProvider: (key: string, provider: AssistantContextProvider) =>
    mockRegisterAssistantContextProvider(key, provider),
}));

const mockWaitForCustomViewSpecApplier = jest.fn<(sessionId: string | undefined) => Promise<any>>();
const mockGetCurrentApplierSessionId = jest.fn<() => string | undefined>(() => 'session-1');
const mockGetCustomViewAuthoringContext = jest.fn<() => any>(() => null);
const mockLatchDispatchedCustomViewApplyTarget = jest.fn();
const mockBuildCustomViewAuthoringGuide = jest.fn((mode: string) => `the-${mode}-guide`);

let capturedConnectorProviderProps:
  | { connector: { openAssistant?: (...args: any[]) => void; isStreaming?: boolean; isPending?: boolean } }
  | undefined;
let capturedDefinitionProviderProps:
  | {
      views: unknown[];
      isLoaded: boolean;
      onPersistView?: unknown;
      onDeleteView?: unknown;
      canModifyPersistedViews?: boolean;
      autoSelectFirstView?: boolean;
    }
  | undefined;

jest.mock('@databricks/web-shared/model-trace-explorer/custom-view', () => ({
  CustomViewAssistantConnectorProvider: (props: any) => {
    capturedConnectorProviderProps = props;
    return <>{props.children}</>;
  },
  CustomViewDefinitionProvider: (props: any) => {
    capturedDefinitionProviderProps = props;
    return <>{props.children}</>;
  },
  RENDER_CUSTOM_VIEW_TOOL_NAME: 'render_custom_view',
  buildCustomViewAuthoringGuide: (mode: string) => mockBuildCustomViewAuthoringGuide(mode),
  getCurrentApplierSessionId: () => mockGetCurrentApplierSessionId(),
  getCustomViewAuthoringContext: () => mockGetCustomViewAuthoringContext(),
  latchDispatchedCustomViewApplyTarget: (target: unknown) => mockLatchDispatchedCustomViewApplyTarget(target),
  waitForCustomViewSpecApplier: (sessionId: string | undefined) => mockWaitForCustomViewSpecApplier(sessionId),
}));

const mockUseAssistant = jest.mocked(useAssistant);

const makeAssistant = (overrides: Record<string, unknown> = {}) => {
  const value = {
    openPanel: jest.fn(),
    requestComposerFocus: jest.fn(),
    sendMessageWhenReady: jest.fn(),
    pendingAutomaticMessage: null,
    isStreaming: false,
    activeProvider: null,
    ...overrides,
  } as unknown as ReturnType<typeof useAssistant>;
  mockUseAssistant.mockReturnValue(value);
  return value;
};

const renderProvider = (experimentId = 'exp-1') =>
  render(
    <ExperimentCustomViewProvider experimentId={experimentId}>
      <div>child content</div>
    </ExperimentCustomViewProvider>,
  );

describe('ExperimentCustomViewProvider', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    capturedToolHandler = undefined;
    capturedContextProvider = undefined;
    capturedConnectorProviderProps = undefined;
    capturedDefinitionProviderProps = undefined;
    mockUseExperimentCustomViewDefinition.mockReturnValue({
      views: [],
      isLoaded: true,
    });
    mockCanEdit = { canEdit: true, isLoading: false };
    mockGetCurrentApplierSessionId.mockReturnValue('session-1');
    mockGetCustomViewAuthoringContext.mockReturnValue(null);
  });

  it('renders its children', () => {
    makeAssistant();
    renderProvider();
    expect(screen.getByText('child content')).toBeInTheDocument();
  });

  it('passes loaded views, mutation callbacks, and permission through to the definition provider', () => {
    const persistView = jest.fn<(view: CustomView) => Promise<void>>().mockResolvedValue(undefined);
    const deleteView = jest.fn<(id: string) => Promise<void>>().mockResolvedValue(undefined);
    mockUseExperimentCustomViewDefinition.mockReturnValue({
      views: [],
      isLoaded: true,
      persistView,
      deleteView,
    });
    makeAssistant();
    mockCanEdit = { canEdit: false, isLoading: false };
    renderProvider();

    expect(capturedDefinitionProviderProps?.isLoaded).toBe(true);
    expect(capturedDefinitionProviderProps?.onPersistView).toBe(persistView);
    expect(capturedDefinitionProviderProps?.onDeleteView).toBe(deleteView);
    expect(capturedDefinitionProviderProps?.canModifyPersistedViews).toBe(false);
  });

  it('enables auto-selecting the first saved view for every experiment (not gated on demo experiments)', () => {
    makeAssistant();
    renderProvider();

    expect(capturedDefinitionProviderProps?.autoSelectFirstView).toBe(true);
  });

  describe('connector.openAssistant', () => {
    it('opens the panel and directly submits a render_custom_view directive in a fresh session', () => {
      const assistant = makeAssistant();
      renderProvider();

      capturedConnectorProviderProps?.connector.openAssistant?.('Show me the failed spans', { newSession: true });

      expect(assistant.openPanel).toHaveBeenCalledTimes(1);
      expect(assistant.sendMessageWhenReady).toHaveBeenCalledWith(expect.stringContaining('Show me the failed spans'), {
        newSession: true,
      });
      expect(assistant.sendMessageWhenReady).toHaveBeenCalledWith(expect.stringContaining('render_custom_view'), {
        newSession: true,
      });
      expect(assistant.requestComposerFocus).toHaveBeenCalledTimes(1);
    });

    it('reuses the current session (no reset) for a prompted edit without newSession', () => {
      const assistant = makeAssistant();
      renderProvider();

      capturedConnectorProviderProps?.connector.openAssistant?.('Add a chart');

      expect(assistant.openPanel).toHaveBeenCalledTimes(1);
      expect(assistant.sendMessageWhenReady).toHaveBeenCalledWith(expect.stringContaining('Add a chart'), undefined);
      expect(assistant.requestComposerFocus).toHaveBeenCalledTimes(1);
    });

    it('submits structured-output instructions for a structured local provider', () => {
      const assistant = makeAssistant({
        activeProvider: { client_tool_delivery: 'structured' },
      });
      renderProvider();

      capturedConnectorProviderProps?.connector.openAssistant?.('Show me the failed spans');

      const [message] = jest.mocked(assistant.sendMessageWhenReady).mock.calls[0];
      expect(message).toContain('structured Custom View response format');
      expect(message).not.toContain('Use the `render_custom_view` tool');
    });

    it('only opens the panel for "Edit with Assistant" when there is no prompt', () => {
      const assistant = makeAssistant();
      renderProvider();

      capturedConnectorProviderProps?.connector.openAssistant?.();

      expect(assistant.openPanel).toHaveBeenCalledTimes(1);
      expect(assistant.sendMessageWhenReady).not.toHaveBeenCalled();
      expect(assistant.requestComposerFocus).toHaveBeenCalledTimes(1);
    });

    it('exposes the assistant context streaming state on the connector', () => {
      makeAssistant({ isStreaming: true });
      renderProvider();

      expect(capturedConnectorProviderProps?.connector.isStreaming).toBe(true);
    });

    it('exposes whether an automatic Assistant message is queued', () => {
      makeAssistant({ pendingAutomaticMessage: { message: 'queued build' } });
      renderProvider();

      expect(capturedConnectorProviderProps?.connector.isPending).toBe(true);
    });
  });

  describe('render_custom_view client tool handler', () => {
    it('registers a handler for RENDER_CUSTOM_VIEW_TOOL_NAME', () => {
      makeAssistant();
      renderProvider();

      expect(mockRegisterClientToolHandler).toHaveBeenCalledWith('render_custom_view', expect.any(Function));
    });

    it('reports an error when no custom view host is open to receive the spec', async () => {
      makeAssistant();
      mockWaitForCustomViewSpecApplier.mockResolvedValue(undefined);
      renderProvider();

      const result = await capturedToolHandler!({ title: 'My view', messages: [] });

      expect(mockWaitForCustomViewSpecApplier).toHaveBeenCalledWith('session-1');
      expect(result).toEqual({
        content: 'The custom view tab is not open, so the view could not be rendered.',
        isError: true,
      });
    });

    it('applies the spec through the resolved applier and reports success', async () => {
      makeAssistant();
      const applier = jest
        .fn<(spec: RenderCustomViewSpec) => Promise<CustomViewApplyResult>>()
        .mockResolvedValue({ ok: true });
      mockWaitForCustomViewSpecApplier.mockResolvedValue(applier);
      renderProvider();

      const result = await capturedToolHandler!({ title: 'My view', messages: [{ id: 'root' }] });

      expect(applier).toHaveBeenCalledWith({ title: 'My view', messages: [{ id: 'root' }] });
      expect(result).toEqual({ content: 'The custom view was rendered successfully.' });
    });

    it('reports the applier error content when the apply fails', async () => {
      makeAssistant();
      const applier = jest
        .fn<(spec: RenderCustomViewSpec) => Promise<CustomViewApplyResult>>()
        .mockResolvedValue({ ok: false, error: 'Unknown component "Widget"', retryable: true });
      mockWaitForCustomViewSpecApplier.mockResolvedValue(applier);
      renderProvider();

      const result = await capturedToolHandler!({ title: 'My view', messages: [] });

      expect(result).toEqual({ content: 'Unknown component "Widget"', isError: true, retryable: true });
    });

    it('does not retry non-validation applier failures', async () => {
      makeAssistant();
      const applier = jest
        .fn<(spec: RenderCustomViewSpec) => Promise<CustomViewApplyResult>>()
        .mockResolvedValue({ ok: false, error: 'Custom views cannot be modified.', retryable: false });
      mockWaitForCustomViewSpecApplier.mockResolvedValue(applier);
      renderProvider();

      const result = await capturedToolHandler!({ title: 'My view', messages: [] });

      expect(result).toEqual({ content: 'Custom views cannot be modified.', isError: true, retryable: false });
    });
  });

  describe('pull-based assistant context provider', () => {
    it('does not register a context provider when Custom View delivery is unsupported', () => {
      makeAssistant({ activeProvider: { client_tool_delivery: 'unsupported' } });
      renderProvider();

      expect(mockRegisterAssistantContextProvider).not.toHaveBeenCalled();
    });

    it('does not register a context provider when there is no active provider yet', () => {
      makeAssistant({ activeProvider: null });
      renderProvider();

      expect(mockRegisterAssistantContextProvider).not.toHaveBeenCalled();
    });

    it('registers a context provider for native tool delivery', () => {
      makeAssistant({ activeProvider: { client_tool_delivery: 'tool' } });
      renderProvider();

      expect(mockRegisterAssistantContextProvider).toHaveBeenCalledWith('customTraceView', expect.any(Function));
    });

    it('returns null and latches nothing when there is no published authoring context', () => {
      makeAssistant({ activeProvider: { client_tool_delivery: 'tool' } });
      mockGetCustomViewAuthoringContext.mockReturnValue(null);
      renderProvider();

      expect(capturedContextProvider!()).toBeNull();
      expect(mockLatchDispatchedCustomViewApplyTarget).not.toHaveBeenCalled();
    });

    it('latches the apply target and returns the guide + trace sample + current template', () => {
      makeAssistant({ activeProvider: { client_tool_delivery: 'structured' } });
      const applyTarget = { id: 'v1', name: 'My view', instruction: 'do it', createdAtMs: 1 };
      mockGetCustomViewAuthoringContext.mockReturnValue({
        traceSample: { foo: 'bar' },
        currentTemplate: [{ id: 'root' }],
        applyTarget,
      });
      renderProvider();

      const result = capturedContextProvider!();

      expect(mockLatchDispatchedCustomViewApplyTarget).toHaveBeenCalledWith(applyTarget);
      expect(result).toEqual({
        guide: 'the-structured-guide',
        traceSample: { foo: 'bar' },
        currentTemplate: [{ id: 'root' }],
      });
      expect(mockBuildCustomViewAuthoringGuide).toHaveBeenCalledWith('structured');
    });
  });
});
