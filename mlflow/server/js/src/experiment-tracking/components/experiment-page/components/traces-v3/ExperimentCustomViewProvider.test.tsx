import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import React from 'react';

import { ExperimentCustomViewProvider } from './ExperimentCustomViewProvider';
import { useAssistant } from '../../../../../assistant/AssistantContext';
import type { ClientToolHandler } from '../../../../../assistant/clientToolHandlers';
import type { AssistantContextProvider } from '../../../../../assistant/contextProviders';
import type {
  CustomViewApplyResult,
  RenderCustomViewSpec,
} from '@databricks/web-shared/model-trace-explorer/custom-view';

// This suite exercises ONLY the OSS-specific wiring (MLflow Assistant open/
// direct send + the Tier 1 client-tool pause/resume + the pull-based context
// provider), not the Databricks Genie/MFE RPC wiring the universe equivalent of
// this file tests — that plumbing does not exist in OSS.
// `useExperimentCustomViewDefinition` / `useCanEditExperimentCustomViews` / the
// `@databricks/web-shared/model-trace-explorer/custom-view` barrel are stubbed
// (and their inputs captured) so this test is isolated to the provider's own
// wiring logic.
jest.mock('./hooks/custom-view/useExperimentCustomViewDefinition', () => ({
  useExperimentCustomViewDefinition: jest.fn(() => ({ views: [], isLoaded: true, persistView: undefined })),
}));

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

let capturedConnectorProviderProps:
  | { connector: { openAssistant?: (...args: any[]) => void; isStreaming?: boolean; isPending?: boolean } }
  | undefined;
let capturedDefinitionProviderProps:
  | { views: unknown[]; isLoaded: boolean; onPersistView?: unknown; canModifyPersistedViews?: boolean }
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
  buildCustomViewAuthoringGuide: () => 'the-guide',
  getCurrentApplierSessionId: () => mockGetCurrentApplierSessionId(),
  getCustomViewAuthoringContext: () => mockGetCustomViewAuthoringContext(),
  latchDispatchedCustomViewApplyTarget: (target: unknown) => mockLatchDispatchedCustomViewApplyTarget(target),
  waitForCustomViewSpecApplier: (sessionId: string | undefined) => mockWaitForCustomViewSpecApplier(sessionId),
}));

const mockUseAssistant = jest.mocked(useAssistant);

const makeAssistant = (overrides: Record<string, unknown> = {}) => {
  const value = {
    openPanel: jest.fn(),
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
    mockCanEdit = { canEdit: true, isLoading: false };
    mockGetCurrentApplierSessionId.mockReturnValue('session-1');
    mockGetCustomViewAuthoringContext.mockReturnValue(null);
  });

  it('renders its children', () => {
    makeAssistant();
    renderProvider();
    expect(screen.getByText('child content')).toBeInTheDocument();
  });

  it('passes the loaded views and persist permission through to the definition provider', () => {
    makeAssistant();
    mockCanEdit = { canEdit: false, isLoading: false };
    renderProvider();

    expect(capturedDefinitionProviderProps?.isLoaded).toBe(true);
    expect(capturedDefinitionProviderProps?.canModifyPersistedViews).toBe(false);
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
    });

    it('reuses the current session (no reset) for a prompted edit without newSession', () => {
      const assistant = makeAssistant();
      renderProvider();

      capturedConnectorProviderProps?.connector.openAssistant?.('Add a chart');

      expect(assistant.openPanel).toHaveBeenCalledTimes(1);
      expect(assistant.sendMessageWhenReady).toHaveBeenCalledWith(expect.stringContaining('Add a chart'), undefined);
    });

    it('only opens the panel for "Edit with Assistant" when there is no prompt', () => {
      const assistant = makeAssistant();
      renderProvider();

      capturedConnectorProviderProps?.connector.openAssistant?.();

      expect(assistant.openPanel).toHaveBeenCalledTimes(1);
      expect(assistant.sendMessageWhenReady).not.toHaveBeenCalled();
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
        .mockResolvedValue({ ok: false, error: 'Unknown component "Widget"' });
      mockWaitForCustomViewSpecApplier.mockResolvedValue(applier);
      renderProvider();

      const result = await capturedToolHandler!({ title: 'My view', messages: [] });

      expect(result).toEqual({ content: 'Unknown component "Widget"', isError: true });
    });
  });

  describe('pull-based assistant context provider', () => {
    it('does not register a context provider when the active provider lacks client-tool support', () => {
      makeAssistant({ activeProvider: { supports_client_tools: false } });
      renderProvider();

      expect(mockRegisterAssistantContextProvider).not.toHaveBeenCalled();
    });

    it('does not register a context provider when there is no active provider yet', () => {
      makeAssistant({ activeProvider: null });
      renderProvider();

      expect(mockRegisterAssistantContextProvider).not.toHaveBeenCalled();
    });

    it('registers a context provider once the active provider supports client tools', () => {
      makeAssistant({ activeProvider: { supports_client_tools: true } });
      renderProvider();

      expect(mockRegisterAssistantContextProvider).toHaveBeenCalledWith('customTraceView', expect.any(Function));
    });

    it('returns null and latches nothing when there is no published authoring context', () => {
      makeAssistant({ activeProvider: { supports_client_tools: true } });
      mockGetCustomViewAuthoringContext.mockReturnValue(null);
      renderProvider();

      expect(capturedContextProvider!()).toBeNull();
      expect(mockLatchDispatchedCustomViewApplyTarget).not.toHaveBeenCalled();
    });

    it('latches the apply target and returns the guide + trace sample + current template', () => {
      makeAssistant({ activeProvider: { supports_client_tools: true } });
      const applyTarget = { id: 'v1', name: 'My view', instruction: 'do it', createdAtMs: 1 };
      mockGetCustomViewAuthoringContext.mockReturnValue({
        guide: 'the-guide',
        traceSample: { foo: 'bar' },
        currentTemplate: [{ id: 'root' }],
        applyTarget,
      });
      renderProvider();

      const result = capturedContextProvider!();

      expect(mockLatchDispatchedCustomViewApplyTarget).toHaveBeenCalledWith(applyTarget);
      expect(result).toEqual({
        guide: 'the-guide',
        traceSample: { foo: 'bar' },
        currentTemplate: [{ id: 'root' }],
      });
    });
  });
});
