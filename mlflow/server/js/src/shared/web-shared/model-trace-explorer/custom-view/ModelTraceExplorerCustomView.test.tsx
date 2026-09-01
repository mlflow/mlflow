import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { ModelTraceExplorerCustomView } from './ModelTraceExplorerCustomView';
import type { RenderCustomViewSpec } from './assistant/customViewSpecApplier';
import { useCustomViewAssistantBridge, type CustomViewAssistantBridge } from './assistant/useCustomViewAssistantBridge';
import { CustomViewDefinitionProvider, useOptionalCustomViewDefinition } from './CustomViewDefinitionContext';
import { latchDispatchedCustomViewApplyTarget } from './assistant/customViewAuthoringContext';
import { MAX_CUSTOM_VIEWS_PER_EXPERIMENT, toCustomViewApplyTarget, type CustomView } from './customViewDefinition';
import type { CreateAssessmentPayload } from '../api';
import { shouldUseTracesV4API } from '../FeatureUtils';
import { ModelSpanType, type ModelTrace, type ModelTraceSpanNode } from '../ModelTrace.types';
import {
  ModelTraceExplorerViewStateContext,
  type ModelTraceExplorerViewState,
} from '../ModelTraceExplorerViewStateContext';
import { setupServer } from '../../test-utils/setup-msw';

// Mock the assistant bridge so the tests drive the component purely off its
// return value (availability / openAssistant / applyError), without the
// module-level registries the real bridge wires up.
jest.mock('./assistant/useCustomViewAssistantBridge', () => ({
  useCustomViewAssistantBridge: jest.fn(),
}));

jest.mock('../FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../FeatureUtils')>('../FeatureUtils'),
  shouldUseTracesV4API: jest.fn(() => false),
  doesTraceSupportV4API: jest.fn(() => false),
  shouldEnableAssessmentsInSessions: jest.fn(() => false),
}));

jest.mock('../../global-settings/getUser', () => ({
  ...jest.requireActual<typeof import('../../global-settings/getUser')>('../../global-settings/getUser'),
  getUser: jest.fn(() => 'test-user@example.com'),
}));

const { server } = setupServer();

const mockUseCustomViewAssistantBridge = jest.mocked(useCustomViewAssistantBridge);

// Minimal legacy trace info: the custom-view builders normalize undefined/legacy
// fields, so no full trace fixture is needed to exercise the authoring UI. The
// default ModelTraceExplorerViewState context supplies an empty nodeMap.
const modelTraceInfo = { request_id: 'tr-test' } as ModelTrace['info'];
const feedbackNodeMap = {
  'span-1': {
    key: 'span-1',
    start: 0,
    title: 'test-tool',
    type: ModelSpanType.TOOL,
  } as unknown as ModelTraceSpanNode,
};

const makeView = (id: string, overrides: Partial<CustomView> = {}): CustomView => ({
  id,
  name: `name-${id}`,
  label: `label-${id}`,
  instruction: `do ${id}`,
  template: [
    {
      version: 'v0.9',
      updateComponents: { surfaceId: 'main', components: [{ id: 'root', component: 'Text', text: 'hi' }] },
    },
  ],
  createdAtMs: 1,
  ...overrides,
});

const persistedView = makeView('v1');
const noViews: CustomView[] = [];
const noopPersistView = (_view: CustomView): Promise<void> => Promise.resolve();

const customView = () => (
  <IntlProvider locale="en" messages={{}}>
    <DesignSystemProvider>
      <QueryClientProvider client={new QueryClient()}>
        <CustomViewDefinitionProvider views={noViews} isLoaded onPersistView={noopPersistView} canModifyPersistedViews>
          <ModelTraceExplorerCustomView modelTraceInfo={modelTraceInfo} />
        </CustomViewDefinitionProvider>
      </QueryClientProvider>
    </DesignSystemProvider>
  </IntlProvider>
);

const renderCustomView = () => render(customView());

const setBridge = (overrides: Partial<CustomViewAssistantBridge> = {}): CustomViewAssistantBridge => {
  const value: CustomViewAssistantBridge = {
    isAvailable: true,
    openAssistant: jest.fn(),
    isStreaming: false,
    isPending: false,
    applyError: undefined,
    clearApplyError: jest.fn(),
    ...overrides,
  };
  mockUseCustomViewAssistantBridge.mockReturnValue(value);
  return value;
};

// Surfaces the active view's stored `instruction` so a test can assert what the
// component recorded through the shared definition context (nothing renders it
// in the authoring UI itself).
const InstructionProbe = () => {
  const cv = useOptionalCustomViewDefinition();
  return <div data-testid="active-instruction">{cv?.activeView?.instruction ?? '(none)'}</div>;
};

// Exposes the context's selectView so a test can change the active view
// programmatically (simulating a background selection change, e.g. an
// assistant apply) without a DOM click — an outside click would dismiss an
// open modal.
const SelectViewProbe = ({ selectViewRef }: { selectViewRef: { current?: (id: string) => void } }) => {
  const cv = useOptionalCustomViewDefinition();
  selectViewRef.current = cv?.selectView;
  return null;
};

const renderWithProvider = () =>
  render(
    <IntlProvider locale="en" messages={{}}>
      <DesignSystemProvider>
        <QueryClientProvider client={new QueryClient()}>
          <CustomViewDefinitionProvider
            views={noViews}
            isLoaded
            onPersistView={noopPersistView}
            canModifyPersistedViews
          >
            <ModelTraceExplorerCustomView modelTraceInfo={modelTraceInfo} />
            <InstructionProbe />
          </CustomViewDefinitionProvider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );

const renderWithPersistProvider = (canModifyPersistedViews: boolean, views: CustomView[] = [persistedView]) => {
  const onPersistView = jest.fn<(view: CustomView) => Promise<void>>().mockResolvedValue(undefined);
  const onDeleteView = jest.fn<(id: string) => Promise<void>>().mockResolvedValue(undefined);
  render(
    <IntlProvider locale="en" messages={{}}>
      <DesignSystemProvider>
        <QueryClientProvider client={new QueryClient()}>
          <CustomViewDefinitionProvider
            views={views}
            isLoaded
            onPersistView={onPersistView}
            onDeleteView={onDeleteView}
            canModifyPersistedViews={canModifyPersistedViews}
          >
            <ModelTraceExplorerCustomView modelTraceInfo={modelTraceInfo} />
          </CustomViewDefinitionProvider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onPersistView, onDeleteView };
};

const renderWithChangingPermission = (canModifyPersistedViews: boolean) => {
  const queryClient = new QueryClient();
  const element = (canModify: boolean) => (
    <IntlProvider locale="en" messages={{}}>
      <DesignSystemProvider>
        <QueryClientProvider client={queryClient}>
          <CustomViewDefinitionProvider
            views={[persistedView]}
            isLoaded
            onPersistView={noopPersistView}
            canModifyPersistedViews={canModify}
          >
            <ModelTraceExplorerCustomView modelTraceInfo={modelTraceInfo} />
          </CustomViewDefinitionProvider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>
  );
  const result = render(element(canModifyPersistedViews));
  return {
    rerenderWithPermission: (canModify: boolean) => result.rerender(element(canModify)),
  };
};

// Renders with a persisting provider whose onPersistView is returned so a test
// can assert what (and with which name) the first save persists.
const renderWithPersist = (views: CustomView[] = []) => {
  const onPersistView = jest.fn<(view: CustomView) => Promise<void>>().mockResolvedValue(undefined);
  render(
    <IntlProvider locale="en" messages={{}}>
      <DesignSystemProvider>
        <QueryClientProvider client={new QueryClient()}>
          <CustomViewDefinitionProvider views={views} isLoaded onPersistView={onPersistView} canModifyPersistedViews>
            <ModelTraceExplorerCustomView modelTraceInfo={modelTraceInfo} />
          </CustomViewDefinitionProvider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onPersistView };
};

// A view whose stored template carries a forbidden trace-specific "#span:"
// deeplink — the shape a directly-written / tampered experiment tag could take.
// It is handed straight to the provider (bypassing parseCustomView's load gate)
// to exercise the render-time re-bind gate.
const tamperedView = makeView('tampered', {
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [{ id: 'root', component: 'Text', text: 'jump to #span:abc123' }],
      },
    },
  ],
});

// A view saved while the TreeView / TreeNode primitives still existed. They were
// removed from the catalog, so the closed allowlist now rejects the stored
// template and the user rebuilds the view with the assistant.
const legacyTreeView = makeView('legacy-tree', {
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [{ id: 'root', component: 'TreeView', title: 'Span Tree', children: { $source: 'spanTree' } }],
      },
    },
  ],
});

// A view saved while the DataTable primitive still existed, binding its rows to
// the (also removed) `toolRows` array source. Same hard-fail path as the tree
// components above.
const legacyDataTableView = makeView('legacy-table', {
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [
          {
            id: 'root',
            component: 'DataTable',
            title: 'Tool Performance',
            columns: [{ label: 'Tool' }],
            rows: { $source: 'toolRows' },
          },
        ],
      },
    },
  ],
});

const feedbackPrimitivesView = makeView('feedback-primitives', {
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [
          {
            id: 'root',
            component: 'Column',
            children: ['thumbs', 'rating', 'rationale', 'note', 'submit'],
          },
          {
            id: 'thumbs',
            component: 'FeedbackThumbsUpDownButtons',
            label: 'Was this helpful?',
            name: 'Helpfulness',
            spanId: { $spanRef: { type: 'TOOL' } },
          },
          {
            id: 'rating',
            component: 'RadioGroup',
            label: 'Accuracy',
            name: 'Accuracy',
            formId: 'review',
            options: [
              { label: 'Accurate', value: 'accurate' },
              { label: 'Inaccurate', value: 'inaccurate' },
            ],
          },
          {
            id: 'rationale',
            component: 'FeedbackInputText',
            label: 'Accuracy rationale',
            name: 'Accuracy',
            formId: 'review',
          },
          {
            id: 'note',
            component: 'FeedbackInputText',
            label: 'Additional note',
            name: 'Notes',
            field: 'value',
            formId: 'review',
          },
          { id: 'submit', component: 'FeedbackSubmit', label: 'Submit review', formId: 'review' },
        ],
      },
    },
  ],
});

const twoFormFeedbackView = makeView('two-forms', {
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [
          {
            id: 'root',
            component: 'Column',
            children: ['trace-rating', 'trace-submit', 'span-rating', 'span-submit'],
          },
          {
            id: 'trace-rating',
            component: 'RadioGroup',
            name: 'TraceQuality',
            formId: 'trace',
            options: [{ label: 'Trace good', value: 'good' }],
          },
          { id: 'trace-submit', component: 'FeedbackSubmit', label: 'Submit trace feedback', formId: 'trace' },
          {
            id: 'span-rating',
            component: 'RadioGroup',
            name: 'SpanQuality',
            spanId: { $spanRef: { type: 'TOOL' } },
            formId: 'span',
            options: [{ label: 'Span bad', value: 'bad' }],
          },
          { id: 'span-submit', component: 'FeedbackSubmit', label: 'Submit span feedback', formId: 'span' },
        ],
      },
    },
  ],
});

const prefilledFeedbackView = makeView('prefilled-feedback', {
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [
          { id: 'root', component: 'Column', children: ['rating', 'note', 'submit'] },
          {
            id: 'rating',
            component: 'RadioGroup',
            name: 'Accuracy',
            formId: 'prefilled',
            value: 'accurate',
            options: [{ label: 'Accurate', value: 'accurate' }],
          },
          {
            id: 'note',
            component: 'FeedbackInputText',
            name: 'Notes',
            field: 'value',
            formId: 'prefilled',
            value: 'Prefilled note',
          },
          { id: 'submit', component: 'FeedbackSubmit', label: 'Submit prefilled', formId: 'prefilled' },
        ],
      },
    },
  ],
});

const sameNameFeedbackView = makeView('same-name-feedback', {
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [
          { id: 'root', component: 'Column', children: ['first', 'second', 'first-submit', 'second-submit'] },
          {
            id: 'first',
            component: 'RadioGroup',
            name: 'Quality',
            formId: 'first-form',
            options: [{ label: 'First good', value: 'good' }],
          },
          {
            id: 'second',
            component: 'RadioGroup',
            name: 'Quality',
            formId: 'second-form',
            options: [{ label: 'Second good', value: 'good' }],
          },
          { id: 'first-submit', component: 'FeedbackSubmit', formId: 'first-form' },
          { id: 'second-submit', component: 'FeedbackSubmit', formId: 'second-form' },
        ],
      },
    },
  ],
});

const preselectedThumbsView = makeView('preselected-thumbs', {
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [
          {
            id: 'root',
            component: 'FeedbackThumbsUpDownButtons',
            name: 'Helpful',
            value: true,
          },
        ],
      },
    },
  ],
});

const renderWithViews = (views: CustomView[], traceInfo: ModelTrace['info'] = modelTraceInfo, canPersist = false) =>
  render(
    <IntlProvider locale="en" messages={{}}>
      <DesignSystemProvider>
        <QueryClientProvider client={new QueryClient()}>
          <ModelTraceExplorerViewStateContext.Provider
            value={{ nodeMap: feedbackNodeMap } as unknown as ModelTraceExplorerViewState}
          >
            <CustomViewDefinitionProvider
              views={views}
              isLoaded
              onPersistView={canPersist ? noopPersistView : undefined}
              canModifyPersistedViews={canPersist}
            >
              <ModelTraceExplorerCustomView modelTraceInfo={traceInfo} />
            </CustomViewDefinitionProvider>
          </ModelTraceExplorerViewStateContext.Provider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );

const selectPersistedViewAndDirty = async () => {
  await userEvent.click(screen.getByRole('button', { name: /Select a custom view/ }));
  await userEvent.click(screen.getByRole('menuitemcheckbox', { name: /name-v1/ }));
  await act(async () => {
    await latestOnSpec()(specWithTitle('Updated label'));
  });
};

// The component hands `onSpec` to the (mocked) bridge on every render; grab the
// latest closure so we can drive a `render_custom_view` apply directly.
const latestOnSpec = (): ((spec: RenderCustomViewSpec) => Promise<void> | void) => {
  const { calls } = mockUseCustomViewAssistantBridge.mock;
  return calls[calls.length - 1][0].onSpec;
};

const specWithTitle = (title: string): RenderCustomViewSpec => ({
  title,
  messages: [
    {
      version: 'v0.9',
      updateComponents: { surfaceId: 'main', components: [{ id: 'root', component: 'Text', text: 'hi' }] },
    },
  ],
});

describe('ModelTraceExplorerCustomView', () => {
  beforeEach(() => {
    mockUseCustomViewAssistantBridge.mockReset();
    jest.mocked(shouldUseTracesV4API).mockReturnValue(false);
    // The latch is module-level state owned by the page's context plugin; clear
    // it so a test that sets it can't retarget a later test's apply.
    latchDispatchedCustomViewApplyTarget(undefined);
  });

  it('renders the authoring prompt UI when the assistant bridge is available', () => {
    setBridge();
    renderCustomView();

    expect(screen.getByText('Build a custom trace view')).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Build with Assistant/ })).toBeInTheDocument();
  });

  it('disables the build button until the user types a prompt', async () => {
    setBridge();
    renderCustomView();

    const buildButton = screen.getByRole('button', { name: /Build with Assistant/ });
    expect(buildButton).toBeDisabled();

    await userEvent.type(screen.getByRole('textbox'), 'Show me the failed spans');
    expect(buildButton).toBeEnabled();
  });

  it('hands the typed prompt to the assistant and shows the building skeleton on submit', async () => {
    const openAssistant = jest.fn();
    openAssistant.mockImplementation(() => setBridge({ openAssistant, isStreaming: true }));
    setBridge({ openAssistant });
    renderCustomView();

    await userEvent.type(screen.getByRole('textbox'), 'Show me the failed spans');
    await userEvent.click(screen.getByRole('button', { name: /Build with Assistant/ }));

    expect(openAssistant).toHaveBeenCalledTimes(1);
    expect(openAssistant).toHaveBeenCalledWith('Show me the failed spans', { newSession: true });
    // The empty-state prompt box is replaced by the loading skeleton while the assistant builds.
    expect(screen.getByText('Building this view…')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Build with Assistant/ })).not.toBeInTheDocument();
  });

  it('submits the typed prompt to the assistant when pressing Enter', async () => {
    const openAssistant = jest.fn();
    openAssistant.mockImplementation(() => setBridge({ openAssistant, isStreaming: true }));
    setBridge({ openAssistant });
    renderCustomView();

    await userEvent.type(screen.getByRole('textbox'), 'Show me the failed spans');
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });

    expect(openAssistant).toHaveBeenCalledTimes(1);
    expect(openAssistant).toHaveBeenCalledWith('Show me the failed spans', { newSession: true });
    expect(screen.getByText('Building this view…')).toBeInTheDocument();
  });

  it('does not submit on Shift+Enter so the user can insert a newline', async () => {
    const openAssistant = jest.fn();
    setBridge({ openAssistant });
    renderCustomView();

    await userEvent.type(screen.getByRole('textbox'), 'Show me the failed spans');
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter', shiftKey: true });

    expect(openAssistant).not.toHaveBeenCalled();
    // The prompt box stays visible instead of switching to the building skeleton.
    expect(screen.getByRole('button', { name: /Build with Assistant/ })).toBeInTheDocument();
  });

  it('does not submit on Enter when the prompt is empty', () => {
    const openAssistant = jest.fn();
    setBridge({ openAssistant });
    renderCustomView();

    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });

    expect(openAssistant).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: /Build with Assistant/ })).toBeInTheDocument();
  });

  it('does not submit on Enter while an IME composition is being confirmed', async () => {
    const openAssistant = jest.fn();
    setBridge({ openAssistant });
    renderCustomView();

    await userEvent.type(screen.getByRole('textbox'), 'Show me the failed spans');
    // isComposing marks the Enter that confirms an in-progress IME composition
    // (e.g. CJK input), which must not submit the prompt.
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter', isComposing: true });

    expect(openAssistant).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: /Build with Assistant/ })).toBeInTheDocument();
  });

  it('keeps the building skeleton after streaming ends until a view is created', () => {
    const openAssistant = jest.fn();
    openAssistant.mockImplementation(() => setBridge({ openAssistant, isStreaming: true }));
    setBridge({ openAssistant, isStreaming: false });
    const { rerender } = renderCustomView();

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Show me the failed spans' } });
    fireEvent.click(screen.getByRole('button', { name: /Build with Assistant/ }));
    expect(screen.getByText('Building this view…')).toBeInTheDocument();

    // Streaming finishes but render_custom_view has not applied a view yet. The
    // skeleton must NOT clear on the streaming falling edge (that flashed the
    // prompt back mid-build) — it stays until activeView or applyError.
    setBridge({ openAssistant, isStreaming: false });
    rerender(customView());
    expect(screen.getByText('Building this view…')).toBeInTheDocument();
  });

  it('keeps the prompt visible while queued and abandons cleanly when the queue is cleared', () => {
    const openAssistant = jest.fn();
    openAssistant.mockImplementation(() => setBridge({ openAssistant, isPending: true }));
    setBridge({ openAssistant });
    const { rerender } = renderCustomView();

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Show me the failed spans' } });
    fireEvent.click(screen.getByRole('button', { name: /Build with Assistant/ }));

    expect(screen.getByRole('textbox')).toHaveValue('Show me the failed spans');
    expect(screen.getByRole('button', { name: /Build with Assistant/ })).toBeDisabled();
    expect(screen.queryByText('Building this view…')).not.toBeInTheDocument();

    setBridge({ openAssistant, isPending: false });
    rerender(customView());

    expect(screen.getByRole('textbox')).toHaveValue('Show me the failed spans');
    expect(screen.getByRole('button', { name: /Build with Assistant/ })).toBeEnabled();
    expect(screen.queryByText('Building this view…')).not.toBeInTheDocument();
  });

  it('clears the building skeleton and surfaces the error when the spec apply fails', () => {
    const openAssistant = jest.fn();
    openAssistant.mockImplementation(() => setBridge({ openAssistant, isStreaming: true }));
    setBridge({ openAssistant });
    const { rerender } = renderCustomView();

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Show me the failed spans' } });
    fireEvent.click(screen.getByRole('button', { name: /Build with Assistant/ }));
    expect(screen.getByText('Building this view…')).toBeInTheDocument();

    // The render_custom_view apply fails: the skeleton clears, the authoring UI
    // returns, and the inline assistant error is shown.
    setBridge({ openAssistant, applyError: 'Unknown component "Widget"' });
    rerender(customView());

    expect(screen.queryByText('Building this view…')).not.toBeInTheDocument();
    expect(screen.getByText('Assistant: Unknown component "Widget"')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Build with Assistant/ })).toBeInTheDocument();
  });

  it('keeps the typed prompt and does not enter the building state when the launcher throws', async () => {
    const throwingOpenAssistant = jest.fn(() => {
      throw new Error('launch failed');
    });
    setBridge({ openAssistant: throwingOpenAssistant });
    renderCustomView();

    await userEvent.type(screen.getByRole('textbox'), 'Show me the failed spans');
    await userEvent.click(screen.getByRole('button', { name: /Build with Assistant/ }));

    expect(throwingOpenAssistant).toHaveBeenCalledTimes(1);
    // The prompt is preserved for retry and the skeleton never appears.
    expect(screen.getByRole('textbox')).toHaveValue('Show me the failed spans');
    expect(screen.queryByText('Building this view…')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Build with Assistant/ })).toBeInTheDocument();
  });

  it('records the submitted prompt as the built view instruction', async () => {
    setBridge();
    renderWithProvider();

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Show me the failed spans' } });
    fireEvent.click(screen.getByRole('button', { name: /Build with Assistant/ }));

    // The render_custom_view tool applies the agent's spec.
    await act(async () => {
      await latestOnSpec()(specWithTitle('Failed spans'));
    });

    // The view saves the prompt that launched it, not an empty instruction.
    expect(screen.getByTestId('active-instruction')).toHaveTextContent('Show me the failed spans');
  });

  it('keeps the prior instruction for an assistant edit that sends no empty-state prompt', async () => {
    setBridge();
    renderWithProvider();

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'First prompt' } });
    fireEvent.click(screen.getByRole('button', { name: /Build with Assistant/ }));
    await act(async () => {
      await latestOnSpec()(specWithTitle('Initial view'));
    });
    expect(screen.getByTestId('active-instruction')).toHaveTextContent('First prompt');

    // "Edit with Assistant" opens the panel with no empty-state prompt — the
    // edit request is typed inside the assistant, which the host never sees.
    fireEvent.click(screen.getByRole('button', { name: /Edit with Assistant/ }));
    await act(async () => {
      await latestOnSpec()(specWithTitle('Edited view'));
    });

    // With no fresh prompt to record, the view keeps its prior instruction
    // rather than wiping it to empty.
    expect(screen.getByTestId('active-instruction')).toHaveTextContent('First prompt');
  });

  it('shows an unavailable empty state when the assistant bridge is not available', () => {
    setBridge({ isAvailable: false, openAssistant: undefined });
    renderCustomView();

    expect(screen.getByText('Assistant unavailable')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Build with Assistant/ })).not.toBeInTheDocument();
  });

  it('hides create, edit, and save controls when the user cannot modify custom views', async () => {
    setBridge();
    renderWithPersistProvider(false);

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Select a custom view/ }));
    expect(screen.queryByRole('menuitem', { name: /Create view/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole('menuitemcheckbox', { name: /name-v1/ }));

    expect(() => latestOnSpec()(specWithTitle('Blocked update'))).toThrow(
      'Custom views cannot be modified in this experiment.',
    );
    expect(screen.queryByRole('button', { name: 'Save' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Edit with Assistant/ })).not.toBeInTheDocument();
    // The overflow menu is absent, so Rename (which lives inside it) is
    // unreachable for read-only users.
    expect(screen.queryByRole('button', { name: 'More view options' })).not.toBeInTheDocument();
  });

  it('shows a non-authoring empty state to read-only users when there are no saved views', () => {
    setBridge();
    renderWithPersistProvider(false, []);

    expect(screen.getByText('No custom views')).toBeInTheDocument();
    expect(screen.queryByText('Build a custom trace view')).not.toBeInTheDocument();
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Build with Assistant/ })).not.toBeInTheDocument();
    expect(mockUseCustomViewAssistantBridge).toHaveBeenLastCalledWith(expect.objectContaining({ enabled: false }));
  });

  it('shows saved views instead of the no-views state when edit permission is lost during a draft', async () => {
    setBridge();
    const { rerenderWithPermission } = renderWithChangingPermission(true);
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await user.click(screen.getByRole('menuitem', { name: /Create view/ }));
    expect(screen.getByRole('button', { name: /Untitled custom view/ })).toBeInTheDocument();

    rerenderWithPermission(false);

    expect(screen.getByRole('button', { name: /Select a custom view/ })).toBeInTheDocument();
    expect(screen.getByText('Choose a saved view from the menu to render it for this trace.')).toBeInTheDocument();
    expect(screen.queryByText('No custom views')).not.toBeInTheDocument();
  });

  it('hides Create view when the assistant bridge is unavailable', async () => {
    setBridge({ isAvailable: false, openAssistant: undefined });
    renderWithPersistProvider(true);

    await userEvent.click(screen.getByRole('button', { name: /Select a custom view/ }));

    expect(screen.queryByRole('menuitem', { name: /Create view/ })).not.toBeInTheDocument();
  });

  it('shows Save and the overflow menu when the user can modify persisted views', async () => {
    setBridge();
    renderWithPersistProvider(true);

    await selectPersistedViewAndDirty();

    expect(screen.getByRole('button', { name: 'Save' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Edit with Assistant/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'More view options' })).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /name-v1/ }));
    expect(screen.getByRole('menuitem', { name: /Create view/ })).toBeInTheDocument();
  });

  it('shows Rename view in the overflow menu for editors', async () => {
    setBridge();
    renderWithPersistProvider(true);

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await user.click(screen.getByRole('menuitemcheckbox', { name: /name-v1/ }));
    await user.click(screen.getByRole('button', { name: 'More view options' }));

    // A readable view: rename is enabled (not aria-disabled).
    expect(screen.getByRole('menuitem', { name: /Rename view/ })).not.toHaveAttribute('aria-disabled', 'true');
    expect(screen.getByRole('menuitem', { name: /Delete view/ })).toBeInTheDocument();
  });

  it('hides Delete view when the provider does not supply a delete callback', async () => {
    setBridge();
    render(
      <IntlProvider locale="en" messages={{}}>
        <DesignSystemProvider>
          <QueryClientProvider client={new QueryClient()}>
            <CustomViewDefinitionProvider
              views={[persistedView]}
              isLoaded
              onPersistView={noopPersistView}
              canModifyPersistedViews
            >
              <ModelTraceExplorerCustomView modelTraceInfo={modelTraceInfo} />
            </CustomViewDefinitionProvider>
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await user.click(screen.getByRole('menuitemcheckbox', { name: /name-v1/ }));
    await user.click(screen.getByRole('button', { name: 'More view options' }));

    expect(screen.getByRole('menuitem', { name: /Rename view/ })).toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: /Delete view/ })).not.toBeInTheDocument();
  });

  it('confirms and deletes the selected persisted view', async () => {
    setBridge();
    const { onDeleteView } = renderWithPersistProvider(true);

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await user.click(screen.getByRole('menuitemcheckbox', { name: /name-v1/ }));
    await user.click(screen.getByRole('button', { name: 'More view options' }));
    await user.click(screen.getByRole('menuitem', { name: /Delete view/ }));

    const dialog = await screen.findByRole('dialog', { name: /Delete view/ });
    expect(within(dialog).getByText(/Delete the view "name-v1"/)).toBeInTheDocument();
    await user.click(within(dialog).getByRole('button', { name: 'Delete' }));

    expect(onDeleteView).toHaveBeenCalledWith('v1');
    expect(await screen.findByText('Build a custom trace view')).toBeInTheDocument();
  });

  it('deletes the view the modal was opened for when selection changes in the background', async () => {
    setBridge();
    const viewA = makeView('v1');
    const viewB = makeView('v2');
    const onDeleteView = jest.fn<(id: string) => Promise<void>>().mockResolvedValue(undefined);
    const selectViewRef: { current?: (id: string) => void } = {};
    render(
      <IntlProvider locale="en" messages={{}}>
        <DesignSystemProvider>
          <QueryClientProvider client={new QueryClient()}>
            <CustomViewDefinitionProvider
              views={[viewA, viewB]}
              isLoaded
              onPersistView={noopPersistView}
              onDeleteView={onDeleteView}
              canModifyPersistedViews
            >
              <ModelTraceExplorerCustomView modelTraceInfo={modelTraceInfo} />
              <SelectViewProbe selectViewRef={selectViewRef} />
            </CustomViewDefinitionProvider>
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await user.click(screen.getByRole('menuitemcheckbox', { name: /name-v1/ }));
    await user.click(screen.getByRole('button', { name: 'More view options' }));
    await user.click(screen.getByRole('menuitem', { name: /Delete view/ }));

    const dialog = await screen.findByRole('dialog', { name: /Delete view/ });
    expect(within(dialog).getByText(/Delete the view "name-v1"/)).toBeInTheDocument();

    act(() => selectViewRef.current?.('v2'));
    expect(within(dialog).getByText(/Delete the view "name-v1"/)).toBeInTheDocument();

    await user.click(within(dialog).getByRole('button', { name: 'Delete' }));
    expect(onDeleteView).toHaveBeenCalledWith('v1');
    expect(onDeleteView).not.toHaveBeenCalledWith('v2');
    expect(await screen.findByRole('button', { name: /name-v2/ })).toBeInTheDocument();
  });

  it('disables (not hides) Rename view for an unreadable persisted view', async () => {
    setBridge();
    // An unreadable persisted view (its saved definition can't be rendered). Rename
    // must stay visible-but-disabled (with an explanatory tooltip) so the user
    // learns why.
    renderWithPersistProvider(true, [makeView('bad', { unreadable: true })]);

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await user.click(screen.getByRole('menuitemcheckbox', { name: /name-bad/ }));
    await user.click(screen.getByRole('button', { name: 'More view options' }));

    // Rename is rendered but disabled (not removed from the menu).
    expect(screen.getByRole('menuitem', { name: /Rename view/ })).toHaveAttribute('aria-disabled', 'true');
    expect(screen.getByRole('menuitem', { name: /Delete view/ })).not.toHaveAttribute('aria-disabled', 'true');
  });

  it('disables Rename for a valid-shape view whose template fails validation (Case-2 unreadable derived at selection)', async () => {
    setBridge();
    // A persisted view with a valid CustomView shape but an invalid template
    // (forbidden `#span:` narrative). It is NOT flagged `unreadable` at load
    // anymore (template validation is deferred), so this proves the Case-2
    // gating is derived when the view becomes active: Rename is disabled and
    // the render placeholder shows.
    renderWithPersistProvider(true, [tamperedView]);

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await user.click(screen.getByRole('menuitemcheckbox', { name: /name-tampered/ }));

    expect(await screen.findByText(/couldn't be read and can't be displayed/)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'More view options' }));
    expect(screen.getByRole('menuitem', { name: /Rename view/ })).toHaveAttribute('aria-disabled', 'true');
    expect(screen.getByRole('menuitem', { name: /Delete view/ })).not.toHaveAttribute('aria-disabled', 'true');
  });

  it('renames the selected view: prefills the current name and persists the trimmed new name against the saved template', async () => {
    setBridge();
    const { onPersistView } = renderWithPersistProvider(true);

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await user.click(screen.getByRole('menuitemcheckbox', { name: /name-v1/ }));
    await user.click(screen.getByRole('button', { name: 'More view options' }));
    await user.click(screen.getByRole('menuitem', { name: /Rename view/ }));

    const dialog = await screen.findByRole('dialog', { name: /Rename custom view/ });
    const input = within(dialog).getByRole('textbox');
    // The modal prefills the current user-facing name.
    expect(input).toHaveValue('name-v1');

    fireEvent.change(input, { target: { value: '  Renamed view  ' } });
    await user.click(within(dialog).getByRole('button', { name: 'Save' }));

    // Persists the trimmed name against the view's saved template (metadata-only).
    expect(onPersistView).toHaveBeenCalledWith({ ...persistedView, name: 'Renamed view' });
    // The switcher trigger reflects the new name.
    expect(await screen.findByRole('button', { name: /Renamed view/ })).toBeInTheDocument();
  });

  it('disables the rename confirm button for an empty name', async () => {
    setBridge();
    renderWithPersistProvider(true);

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await user.click(screen.getByRole('menuitemcheckbox', { name: /name-v1/ }));
    await user.click(screen.getByRole('button', { name: 'More view options' }));
    await user.click(screen.getByRole('menuitem', { name: /Rename view/ }));

    const dialog = await screen.findByRole('dialog', { name: /Rename custom view/ });
    await user.clear(within(dialog).getByRole('textbox'));

    expect(within(dialog).getByRole('button', { name: 'Save' })).toBeDisabled();
  });

  it('renames the view the modal was opened for, even if the active selection changes while it is open', async () => {
    setBridge();
    const viewA = makeView('v1');
    const viewB = makeView('v2');
    const onPersistView = jest.fn<(view: CustomView) => Promise<void>>().mockResolvedValue(undefined);
    const selectViewRef: { current?: (id: string) => void } = {};
    render(
      <IntlProvider locale="en" messages={{}}>
        <DesignSystemProvider>
          <QueryClientProvider client={new QueryClient()}>
            <CustomViewDefinitionProvider
              views={[viewA, viewB]}
              isLoaded
              onPersistView={onPersistView}
              canModifyPersistedViews
            >
              <ModelTraceExplorerCustomView modelTraceInfo={modelTraceInfo} />
              <SelectViewProbe selectViewRef={selectViewRef} />
            </CustomViewDefinitionProvider>
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );

    const user = userEvent.setup();
    // Select v1 and open its rename modal (captures v1 as the target).
    await user.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await user.click(screen.getByRole('menuitemcheckbox', { name: /name-v1/ }));
    await user.click(screen.getByRole('button', { name: 'More view options' }));
    await user.click(screen.getByRole('menuitem', { name: /Rename view/ }));

    const dialog = await screen.findByRole('dialog', { name: /Rename custom view/ });
    expect(within(dialog).getByRole('textbox')).toHaveValue('name-v1');

    // The active selection changes to v2 while the modal is still open (no DOM
    // click, which would dismiss the modal — this simulates a background change).
    act(() => selectViewRef.current?.('v2'));

    fireEvent.change(within(dialog).getByRole('textbox'), { target: { value: 'Renamed v1' } });
    await user.click(within(dialog).getByRole('button', { name: 'Save' }));

    // The rename lands on v1 (the modal's captured target), NOT the now-active v2.
    expect(onPersistView).toHaveBeenCalledWith({ ...viewA, name: 'Renamed v1' });
    expect(onPersistView).not.toHaveBeenCalledWith(expect.objectContaining({ id: 'v2' }));
  });

  it('applies an assistant edit to the view the request was made against, not the one selected mid-flight', async () => {
    setBridge();
    const viewA = makeView('v1');
    const viewB = makeView('v2');
    const selectViewRef: { current?: (id: string) => void } = {};
    render(
      <IntlProvider locale="en" messages={{}}>
        <DesignSystemProvider>
          <QueryClientProvider client={new QueryClient()}>
            <CustomViewDefinitionProvider
              views={[viewA, viewB]}
              isLoaded
              onPersistView={noopPersistView}
              canModifyPersistedViews
            >
              <ModelTraceExplorerCustomView modelTraceInfo={modelTraceInfo} />
              <SelectViewProbe selectViewRef={selectViewRef} />
            </CustomViewDefinitionProvider>
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await user.click(screen.getByRole('menuitemcheckbox', { name: /name-v1/ }));

    // The user types the request straight into the assistant panel, so neither
    // host launcher runs. This is the page context plugin latching v1 as the
    // prompt for that turn is assembled.
    act(() => latchDispatchedCustomViewApplyTarget(toCustomViewApplyTarget(viewA)));

    // While the agent is running, the user switches to v2.
    act(() => selectViewRef.current?.('v2'));

    await act(async () => {
      await latestOnSpec()(specWithTitle('Updated label'));
    });

    // v2 keeps both the selection and its own content — the edit did not follow
    // the user's navigation.
    expect(screen.getByRole('button', { name: /name-v2/ })).toBeInTheDocument();
    expect(screen.getByText('label-v2')).toBeInTheDocument();
    expect(screen.queryByText('Updated label')).not.toBeInTheDocument();

    // v1 received the edit in the background, which leaves it dirty.
    await user.click(screen.getByRole('button', { name: /name-v2/ }));
    const v1Item = screen.getByRole('menuitemcheckbox', { name: /name-v1/ });
    expect(within(v1Item).getByText('(Draft)')).toBeInTheDocument();
    expect(within(screen.getByRole('menuitemcheckbox', { name: /name-v2/ })).queryByText('(Draft)')).toBeNull();

    await user.click(v1Item);
    expect(screen.getByText('Updated label')).toBeInTheDocument();
  });

  it('still builds a brand-new view whose reserved id is absent from the working set', async () => {
    setBridge();
    const { onPersistView } = renderWithPersist([]);

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Show failed spans' } });
    fireEvent.click(screen.getByRole('button', { name: /Build with Assistant/ }));
    await act(async () => {
      await latestOnSpec()(specWithTitle('Failed spans'));
    });

    expect(screen.getByText('Failed spans')).toBeInTheDocument();
    expect(onPersistView).not.toHaveBeenCalled();
  });

  it('preserves the saved view name when the assistant applies without a launch binding', async () => {
    setBridge();
    renderWithPersistProvider(true);

    await userEvent.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: /name-v1/ }));
    await act(async () => {
      await latestOnSpec()(specWithTitle('Updated label'));
    });

    // Dropdown trigger shows the user-provided name, not "Untitled view".
    expect(screen.getByRole('button', { name: /name-v1/ })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(screen.queryByRole('dialog', { name: /Name this custom view/ })).not.toBeInTheDocument();
  });

  it('takes the user straight to the draft authoring UI when Create view is clicked, with no naming modal', async () => {
    setBridge();
    renderWithPersistProvider(true);

    await userEvent.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await userEvent.click(screen.getByRole('menuitem', { name: /Create view/ }));

    // No up-front naming modal — the user lands directly in the authoring UI.
    expect(screen.queryByRole('dialog', { name: /Name this custom view/ })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Build with Assistant/ })).toBeInTheDocument();
    // The switcher shows the unsaved-view fallback label.
    expect(screen.getByRole('button', { name: /Untitled custom view/ })).toBeInTheDocument();
  });

  it('disables Create view at the per-experiment view limit', async () => {
    setBridge();
    const maxViews = Array.from({ length: MAX_CUSTOM_VIEWS_PER_EXPERIMENT }, (_unused, index) =>
      makeView(`limit-${index}`),
    );
    renderWithPersistProvider(true, maxViews);

    await userEvent.click(screen.getByRole('button', { name: /Select a custom view/ }));
    const createItem = screen.getByRole('menuitem', { name: /Create view/ });
    expect(createItem).toHaveAttribute('aria-disabled', 'true');
    expect(createItem.querySelector('[data-disabled-tooltip]')).toBeInTheDocument();

    // Dispatch directly because userEvent intentionally refuses pointer events on disabled items.
    // The handler guard must also keep the authoring UI closed.
    fireEvent.click(createItem);
    expect(screen.queryByRole('button', { name: /Build with Assistant/ })).not.toBeInTheDocument();
  });

  it('keeps Create view enabled one view below the limit', async () => {
    setBridge();
    const belowLimit = Array.from({ length: MAX_CUSTOM_VIEWS_PER_EXPERIMENT - 1 }, (_unused, index) =>
      makeView(`limit-${index}`),
    );
    renderWithPersistProvider(true, belowLimit);

    await userEvent.click(screen.getByRole('button', { name: /Select a custom view/ }));

    expect(screen.getByRole('menuitem', { name: /Create view/ })).not.toHaveAttribute('aria-disabled', 'true');
  });

  it('prompts for a name on the first save of a newly built view and persists with it', async () => {
    setBridge();
    const { onPersistView } = renderWithPersist([]);

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Show failed spans' } });
    fireEvent.click(screen.getByRole('button', { name: /Build with Assistant/ }));
    await act(async () => {
      await latestOnSpec()(specWithTitle('Failed spans'));
    });

    // A newly built, not-yet-persisted view: Save opens the naming modal.
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));
    const dialog = await screen.findByRole('dialog', { name: /Name this custom view/ });
    fireEvent.change(within(dialog).getByRole('textbox'), { target: { value: 'My View' } });
    await userEvent.click(within(dialog).getByRole('button', { name: 'Save' }));

    expect(onPersistView).toHaveBeenCalledWith(expect.objectContaining({ name: 'My View' }));
  });

  it('marks a newly built, never-saved view as a draft in the switcher and with a banner', async () => {
    setBridge();
    renderWithPersist([]);

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Show failed spans' } });
    fireEvent.click(screen.getByRole('button', { name: /Build with Assistant/ }));
    await act(async () => {
      await latestOnSpec()(specWithTitle('Failed spans'));
    });

    // The switcher trigger shows the untitled fallback with the draft marker.
    expect(screen.getByRole('button', { name: /Untitled custom view/ })).toBeInTheDocument();
    expect(screen.getByText('(Draft)')).toBeInTheDocument();
    // The unsaved-changes banner is shown for the never-saved view.
    expect(screen.getByText(/unsaved changes/)).toBeInTheDocument();
  });

  it('marks an edited saved view as a draft in the trigger, the list item, and with a banner', async () => {
    setBridge();
    renderWithPersistProvider(true);

    await selectPersistedViewAndDirty();

    // Trigger shows the saved name with the draft marker, plus the unsaved-changes banner.
    expect(screen.getByRole('button', { name: /name-v1/ })).toBeInTheDocument();
    expect(screen.getByText('(Draft)')).toBeInTheDocument();
    expect(screen.getByText(/unsaved changes/)).toBeInTheDocument();

    // The list item for the edited view carries the draft marker too.
    await userEvent.click(screen.getByRole('button', { name: /name-v1/ }));
    const item = screen.getByRole('menuitemcheckbox', { name: /name-v1/ });
    expect(within(item).getByText('(Draft)')).toBeInTheDocument();
  });

  it('does not mark a clean saved view as a draft', async () => {
    setBridge();
    renderWithPersistProvider(true);

    await userEvent.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: /name-v1/ }));

    // Selecting without editing leaves the view clean: no draft marker, no banner.
    expect(screen.getByRole('button', { name: /name-v1/ })).toBeInTheDocument();
    expect(screen.queryByText('(Draft)')).not.toBeInTheDocument();
    expect(screen.queryByText(/unsaved changes/)).not.toBeInTheDocument();
  });

  it('renders the inline error placeholder for a tampered template instead of the raw sink', async () => {
    setBridge();
    renderWithViews([tamperedView]);

    await userEvent.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: /name-tampered/ }));

    // The re-bind gate rejects the tampered template and shows the safe inline
    // placeholder rather than binding/rendering its untrusted content.
    expect(await screen.findByText(/couldn't be read and can't be displayed/)).toBeInTheDocument();
    expect(screen.queryByText(/jump to #span:abc123/)).not.toBeInTheDocument();
  });

  it('renders the inline error placeholder for a view saved with the removed TreeView component', async () => {
    setBridge();
    renderWithViews([legacyTreeView]);

    await userEvent.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: /name-legacy-tree/ }));

    expect(await screen.findByText(/couldn't be read and can't be displayed/)).toBeInTheDocument();
    expect(screen.queryByText('Span Tree')).not.toBeInTheDocument();
  });

  it('renders the inline error placeholder for a view saved with the removed DataTable component', async () => {
    setBridge();
    renderWithViews([legacyDataTableView]);

    await userEvent.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: /name-legacy-table/ }));

    expect(await screen.findByText(/couldn't be read and can't be displayed/)).toBeInTheDocument();
    expect(screen.queryByText('Tool Performance')).not.toBeInTheDocument();
  });

  const captureAssessmentPosts = (): CreateAssessmentPayload['assessment'][] => {
    const posted: CreateAssessmentPayload['assessment'][] = [];
    server.use(
      rest.post('*/ajax-api/*/mlflow/traces/*/assessments', async (req, res, ctx) => {
        const body = (await req.json()) as CreateAssessmentPayload;
        posted.push(body.assessment);
        return res(
          ctx.json({
            assessment: {
              ...body.assessment,
              assessment_id: `assessment-${posted.length}`,
              create_time: '2026-01-01T00:00:00.000Z',
              last_update_time: '2026-01-01T00:00:00.000Z',
            },
          }),
        );
      }),
    );
    return posted;
  };

  const selectView = async (name: RegExp) => {
    await userEvent.click(screen.getByRole('button', { name: /Select a custom view/ }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name }));
  };

  describe('feedback assessment integration', () => {
    it('creates a span-scoped boolean assessment immediately from thumbs feedback', async () => {
      const posted = captureAssessmentPosts();
      setBridge();
      renderWithViews([feedbackPrimitivesView]);
      await selectView(/name-feedback-primitives/);

      await userEvent.click(await screen.findByRole('button', { name: 'Thumbs up' }));

      await waitFor(() => expect(posted).toHaveLength(1));
      expect(posted[0]).toMatchObject({
        assessment_name: 'Helpfulness',
        trace_id: 'tr-test',
        span_id: 'span-1',
        source: { source_type: 'HUMAN', source_id: 'test-user@example.com' },
        feedback: { value: true },
      });
    });

    it('keeps staged form values when thumbs feedback refreshes the view', async () => {
      jest.mocked(shouldUseTracesV4API).mockReturnValue(true);
      const posted = captureAssessmentPosts();
      setBridge();
      renderWithViews([feedbackPrimitivesView]);
      await selectView(/name-feedback-primitives/);

      const accurate = await screen.findByRole('radio', { name: 'Accurate' });
      await userEvent.click(accurate);
      const [rationale, note] = screen.getAllByRole('textbox');
      await userEvent.type(rationale, 'Keep this rationale');
      await userEvent.type(note, 'Keep this note');
      await userEvent.click(screen.getByRole('button', { name: 'Thumbs up' }));

      await waitFor(() => expect(posted).toHaveLength(1));
      await screen.findByText('Feedback submitted');
      expect(screen.getByRole('radio', { name: 'Accurate' })).toBeChecked();
      const [refreshedRationale, refreshedNote] = screen.getAllByRole('textbox');
      expect(refreshedRationale).toHaveValue('Keep this rationale');
      expect(refreshedNote).toHaveValue('Keep this note');

      const submit = screen.getByRole('button', { name: 'Submit review' });
      await waitFor(() => expect(submit).toBeEnabled());
      await userEvent.click(submit);
      await waitFor(() => expect(posted).toHaveLength(3));
      expect(posted.slice(1)).toEqual([
        expect.objectContaining({
          assessment_name: 'Accuracy',
          feedback: { value: 'accurate' },
          rationale: 'Keep this rationale',
        }),
        expect.objectContaining({ assessment_name: 'Notes', feedback: { value: 'Keep this note' } }),
      ]);
    });

    it('reflects a bound thumbs value as the selected button', async () => {
      setBridge();
      renderWithViews([preselectedThumbsView]);
      await selectView(/name-preselected-thumbs/);

      expect(await screen.findByRole('button', { name: 'Thumbs up' })).toHaveAttribute('aria-pressed', 'true');
      expect(screen.getByRole('button', { name: 'Thumbs down' })).toHaveAttribute('aria-pressed', 'false');
    });

    it('submits radio, rationale, and free-text feedback through the assessment endpoint', async () => {
      const posted = captureAssessmentPosts();
      setBridge();
      renderWithViews([feedbackPrimitivesView]);
      await selectView(/name-feedback-primitives/);

      const submit = await screen.findByRole('button', { name: 'Submit review' });
      expect(submit).toBeDisabled();

      await userEvent.click(screen.getByRole('radio', { name: 'Accurate' }));
      const [rationale, note] = screen.getAllByRole('textbox');
      await userEvent.type(rationale, 'The response matches the reference.');
      await userEvent.type(note, 'Clear and concise.');
      expect(submit).toBeEnabled();
      await userEvent.click(submit);

      await waitFor(() => expect(posted).toHaveLength(2));
      expect(posted).toEqual([
        expect.objectContaining({
          assessment_name: 'Accuracy',
          trace_id: 'tr-test',
          feedback: { value: 'accurate' },
          rationale: 'The response matches the reference.',
        }),
        expect.objectContaining({
          assessment_name: 'Notes',
          trace_id: 'tr-test',
          feedback: { value: 'Clear and concise.' },
        }),
      ]);
      expect(await screen.findByRole('button', { name: 'Feedback submitted' })).toBeDisabled();
      expect(screen.getByRole('radio', { name: 'Accurate' })).not.toBeChecked();
      expect(rationale).toHaveValue('');
      expect(note).toHaveValue('');
    });

    it('stages prefilled radio and text values without another user edit', async () => {
      const posted = captureAssessmentPosts();
      setBridge();
      renderWithViews([prefilledFeedbackView]);
      await selectView(/name-prefilled-feedback/);

      expect(await screen.findByRole('radio', { name: 'Accurate' })).toBeChecked();
      expect(screen.getByRole('textbox')).toHaveValue('Prefilled note');
      const submit = screen.getByRole('button', { name: 'Submit prefilled' });
      await waitFor(() => expect(submit).toBeEnabled());
      await userEvent.click(submit);

      await waitFor(() => expect(posted).toHaveLength(2));
      expect(posted).toEqual([
        expect.objectContaining({ assessment_name: 'Accuracy', feedback: { value: 'accurate' } }),
        expect.objectContaining({ assessment_name: 'Notes', feedback: { value: 'Prefilled note' } }),
      ]);
    });

    it('keeps separate radio groups independent when they share an assessment name', async () => {
      setBridge();
      renderWithViews([sameNameFeedbackView]);
      await selectView(/name-same-name-feedback/);

      const first = await screen.findByRole('radio', { name: 'First good' });
      const second = screen.getByRole('radio', { name: 'Second good' });
      await userEvent.click(first);
      await userEvent.click(second);

      expect(first).toBeChecked();
      expect(second).toBeChecked();
    });

    it('clears staged values when the assistant replaces a view template', async () => {
      setBridge();
      renderWithViews([feedbackPrimitivesView], modelTraceInfo, true);
      await selectView(/name-feedback-primitives/);

      await userEvent.click(await screen.findByRole('radio', { name: 'Accurate' }));
      expect(screen.getByRole('button', { name: 'Submit review' })).toBeEnabled();

      await act(async () => {
        await latestOnSpec()(specWithTitle('Replacement'));
      });
      expect(await screen.findByText('Replacement')).toBeInTheDocument();

      await act(async () => {
        await latestOnSpec()({ title: 'Feedback again', messages: feedbackPrimitivesView.template });
      });
      expect(await screen.findByRole('button', { name: 'Submit review' })).toBeDisabled();
    });

    it('keeps a newer edit staged when an older request finishes', async () => {
      const posted: CreateAssessmentPayload['assessment'][] = [];
      let releaseRequest: () => void = () => {};
      const requestGate = new Promise<void>((resolve) => {
        releaseRequest = resolve;
      });
      server.use(
        rest.post('*/ajax-api/*/mlflow/traces/*/assessments', async (req, res, ctx) => {
          const body = (await req.json()) as CreateAssessmentPayload;
          posted.push(body.assessment);
          await requestGate;
          return res(ctx.json({ assessment: { ...body.assessment, assessment_id: 'assessment-1' } }));
        }),
      );
      setBridge();
      renderWithViews([feedbackPrimitivesView]);
      await selectView(/name-feedback-primitives/);

      await userEvent.click(await screen.findByRole('radio', { name: 'Accurate' }));
      await userEvent.click(screen.getByRole('button', { name: 'Submit review' }));
      await waitFor(() => expect(posted).toHaveLength(1));
      await userEvent.click(screen.getByRole('radio', { name: 'Inaccurate' }));
      releaseRequest();

      await screen.findByRole('button', { name: 'Feedback submitted' });
      expect(screen.getByRole('radio', { name: 'Inaccurate' })).toBeChecked();
    });

    it('reports a partial failure and keeps only the failed feedback staged', async () => {
      let requestCount = 0;
      server.use(
        rest.post('*/ajax-api/*/mlflow/traces/*/assessments', async (req, res, ctx) => {
          requestCount += 1;
          const body = (await req.json()) as CreateAssessmentPayload;
          if (requestCount === 2) {
            return res(ctx.status(500), ctx.json({ message: 'failed' }));
          }
          return res(ctx.json({ assessment: { ...body.assessment, assessment_id: 'assessment-1' } }));
        }),
      );
      setBridge();
      renderWithViews([feedbackPrimitivesView]);
      await selectView(/name-feedback-primitives/);

      await userEvent.click(await screen.findByRole('radio', { name: 'Accurate' }));
      const [, note] = screen.getAllByRole('textbox');
      await userEvent.type(note, 'Retry this note');
      await userEvent.click(screen.getByRole('button', { name: 'Submit review' }));

      expect(await screen.findByText('Could not submit feedback. Try again.')).toBeInTheDocument();
      expect(screen.getByRole('radio', { name: 'Accurate' })).not.toBeChecked();
      expect(note).toHaveValue('Retry this note');
      expect(screen.getByRole('button', { name: 'Submit review' })).toBeEnabled();
    });

    it('flushes only the staged controls owned by the clicked form', async () => {
      const posted = captureAssessmentPosts();
      setBridge();
      renderWithViews([twoFormFeedbackView]);
      await selectView(/name-two-forms/);

      await userEvent.click(await screen.findByRole('radio', { name: 'Trace good' }));
      await userEvent.click(screen.getByRole('radio', { name: 'Span bad' }));
      await userEvent.click(screen.getByRole('button', { name: 'Submit trace feedback' }));

      await waitFor(() => expect(posted).toHaveLength(1));
      expect(posted[0]).toMatchObject({
        assessment_name: 'TraceQuality',
        feedback: { value: 'good' },
      });
      expect(posted[0]).not.toHaveProperty('span_id');
      expect(screen.getByRole('button', { name: 'Submit span feedback' })).toBeEnabled();
    });
  });
});
