import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { render } from '@databricks/testing-library';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerCustomView } from './ModelTraceExplorerCustomView';
import { ModelTraceExplorerCustomViewSelector } from '../ModelTraceExplorerCustomViewSelector';
import type { RenderCustomViewSpec } from '../../custom-view/assistant/customViewSpecApplier';
import {
  useCustomViewAssistantBridge,
  type CustomViewAssistantBridge,
} from '../../custom-view/assistant/useCustomViewAssistantBridge';
import {
  CustomViewDefinitionProvider,
  useOptionalCustomViewDefinition,
} from '../../custom-view/CustomViewDefinitionContext';
import { latchDispatchedCustomViewApplyTarget } from '../../custom-view/assistant/customViewAuthoringContext';
import {
  toCustomViewApplyTarget,
  type CustomView,
  MAX_CUSTOM_VIEWS_PER_EXPERIMENT,
} from '../../custom-view/customViewDefinition';
import type { CreateAssessmentPayload } from '../../api';
import type { ModelTrace } from '../ModelTrace.types';
import { QueryClient, QueryClientProvider } from '../../../query-client/queryClient';

// Mock the assistant bridge so the tests drive the component purely off its
// return value (availability / openAssistant / applyError), without the
// module-level Genie registries the real bridge wires up.
jest.mock('../../custom-view/assistant/useCustomViewAssistantBridge', () => ({
  useCustomViewAssistantBridge: jest.fn(),
}));

// Route assessment writes through the V3 endpoint (the feedback-scoping test
// asserts on the request bodies) and pin the assessment author.
jest.mock('../../FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../FeatureUtils')>('../../FeatureUtils'),
  shouldUseTracesV4API: jest.fn(() => false),
  doesTraceSupportV4API: jest.fn(() => false),
  shouldEnableAssessmentsInSessions: jest.fn(() => false),
}));

jest.mock('../../../global-settings/getUser', () => ({
  ...jest.requireActual<typeof import('../../../global-settings/getUser')>('../../../global-settings/getUser'),
  getUser: jest.fn(() => 'test-user@databricks.com'),
}));

const mockUseCustomViewAssistantBridge = jest.mocked(useCustomViewAssistantBridge);

// Minimal legacy trace info: the custom-view builders normalize undefined/legacy
// fields, so no full trace fixture is needed to exercise the authoring UI. The
// default ModelTraceExplorerViewState context supplies an empty nodeMap.
const modelTraceInfo = { request_id: 'tr-test' } as ModelTrace['info'];

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

const CustomViewTestHost = ({ traceInfo }: { traceInfo: ModelTrace['info'] }) => {
  const customViewDefinition = useOptionalCustomViewDefinition();
  const assistantBridge = useCustomViewAssistantBridge({
    data: {} as AgentTraceData,
    onSpec: () => {},
  });
  const [displayMode, setDisplayMode] = React.useState<'default' | 'custom'>('default');

  if (!customViewDefinition) {
    return <ModelTraceExplorerCustomView modelTraceInfo={traceInfo} />;
  }

  return (
    <>
      <ModelTraceExplorerCustomViewSelector
        value={displayMode}
        onValueChange={setDisplayMode}
        onCreateCustomView={() => {
          customViewDefinition.startNewView('');
          setDisplayMode('custom');
        }}
        isCustomViewEnabled
        canCreateCustomView={
          customViewDefinition.canPersist && assistantBridge.isAvailable && !customViewDefinition.hasReachedViewLimit
        }
      />
      <ModelTraceExplorerCustomView modelTraceInfo={traceInfo} />
    </>
  );
};

const customView = () => (
  <IntlProvider locale="en" messages={{}}>
    <DesignSystemProvider>
      <QueryClientProvider client={new QueryClient()}>
        <CustomViewDefinitionProvider views={noViews} isLoaded onPersistView={noopPersistView} canModifyPersistedViews>
          <CustomViewTestHost traceInfo={modelTraceInfo} />
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
// programmatically (simulating a background selection change, e.g. a Genie
// apply) without a DOM click — an outside click would dismiss an open modal.
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
            <CustomViewTestHost traceInfo={modelTraceInfo} />
            <InstructionProbe />
          </CustomViewDefinitionProvider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );

const renderWithPersistProvider = (canModifyPersistedViews: boolean, views: CustomView[] = [persistedView]) => {
  const onPersistView = jest.fn<(view: CustomView) => Promise<void>>().mockResolvedValue(undefined);
  render(
    <IntlProvider locale="en" messages={{}}>
      <DesignSystemProvider>
        <QueryClientProvider client={new QueryClient()}>
          <CustomViewDefinitionProvider
            views={views}
            isLoaded
            onPersistView={onPersistView}
            onDeleteView={jest.fn<() => Promise<void>>().mockResolvedValue(undefined)}
            canModifyPersistedViews={canModifyPersistedViews}
          >
            <CustomViewTestHost traceInfo={modelTraceInfo} />
          </CustomViewDefinitionProvider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onPersistView };
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
            <CustomViewTestHost traceInfo={modelTraceInfo} />
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
          <CustomViewDefinitionProvider
            views={views}
            isLoaded
            onPersistView={onPersistView}
            onDeleteView={jest.fn<() => Promise<void>>().mockResolvedValue(undefined)}
            canModifyPersistedViews
          >
            <CustomViewTestHost traceInfo={modelTraceInfo} />
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
// template and the user rebuilds the view with Genie.
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

// A view whose surface holds TWO independent feedback forms in one column,
// separated by `formId`: a trace-level rating + submit ("form-trace", no spanId)
// and a span-level rating + submit ("form-span", controls scoped to `span-1`).
// Used to prove each Submit button flushes only its own form's staged feedback.
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
            label: 'Trace rating',
            name: 'TraceQuality',
            formId: 'form-trace',
            options: [
              { label: 'Trace good', value: 'good' },
              { label: 'Trace bad', value: 'bad' },
            ],
          },
          { id: 'trace-submit', component: 'FeedbackSubmit', label: 'Submit trace feedback', formId: 'form-trace' },
          {
            id: 'span-rating',
            component: 'RadioGroup',
            label: 'Span rating',
            name: 'SpanQuality',
            spanId: 'span-1',
            formId: 'form-span',
            options: [
              { label: 'Span good', value: 'good' },
              { label: 'Span bad', value: 'bad' },
            ],
          },
          { id: 'span-submit', component: 'FeedbackSubmit', label: 'Submit span feedback', formId: 'form-span' },
        ],
      },
    },
  ],
});

// A cross-span, multi-dimensional form with one Submit. This mirrors the
// reported tool-call-card layout: two dimensions per span, each with a same-name
// rationale field, and one button below both cards. All controls and the submit
// share ONE `formId` ("cross") — that is what makes the single button flush both
// spans' ratings; the controls differ only by `spanId`.
const crossSpanFeedbackView = makeView('cross-span', {
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [
          {
            id: 'root',
            component: 'Column',
            children: [
              'accuracy-1',
              'rationale-accuracy-1',
              'completeness-1',
              'rationale-completeness-1',
              'accuracy-2',
              'rationale-accuracy-2',
              'completeness-2',
              'rationale-completeness-2',
              'submit',
            ],
          },
          {
            id: 'accuracy-1',
            component: 'RadioGroup',
            label: 'Span 1 accuracy',
            name: 'accuracy_span-1',
            spanId: 'span-1',
            formId: 'cross',
            options: [{ label: 'Span 1 mostly accurate', value: 'mostly_accurate' }],
          },
          {
            id: 'rationale-accuracy-1',
            component: 'FeedbackInputText',
            label: 'Span 1 accuracy rationale',
            name: 'accuracy_span-1',
            spanId: 'span-1',
            formId: 'cross',
          },
          {
            id: 'completeness-1',
            component: 'RadioGroup',
            label: 'Span 1 completeness',
            name: 'completeness_span-1',
            spanId: 'span-1',
            formId: 'cross',
            options: [{ label: 'Span 1 mostly complete', value: 'mostly_complete' }],
          },
          {
            id: 'rationale-completeness-1',
            component: 'FeedbackInputText',
            label: 'Span 1 completeness rationale',
            name: 'completeness_span-1',
            spanId: 'span-1',
            formId: 'cross',
          },
          {
            id: 'accuracy-2',
            component: 'RadioGroup',
            label: 'Span 2 accuracy',
            name: 'accuracy_span-2',
            spanId: 'span-2',
            formId: 'cross',
            options: [{ label: 'Span 2 less accurate', value: 'less_accurate' }],
          },
          {
            id: 'rationale-accuracy-2',
            component: 'FeedbackInputText',
            label: 'Span 2 accuracy rationale',
            name: 'accuracy_span-2',
            spanId: 'span-2',
            formId: 'cross',
          },
          {
            id: 'completeness-2',
            component: 'RadioGroup',
            label: 'Span 2 completeness',
            name: 'completeness_span-2',
            spanId: 'span-2',
            formId: 'cross',
            options: [{ label: 'Span 2 not complete', value: 'not_complete' }],
          },
          {
            id: 'rationale-completeness-2',
            component: 'FeedbackInputText',
            label: 'Span 2 completeness rationale',
            name: 'completeness_span-2',
            spanId: 'span-2',
            formId: 'cross',
          },
          { id: 'submit', component: 'FeedbackSubmit', label: 'Submit feedback', formId: 'cross' },
        ],
      },
    },
  ],
});

// A view surfacing the assessment count (a StatCard bound to metrics.assessments)
// and the AssessmentBoard (bound to the assessments source). Used to prove both
// stay live after feedback lands in the trace-cached-actions store.
const assessmentSummaryView = makeView('assessments', {
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [
          { id: 'root', component: 'Column', children: ['count', 'board'] },
          { id: 'count', component: 'StatCard', value: { $source: 'metrics.assessments' }, label: 'Assessments' },
          {
            id: 'board',
            component: 'AssessmentBoard',
            title: 'Assessments',
            children: { $source: 'assessments' },
            emptyMessage: 'No assessments yet',
          },
        ],
      },
    },
  ],
});

const renderWithViews = (views: CustomView[], traceInfo: ModelTrace['info'] = modelTraceInfo) =>
  render(
    <IntlProvider locale="en" messages={{}}>
      <DesignSystemProvider>
        <QueryClientProvider client={new QueryClient()}>
          <CustomViewDefinitionProvider views={views} isLoaded>
            <CustomViewTestHost traceInfo={traceInfo} />
          </CustomViewDefinitionProvider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );

const selectPersistedViewAndDirty = async () => {
  await userEvent.click(screen.getByRole('button', { name: /Default view/ }));
  await userEvent.click(screen.getByRole('menuitemradio', { name: /name-v1/ }));
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
    // The latch is module-level state owned by the page's context plugin; clear
    // it so a test that sets it can't retarget a later test's apply.
    latchDispatchedCustomViewApplyTarget(undefined);
  });

  it('renders the authoring prompt UI when the assistant bridge is available', () => {
    setBridge();
    renderCustomView();

    expect(screen.getByText('Build a custom trace view')).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Build with Genie/ })).toBeInTheDocument();
  });

  it('disables the build button until the user types a prompt', async () => {
    setBridge();
    renderCustomView();

    const buildButton = screen.getByRole('button', { name: /Build with Genie/ });
    expect(buildButton).toBeDisabled();

    // eslint-disable-next-line @databricks/no-userevent-type
    await userEvent.type(screen.getByRole('textbox'), 'Show me the failed spans');
    expect(buildButton).toBeEnabled();
  });

  it('hands the typed prompt to the assistant and shows the building skeleton on submit', async () => {
    const openAssistant = jest.fn();
    setBridge({ openAssistant });
    renderCustomView();

    // eslint-disable-next-line @databricks/no-userevent-type
    await userEvent.type(screen.getByRole('textbox'), 'Show me the failed spans');
    await userEvent.click(screen.getByRole('button', { name: /Build with Genie/ }));

    expect(openAssistant).toHaveBeenCalledTimes(1);
    expect(openAssistant).toHaveBeenCalledWith('Show me the failed spans', { newSession: true });
    // The empty-state prompt box is replaced by the loading skeleton while Genie builds.
    expect(screen.getByText('Building this view…')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Build with Genie/ })).not.toBeInTheDocument();
  });

  it('keeps the building skeleton after streaming ends until a view is created', () => {
    const openAssistant = jest.fn();
    setBridge({ openAssistant, isStreaming: false });
    const { rerender } = renderCustomView();

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Show me the failed spans' } });
    fireEvent.click(screen.getByRole('button', { name: /Build with Genie/ }));
    expect(screen.getByText('Building this view…')).toBeInTheDocument();

    // Streaming starts: the skeleton stays.
    setBridge({ openAssistant, isStreaming: true });
    rerender(customView());
    expect(screen.getByText('Building this view…')).toBeInTheDocument();

    // Streaming finishes but render_custom_view has not applied a view yet. The
    // skeleton must NOT clear on the streaming falling edge (that flashed the
    // prompt back mid-build) — it stays until activeView or applyError.
    setBridge({ openAssistant, isStreaming: false });
    rerender(customView());
    expect(screen.getByText('Building this view…')).toBeInTheDocument();
  });

  it('does not revert to the prompt after a start timeout (no timeout guard)', () => {
    jest.useFakeTimers();
    try {
      setBridge({ isStreaming: false });
      renderCustomView();

      fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Show me the failed spans' } });
      fireEvent.click(screen.getByRole('button', { name: /Build with Genie/ }));
      expect(screen.getByText('Building this view…')).toBeInTheDocument();

      // The MLflow connector never reports streaming, so the old 15s start-timeout
      // would have flashed the prompt back before render_custom_view completed.
      // With the guard removed, the skeleton persists past any such window.
      act(() => {
        jest.advanceTimersByTime(60_000);
      });

      expect(screen.getByText('Building this view…')).toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /Build with Genie/ })).not.toBeInTheDocument();
    } finally {
      jest.useRealTimers();
    }
  });

  it('clears the building skeleton and surfaces the error when the spec apply fails', () => {
    const openAssistant = jest.fn();
    setBridge({ openAssistant });
    const { rerender } = renderCustomView();

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Show me the failed spans' } });
    fireEvent.click(screen.getByRole('button', { name: /Build with Genie/ }));
    expect(screen.getByText('Building this view…')).toBeInTheDocument();

    // The render_custom_view apply fails: the skeleton clears, the authoring UI
    // returns, and the inline assistant error is shown.
    setBridge({ openAssistant, applyError: 'Unknown component "Widget"' });
    rerender(customView());

    expect(screen.queryByText('Building this view…')).not.toBeInTheDocument();
    expect(screen.getByText('Assistant: Unknown component "Widget"')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Build with Genie/ })).toBeInTheDocument();
  });

  it('keeps the typed prompt and does not enter the building state when the launcher throws', async () => {
    const throwingOpenAssistant = jest.fn(() => {
      throw new Error('launch failed');
    });
    setBridge({ openAssistant: throwingOpenAssistant });
    renderCustomView();

    // eslint-disable-next-line @databricks/no-userevent-type
    await userEvent.type(screen.getByRole('textbox'), 'Show me the failed spans');
    await userEvent.click(screen.getByRole('button', { name: /Build with Genie/ }));

    expect(throwingOpenAssistant).toHaveBeenCalledTimes(1);
    // The prompt is preserved for retry and the skeleton never appears.
    expect(screen.getByRole('textbox')).toHaveValue('Show me the failed spans');
    expect(screen.queryByText('Building this view…')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Build with Genie/ })).toBeInTheDocument();
  });

  it('records the submitted prompt as the built view instruction', async () => {
    setBridge();
    renderWithProvider();

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Show me the failed spans' } });
    fireEvent.click(screen.getByRole('button', { name: /Build with Genie/ }));

    // The render_custom_view tool applies the agent's spec.
    await act(async () => {
      await latestOnSpec()(specWithTitle('Failed spans'));
    });

    // The view saves the prompt that launched it, not an empty instruction.
    expect(screen.getByTestId('active-instruction')).toHaveTextContent('Show me the failed spans');
  });

  it('keeps the prior instruction for a Genie edit that sends no empty-state prompt', async () => {
    setBridge();
    renderWithProvider();

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'First prompt' } });
    fireEvent.click(screen.getByRole('button', { name: /Build with Genie/ }));
    await act(async () => {
      await latestOnSpec()(specWithTitle('Initial view'));
    });
    expect(screen.getByTestId('active-instruction')).toHaveTextContent('First prompt');

    // "Edit with Genie" opens the panel with no empty-state prompt — the edit
    // request is typed inside Genie, which the host never sees.
    fireEvent.click(screen.getByRole('button', { name: /Edit with Genie/ }));
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
    expect(screen.queryByRole('button', { name: /Build with Genie/ })).not.toBeInTheDocument();
  });

  it('hides create, edit, save, and delete controls when the user cannot modify custom views', async () => {
    setBridge();
    renderWithPersistProvider(false);

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Default view/ }));
    expect(screen.queryByRole('menuitem', { name: /Create custom view/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole('menuitemradio', { name: /name-v1/ }));

    expect(() => latestOnSpec()(specWithTitle('Blocked update'))).toThrow(
      'Custom views cannot be modified in this experiment.',
    );
    expect(screen.queryByRole('button', { name: 'Save' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Edit with Genie/ })).not.toBeInTheDocument();
    // The overflow menu is absent, so Rename and Delete (which live inside it)
    // are unreachable for read-only users.
    expect(screen.queryByRole('button', { name: 'More view options' })).not.toBeInTheDocument();
  });

  it('shows a non-authoring empty state to read-only users when there are no saved views', () => {
    setBridge();
    renderWithPersistProvider(false, []);

    expect(screen.getByText('No custom views')).toBeInTheDocument();
    expect(screen.queryByText('Build a custom trace view')).not.toBeInTheDocument();
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Build with Genie/ })).not.toBeInTheDocument();
    expect(mockUseCustomViewAssistantBridge).toHaveBeenLastCalledWith(expect.objectContaining({ enabled: false }));
  });

  it('shows saved views instead of the no-views state when edit permission is lost during a draft', async () => {
    setBridge();
    const { rerenderWithPermission } = renderWithChangingPermission(true);
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: /Default view/ }));
    await user.click(screen.getByRole('menuitem', { name: /Create custom view/ }));
    expect(screen.getByRole('button', { name: /Custom view/ })).toBeInTheDocument();

    rerenderWithPermission(false);

    expect(screen.getByRole('button', { name: /Custom view/ })).toBeInTheDocument();
    expect(screen.getByText('Choose a saved view from the menu to render it for this trace.')).toBeInTheDocument();
    expect(screen.queryByText('No custom views')).not.toBeInTheDocument();
  });

  it('hides Create custom view when the assistant bridge is unavailable', async () => {
    setBridge({ isAvailable: false, openAssistant: undefined });
    renderWithPersistProvider(true);

    await userEvent.click(screen.getByRole('button', { name: /Default view/ }));

    expect(screen.queryByRole('menuitem', { name: /Create custom view/ })).not.toBeInTheDocument();
  });

  it('shows Save and Delete when the user can modify persisted views', async () => {
    setBridge();
    renderWithPersistProvider(true);

    await selectPersistedViewAndDirty();

    expect(screen.getByRole('button', { name: 'Save' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Edit with Genie/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'More view options' })).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /name-v1/ }));
    expect(screen.getByRole('menuitem', { name: /Create custom view/ })).toBeInTheDocument();
  });

  it('shows Rename view in the overflow menu for editors', async () => {
    setBridge();
    renderWithPersistProvider(true);

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Default view/ }));
    await user.click(screen.getByRole('menuitemradio', { name: /name-v1/ }));
    await user.click(screen.getByRole('button', { name: 'More view options' }));

    // A readable view: rename is enabled (not aria-disabled).
    expect(screen.getByRole('menuitem', { name: /Rename view/ })).not.toHaveAttribute('aria-disabled', 'true');
    expect(screen.getByRole('menuitem', { name: /Delete view/ })).toBeInTheDocument();
  });

  it('disables (not hides) Rename view for an unreadable persisted view, keeping Delete available', async () => {
    setBridge();
    // An unreadable persisted view (its saved definition can't be rendered). Rename
    // must stay visible-but-disabled (with an explanatory tooltip) so the user
    // learns why, while Delete stays available to remove the broken view.
    renderWithPersistProvider(true, [makeView('bad', { unreadable: true })]);

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Default view/ }));
    await user.click(screen.getByRole('menuitemradio', { name: /name-bad/ }));
    await user.click(screen.getByRole('button', { name: 'More view options' }));

    // Rename is rendered but disabled (not removed from the menu).
    expect(screen.getByRole('menuitem', { name: /Rename view/ })).toHaveAttribute('aria-disabled', 'true');
    // Delete stays enabled so the broken view can still be removed.
    expect(screen.getByRole('menuitem', { name: /Delete view/ })).not.toHaveAttribute('aria-disabled', 'true');
  });

  it('disables Rename for a valid-shape view whose template fails validation (Case-2 unreadable derived at selection)', async () => {
    setBridge();
    // A persisted view with a valid CustomView shape but an invalid template
    // (forbidden `#span:` narrative). It is NOT flagged `unreadable` at load
    // anymore (template validation is deferred), so this proves the Case-2 gating
    // is derived when the view becomes active: Rename is disabled, Delete stays
    // available, and the render placeholder shows.
    renderWithPersistProvider(true, [tamperedView]);

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Default view/ }));
    await user.click(screen.getByRole('menuitemradio', { name: /name-tampered/ }));

    expect(await screen.findByText(/couldn't be read and can't be displayed/)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'More view options' }));
    expect(screen.getByRole('menuitem', { name: /Rename view/ })).toHaveAttribute('aria-disabled', 'true');
    expect(screen.getByRole('menuitem', { name: /Delete view/ })).not.toHaveAttribute('aria-disabled', 'true');
  });

  it('renames the selected view: prefills the current name and persists the trimmed new name against the saved template', async () => {
    setBridge();
    const { onPersistView } = renderWithPersistProvider(true);

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Default view/ }));
    await user.click(screen.getByRole('menuitemradio', { name: /name-v1/ }));
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
    await user.click(screen.getByRole('button', { name: /Default view/ }));
    await user.click(screen.getByRole('menuitemradio', { name: /name-v1/ }));
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
              onDeleteView={jest.fn<() => Promise<void>>().mockResolvedValue(undefined)}
              canModifyPersistedViews
            >
              <CustomViewTestHost traceInfo={modelTraceInfo} />
              <SelectViewProbe selectViewRef={selectViewRef} />
            </CustomViewDefinitionProvider>
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );

    const user = userEvent.setup();
    // Select v1 and open its rename modal (captures v1 as the target).
    await user.click(screen.getByRole('button', { name: /Default view/ }));
    await user.click(screen.getByRole('menuitemradio', { name: /name-v1/ }));
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

  it('applies a Genie edit to the view the request was made against, not the one selected mid-flight', async () => {
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
              <CustomViewTestHost traceInfo={modelTraceInfo} />
              <SelectViewProbe selectViewRef={selectViewRef} />
            </CustomViewDefinitionProvider>
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Default view/ }));
    await user.click(screen.getByRole('menuitemradio', { name: /name-v1/ }));

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

    // v1 received the edit in the background.
    await user.click(screen.getByRole('button', { name: /name-v2/ }));
    const v1Item = screen.getByRole('menuitemradio', { name: /name-v1/ });

    await user.click(v1Item);
    expect(screen.getByText('Updated label')).toBeInTheDocument();
  });

  it('drops a Genie edit whose target view was deleted mid-flight instead of resurrecting it', async () => {
    setBridge();
    const viewA = makeView('v1');
    const viewB = makeView('v2');
    render(
      <IntlProvider locale="en" messages={{}}>
        <DesignSystemProvider>
          <QueryClientProvider client={new QueryClient()}>
            <CustomViewDefinitionProvider
              views={[viewA, viewB]}
              isLoaded
              onPersistView={noopPersistView}
              onDeleteView={jest.fn<() => Promise<void>>().mockResolvedValue(undefined)}
              canModifyPersistedViews
            >
              <CustomViewTestHost traceInfo={modelTraceInfo} />
            </CustomViewDefinitionProvider>
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Default view/ }));
    await user.click(screen.getByRole('menuitemradio', { name: /name-v1/ }));

    // The page context plugin latches v1 as this turn's prompt is assembled.
    act(() => latchDispatchedCustomViewApplyTarget(toCustomViewApplyTarget(viewA)));

    // The user deletes v1 while the agent is still running.
    await user.click(screen.getByRole('button', { name: /More view options/ }));
    await user.click(screen.getByRole('menuitem', { name: /Delete view/ }));
    await user.click(screen.getByRole('button', { name: 'Delete' }));
    // The delete is fire-and-forget, so wait for the cleared selection before
    // driving the apply.
    expect(await screen.findByRole('button', { name: /Custom view/ })).toBeInTheDocument();

    // The reply targets a view that no longer exists. The apply carries a full
    // snapshot of that view, so writing it would recreate what the user just
    // deleted — under the same id and name, and auto-selected, because deletion
    // left nothing selected. It must fail instead, loudly enough for the bridge to
    // report it (covered in useCustomViewAssistantBridge.test.tsx).
    expect(() => latestOnSpec()(specWithTitle('Updated label'))).toThrow(
      /"name-v1" was deleted while the assistant was working/,
    );

    // v1 stays gone and nothing is selected; only v2 remains in the switcher.
    await user.click(screen.getByRole('button', { name: /Custom view/ }));
    expect(screen.queryByRole('menuitemradio', { name: /name-v1/ })).toBeNull();
    expect(screen.getByRole('menuitemradio', { name: /name-v2/ })).toBeInTheDocument();
  });

  it('still builds a brand-new view whose reserved id is absent from the working set', async () => {
    // The counterpart to the deleted-view test above: a new build's target id is
    // deliberately not in `views` yet, so the refusal above must key on deletion
    // and never on "this id isn't in views" — that check would break creation.
    setBridge();
    const { onPersistView } = renderWithPersist([]);

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Show failed spans' } });
    fireEvent.click(screen.getByRole('button', { name: /Build with Genie/ }));
    await act(async () => {
      await latestOnSpec()(specWithTitle('Failed spans'));
    });

    expect(screen.getByText('Failed spans')).toBeInTheDocument();
    expect(onPersistView).not.toHaveBeenCalled();
  });

  it('preserves the saved view name when Genie applies without a launch binding', async () => {
    setBridge();
    renderWithPersistProvider(true);

    await userEvent.click(screen.getByRole('button', { name: /Default view/ }));
    await userEvent.click(screen.getByRole('menuitemradio', { name: /name-v1/ }));
    await act(async () => {
      await latestOnSpec()(specWithTitle('Updated label'));
    });

    // Dropdown trigger shows the user-provided name, not "Untitled view".
    expect(screen.getByRole('button', { name: /name-v1/ })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(screen.queryByRole('dialog', { name: /Name this custom view/ })).not.toBeInTheDocument();
  });

  it('takes the user straight to the draft authoring UI when Create custom view is clicked, with no naming modal', async () => {
    setBridge();
    renderWithPersistProvider(true);

    await userEvent.click(screen.getByRole('button', { name: /Default view/ }));
    await userEvent.click(screen.getByRole('menuitem', { name: /Create custom view/ }));

    // No up-front naming modal — the user lands directly in the authoring UI.
    expect(screen.queryByRole('dialog', { name: /Name this custom view/ })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Build with Genie/ })).toBeInTheDocument();
    // The switcher shows the unsaved-view fallback label.
    expect(screen.getByRole('button', { name: /Custom view/ })).toBeInTheDocument();
  });

  it('hides Create custom view at the per-experiment view limit', async () => {
    setBridge();
    const maxViews = Array.from({ length: MAX_CUSTOM_VIEWS_PER_EXPERIMENT }, (_unused, index) =>
      makeView(`limit-${index}`),
    );
    renderWithPersistProvider(true, maxViews);

    await userEvent.click(screen.getByRole('button', { name: /Default view/ }));
    expect(screen.queryByRole('menuitem', { name: /Create custom view/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Build with Genie/ })).not.toBeInTheDocument();
  });

  it('keeps Create custom view enabled one view below the limit', async () => {
    setBridge();
    const belowLimit = Array.from({ length: MAX_CUSTOM_VIEWS_PER_EXPERIMENT - 1 }, (_unused, index) =>
      makeView(`limit-${index}`),
    );
    renderWithPersistProvider(true, belowLimit);

    await userEvent.click(screen.getByRole('button', { name: /Default view/ }));

    expect(screen.getByRole('menuitem', { name: /Create custom view/ })).not.toHaveAttribute('aria-disabled', 'true');
  });

  it('still allows editing and saving an existing view at the limit', async () => {
    setBridge();
    const maxViews = Array.from({ length: MAX_CUSTOM_VIEWS_PER_EXPERIMENT }, (_unused, index) =>
      makeView(`limit-${index}`),
    );
    const { onPersistView } = renderWithPersistProvider(true, maxViews);

    // Re-saving an existing view overwrites its tag, so the cap must not lock the user
    // out of their own views.
    await userEvent.click(screen.getByRole('button', { name: /Default view/ }));
    await userEvent.click(screen.getByRole('menuitemradio', { name: /name-limit-0/ }));
    await act(async () => {
      await latestOnSpec()(specWithTitle('Edited at the cap'));
    });
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() => expect(onPersistView).toHaveBeenCalledWith(expect.objectContaining({ id: 'limit-0' })));
  });

  it('prompts for a name on the first save of a newly built view and persists with it', async () => {
    setBridge();
    const { onPersistView } = renderWithPersist([]);

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Show failed spans' } });
    fireEvent.click(screen.getByRole('button', { name: /Build with Genie/ }));
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
    fireEvent.click(screen.getByRole('button', { name: /Build with Genie/ }));
    await act(async () => {
      await latestOnSpec()(specWithTitle('Failed spans'));
    });

    expect(screen.getByRole('button', { name: /Default view/ })).toBeInTheDocument();
    // The unsaved-changes banner is shown for the never-saved view.
    expect(screen.getByText(/unsaved changes/)).toBeInTheDocument();
  });

  it('marks an edited saved view as a draft in the trigger, the list item, and with a banner', async () => {
    setBridge();
    renderWithPersistProvider(true);

    await selectPersistedViewAndDirty();

    // The selected view remains visible while its unsaved-changes banner is shown.
    expect(screen.getByRole('button', { name: /name-v1/ })).toBeInTheDocument();
    expect(screen.getByText(/unsaved changes/)).toBeInTheDocument();
  });

  it('does not mark a clean saved view as a draft', async () => {
    setBridge();
    renderWithPersistProvider(true);

    await userEvent.click(screen.getByRole('button', { name: /Default view/ }));
    await userEvent.click(screen.getByRole('menuitemradio', { name: /name-v1/ }));

    // Selecting without editing leaves the view clean: no draft marker, no banner.
    expect(screen.getByRole('button', { name: /name-v1/ })).toBeInTheDocument();
    expect(screen.queryByText('(Draft)')).not.toBeInTheDocument();
    expect(screen.queryByText(/unsaved changes/)).not.toBeInTheDocument();
  });

  it('renders the inline error placeholder for a tampered template instead of the raw sink', async () => {
    setBridge();
    renderWithViews([tamperedView]);

    await userEvent.click(screen.getByRole('button', { name: /Default view/ }));
    await userEvent.click(screen.getByRole('menuitemradio', { name: /name-tampered/ }));

    // The re-bind gate rejects the tampered template and shows the safe inline
    // placeholder rather than binding/rendering its untrusted content.
    expect(await screen.findByText(/couldn't be read and can't be displayed/)).toBeInTheDocument();
    expect(screen.queryByText(/jump to #span:abc123/)).not.toBeInTheDocument();
  });

  it('renders the inline error placeholder for a view saved with the removed TreeView component', async () => {
    setBridge();
    renderWithViews([legacyTreeView]);

    await userEvent.click(screen.getByRole('button', { name: /Default view/ }));
    await userEvent.click(screen.getByRole('menuitemradio', { name: /name-legacy-tree/ }));

    expect(await screen.findByText(/couldn't be read and can't be displayed/)).toBeInTheDocument();
    expect(screen.queryByText('Span Tree')).not.toBeInTheDocument();
  });

  it('renders the inline error placeholder for a view saved with the removed DataTable component', async () => {
    setBridge();
    renderWithViews([legacyDataTableView]);

    await userEvent.click(screen.getByRole('button', { name: /Default view/ }));
    await userEvent.click(screen.getByRole('menuitemradio', { name: /name-legacy-table/ }));

    expect(await screen.findByText(/couldn't be read and can't be displayed/)).toBeInTheDocument();
    expect(screen.queryByText('Tool Performance')).not.toBeInTheDocument();
  });
});
