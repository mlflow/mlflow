import { jest, beforeAll, afterAll, describe, it, expect } from '@jest/globals';
import { screen, within, waitForElementToBeRemoved, waitFor } from '@testing-library/react';
import { render } from '@databricks/testing-library';
import userEvent from '@testing-library/user-event';
import { cloneDeep } from 'lodash';
import { rest } from 'msw';

import { ProvidersWrapper } from '../../test-utils/testUtilProviderWrappers';

import type { ModelTrace, ModelTraceInfo, ModelTraceSpanV2 } from './ModelTrace.types';
import { ModelTraceExplorer } from './ModelTraceExplorer';
import {
  MOCK_ASSESSMENT,
  MOCK_CHAT_TOOL_CALL_SPAN,
  MOCK_EVENTS_SPAN,
  MOCK_EXPECTATION,
  MOCK_SPAN_ASSESSMENT,
  MOCK_TRACE,
  MOCK_V3_TRACE,
  MOCK_TRACE_INFO_V3,
  MOCK_V3_SPANS,
} from '../ModelTraceExplorer.test-utils';
import { AssessmentSchemaContextProvider } from '../contexts/AssessmentSchemaContext';
import { ModelTraceExplorerPreferencesProvider } from './ModelTraceExplorerPreferencesContext';
import { ModelTraceExplorerContextProvider } from './ModelTraceExplorerContext';

const LINKED_PROMPTS_TAGS = {
  'mlflow.linkedPrompts': JSON.stringify([
    { name: 'customer-support', version: '12' },
    { name: 'request-router', version: '4' },
  ]),
};

const MOCK_V3_TRACE_WITH_LINKED_PROMPTS: ModelTrace = {
  data: { spans: MOCK_V3_SPANS },
  info: {
    ...MOCK_TRACE_INFO_V3,
    tags: LINKED_PROMPTS_TAGS,
  },
};

// increase timeout and it's a heavy test
// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000);

// mock the scrollIntoView function to prevent errors
window.HTMLElement.prototype.scrollIntoView = jest.fn();

// @xyflow/react reads m22 (zoom) from DOMMatrixReadOnly. JSDOM does not implement it.
Object.defineProperty(global, 'DOMMatrixReadOnly', {
  value: function DOMMatrixReadOnly() {
    Object.defineProperty(this, 'm22', { value: 1, writable: true });
  },
  writable: true,
  configurable: true,
});

jest.mock('../FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../FeatureUtils')>('../FeatureUtils'),
  shouldEnableTracesTabLabelingSchemas: jest.fn().mockReturnValue(false),
}));
jest.mock('../FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../FeatureUtils')>('../FeatureUtils'),
  shouldUseTracesV4API: jest.fn().mockReturnValue(false),
}));
jest.mock('../hooks/useGetModelTraceInfo', () => ({
  useGetModelTraceInfo: jest.fn().mockReturnValue({
    refetch: jest.fn(),
  }),
}));

// Since working ResizeObserver is a hard requirement for Gantt chart, let's mock it
let originalResizeObserver: typeof ResizeObserver;
beforeAll(() => {
  originalResizeObserver = globalThis.ResizeObserver;
  const mockedRect = {
    x: 0,
    y: 0,
    width: 1000,
    height: 100,
    top: 0,
    right: 1000,
    bottom: 100,
    left: 0,
  } as DOMRectReadOnly;

  globalThis.ResizeObserver = class MockResizeObserver {
    observerCallback: ResizeObserverCallback;
    targets: Element[];
    constructor(callback: ResizeObserverCallback) {
      this.observerCallback = callback;
      this.targets = [];
    }

    observe = (element: Element) => {
      this.targets.push(element);

      this.observerCallback(
        this.targets.map((target) => ({
          target,
          borderBoxSize: [{ inlineSize: mockedRect.width, blockSize: mockedRect.height }],
          contentBoxSize: [{ inlineSize: mockedRect.width, blockSize: mockedRect.height }],
          contentRect: mockedRect,
          devicePixelContentBoxSize: [{ inlineSize: mockedRect.width, blockSize: mockedRect.height }],
        })),
        this,
      );
    };

    unobserve = (element: Element) => {
      this.targets = this.targets.filter((target) => target !== element);
    };

    disconnect = () => {
      this.targets.length = 0;
    };
  };
});

afterAll(() => {
  globalThis.ResizeObserver = originalResizeObserver;
});

const TestComponent = ({
  modelTrace,
  isSearchVisible = false,
  enableGraphView,
  collapseAssessmentPane,
  explorerKey,
}: {
  modelTrace: ModelTrace;
  isSearchVisible?: boolean;
  enableGraphView?: boolean;
  collapseAssessmentPane?: boolean | 'force-open';
  explorerKey?: string;
}) => {
  return (
    <ProvidersWrapper>
      <ModelTraceExplorerPreferencesProvider initialRenderMode="default">
        <ModelTraceExplorerContextProvider isSearchVisible={isSearchVisible}>
          <ModelTraceExplorer
            key={explorerKey}
            modelTrace={modelTrace}
            enableGraphView={enableGraphView}
            collapseAssessmentPane={collapseAssessmentPane}
          />
        </ModelTraceExplorerContextProvider>
      </ModelTraceExplorerPreferencesProvider>
    </ProvidersWrapper>
  );
};

// ui-test-level: integration. The observable combines the explorer view-state, header, toggle, and preferences providers.
describe('ModelTraceExplorer', () => {
  it('opens linked prompts from the selected root span metadata', async () => {
    const user = userEvent.setup();
    render(<TestComponent modelTrace={MOCK_V3_TRACE_WITH_LINKED_PROMPTS} />);

    await user.click(screen.getByRole('button', { name: '2 linked prompts in trace metadata' }));

    const customerSupportPrompt = await screen.findByRole('link', { name: 'customer-support' });
    expect(customerSupportPrompt).toHaveAttribute(
      'href',
      expect.stringContaining('/experiments/3363486573189371/prompts/customer-support?promptVersion=12'),
    );
    expect(customerSupportPrompt).toHaveAttribute('target', '_blank');
    expect(screen.getByText('Version 12')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'request-router' })).toHaveAttribute('target', '_blank');
    expect(screen.getByText('Version 4')).toBeInTheDocument();
  });

  it('hides linked prompts metadata when a child span is selected', async () => {
    const user = userEvent.setup();
    render(<TestComponent modelTrace={MOCK_V3_TRACE_WITH_LINKED_PROMPTS} />);

    expect(screen.getByRole('button', { name: '2 linked prompts in trace metadata' })).toBeInTheDocument();
    await user.click(screen.getByText('rephrase_chat_to_queue'));

    expect(screen.queryByRole('button', { name: '2 linked prompts in trace metadata' })).not.toBeInTheDocument();
  });

  it.each<{ caseName: string; tags: Record<string, string> }>([
    { caseName: 'missing metadata', tags: {} },
    { caseName: 'malformed metadata', tags: { 'mlflow.linkedPrompts': 'invalid-json' } },
  ])('renders no linked prompt metadata for $caseName', ({ tags }) => {
    const trace: ModelTrace = {
      data: { spans: MOCK_V3_SPANS },
      info: { ...MOCK_TRACE_INFO_V3, tags },
    };

    render(<TestComponent modelTrace={trace} />);

    expect(screen.queryByRole('button', { name: '2 linked prompts in trace metadata' })).not.toBeInTheDocument();
  });

  it.each([MOCK_TRACE, MOCK_V3_TRACE])(
    'renders the component and allows to inspect selected spans',
    async (trace: ModelTrace) => {
      render(<TestComponent modelTrace={trace} />);

      // Assert existence of the span tree header
      expect(screen.getByText('Spans')).toBeInTheDocument();

      // Expect timeline view to be closed at first (due to JSDOM's 1024 default screen width)
      expect(screen.queryByTestId('time-marker-area')).not.toBeInTheDocument();
      await userEvent.click(screen.getByRole('button', { name: 'Show execution timeline' }));

      // Assert existence of all calculated time spans
      expect(within(screen.getByTestId('time-marker-area')).getByText('26.00s')).toBeInTheDocument();
      expect(within(screen.getByTestId('time-marker-area')).getByText('10.00s')).toBeInTheDocument();
      expect(within(screen.getByTestId('time-marker-area')).getByText('0s')).toBeInTheDocument();

      // Check if the default input is rendered
      expect(screen.getByText('document-qa-chain-input')).toBeInTheDocument();

      // Switch to another span
      await userEvent.click(screen.getAllByText('rephrase_chat_to_queue')[0]);

      // Check if the new input is rendered
      expect(screen.getByText('rephrase_chat_to_queue-input')).toBeInTheDocument();
    },
  );

  it('filters the tree based on the search string', async () => {
    const user = userEvent.setup({ delay: null });
    render(<TestComponent modelTrace={MOCK_TRACE} isSearchVisible />);

    // de-select "show parents" button so the rest of the test is easier to run
    await userEvent.click(screen.getByRole('button', { name: 'Trace display settings' }));
    await userEvent.hover(screen.getByRole('menuitem', { name: 'Filter by span type' }));
    const showParentsCheckbox = await screen.findByRole('menuitemcheckbox', { name: 'Show all parent spans' });
    await userEvent.click(showParentsCheckbox);
    await waitFor(() => expect(showParentsCheckbox).not.toBeChecked());
    await userEvent.keyboard('{Escape}{Escape}');

    // enter search term
    const searchBar = screen.getByPlaceholderText('Search');

    // eslint-disable-next-line @databricks/no-userevent-type
    await user.type(searchBar, 'rephrase');
    await waitForElementToBeRemoved(await screen.findByTestId('timeline-tree-node-document-qa-chain'));

    // Assert that only the filtered span is rendered
    expect(await screen.findByTestId('timeline-tree-node-rephrase_chat_to_queue')).toBeInTheDocument();

    await user.clear(searchBar);

    // Assert that the tree is reset
    expect(await screen.findByTestId('timeline-tree-node-document-qa-chain')).toBeInTheDocument();

    // eslint-disable-next-line @databricks/no-userevent-type
    await user.type(searchBar, 'string with no match');

    // Assert that no spans are rendered
    expect(await screen.findByText('No results found. Try using a different search term.')).toBeInTheDocument();
  });

  it('rerenders only when a new root span ID is provided', async () => {
    const { rerender } = render(<TestComponent modelTrace={MOCK_TRACE} />);

    // Assert that all spans are expanded
    expect(screen.getByTestId('timeline-tree-node-document-qa-chain')).toBeInTheDocument();
    expect(screen.getByTestId('timeline-tree-node-_generate_response')).toBeInTheDocument();
    expect(screen.getByTestId('timeline-tree-node-rephrase_chat_to_queue')).toBeInTheDocument();

    // Select the third span
    await userEvent.click(screen.getByText('rephrase_chat_to_queue'));
    expect(await screen.findByText('rephrase_chat_to_queue-input')).toBeInTheDocument();

    // assert that the tree is not rerendered when the same root node is passed
    const clonedTrace = cloneDeep(MOCK_TRACE); // deep copy to make objects not referentially equal
    rerender(<TestComponent modelTrace={clonedTrace} />);
    expect(await screen.findByText('rephrase_chat_to_queue-input')).toBeInTheDocument();

    // assert that the tree is rerendered when a new root span is passed
    const newTrace = cloneDeep(MOCK_TRACE);
    // rewrite trace id to indicate a new trace
    const traceInfo = newTrace.info as ModelTraceInfo;
    traceInfo.request_id = 'new-trace-id';
    const spans = newTrace.data.spans as ModelTraceSpanV2[];
    spans[0].name = 'new-span';
    spans[0].context.span_id = 'new-span';
    spans[1].parent_id = 'new-span';
    rerender(<TestComponent modelTrace={newTrace} />);

    // expect that the new span is rendered (appears in both tree and graph node)
    expect((await screen.findAllByText('new-span')).length).toBeGreaterThanOrEqual(1);

    // expect that the span selection doesn't change if the previous node is still in the tree
    expect(await screen.findByText('rephrase_chat_to_queue-input')).toBeInTheDocument();
  });

  it('should allow jumping to matches', async () => {
    const user = userEvent.setup({ delay: null });
    render(<TestComponent modelTrace={MOCK_TRACE} isSearchVisible />);

    // Search for the word "input"
    const searchBar = screen.getByPlaceholderText('Search');
    // eslint-disable-next-line @databricks/no-userevent-type
    await user.type(searchBar, 'input');

    // expect 3 matches (one in each span)
    expect(await screen.findByText('1 / 3')).toBeInTheDocument();

    // assert that the first span is selected by checking for the output
    // text (since the input text is broken up by a highlighted span)
    expect(await screen.findByText('document-qa-chain-output')).toBeInTheDocument();

    // next match
    const nextButton = await screen.findByTestId('next-search-match');
    await user.click(nextButton);

    // assert that match label updates, and new span is selected
    expect(await screen.findByText('2 / 3')).toBeInTheDocument();
    expect(await screen.findByText('_generate_response-output')).toBeInTheDocument();

    await user.click(nextButton);
    expect(await screen.findByText('3 / 3')).toBeInTheDocument();
    expect(await screen.findByText('rephrase_chat_to_queue-output')).toBeInTheDocument();

    // user shouldn't be able to progress past the last match
    await user.click(nextButton);
    expect(await screen.findByText('3 / 3')).toBeInTheDocument();
    expect(await screen.findByText('rephrase_chat_to_queue-output')).toBeInTheDocument();

    const prevButton = await screen.findByTestId('prev-search-match');
    await user.click(prevButton);
    expect(await screen.findByText('2 / 3')).toBeInTheDocument();
    expect(await screen.findByText('_generate_response-output')).toBeInTheDocument();

    await user.click(prevButton);
    expect(await screen.findByText('1 / 3')).toBeInTheDocument();
    expect(await screen.findByText('document-qa-chain-output')).toBeInTheDocument();

    // user shouldn't be able to progress past the first match
    await user.click(prevButton);
    expect(await screen.findByText('1 / 3')).toBeInTheDocument();
    expect(await screen.findByText('document-qa-chain-output')).toBeInTheDocument();
  });

  it('should open the correct tabs when searching', async () => {
    const user = userEvent.setup({ delay: null });
    const trace = {
      data: {
        spans: [MOCK_EVENTS_SPAN],
      },
      info: {},
    };

    render(<TestComponent modelTrace={trace} isSearchVisible />);

    // expect that the content tab is open by default
    expect(await screen.findByText('events_span-input')).toBeInTheDocument();

    // search for an attribute
    const searchBar = screen.getByPlaceholderText('Search');
    // eslint-disable-next-line @databricks/no-userevent-type
    await user.type(searchBar, 'top-level-attribute');

    // expect that the attributes tab is open
    expect(await screen.findByText('top-level-attribute')).toBeInTheDocument();

    await user.clear(searchBar);
    // eslint-disable-next-line @databricks/no-userevent-type
    await user.type(searchBar, 'event1-attr1');

    expect(await screen.findByText('event-level-attribute')).toBeInTheDocument();
  });

  it('should default to content tab when the selected node does not have chats', async () => {
    const trace = {
      data: {
        spans: [MOCK_CHAT_TOOL_CALL_SPAN, { ...MOCK_EVENTS_SPAN, parent_id: MOCK_CHAT_TOOL_CALL_SPAN.context.span_id }],
      },
      info: {},
    };

    render(<TestComponent modelTrace={trace} />);

    // Chat-shaped inputs and outputs render in the content tab.
    expect(await screen.findByTestId('model-trace-explorer-content-tab')).toBeInTheDocument();

    // click the non-chat span (also appears as a graph node)
    const eventSpan = screen.getAllByText('events_span')[0];
    await userEvent.click(eventSpan);

    // expect that the content tab is open
    expect(await screen.findByTestId('model-trace-explorer-content-tab')).toBeInTheDocument();
  });

  it('should correctly filter the tree', async () => {
    render(<TestComponent modelTrace={MOCK_TRACE} />);

    expect(screen.getByTestId('timeline-tree-node-document-qa-chain')).toBeInTheDocument();
    expect(screen.getByTestId('timeline-tree-node-_generate_response')).toBeInTheDocument();
    expect(screen.getByTestId('timeline-tree-node-rephrase_chat_to_queue')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Trace display settings' }));
    await userEvent.hover(screen.getByRole('menuitem', { name: 'Filter by span type' }));

    // de-select the "Chain" and "Chat model" span types
    const chainSelector = await screen.findByRole('menuitemcheckbox', { name: 'Chain' });
    await userEvent.click(chainSelector);
    const chatModelSelector = await screen.findByRole('menuitemcheckbox', { name: 'Chat model' });
    await userEvent.click(chatModelSelector);

    // since the "show parents" checkbox is checked by default, all spans should still be visible
    expect(screen.getAllByText('document-qa-chain')).not.toHaveLength(0);
    expect(screen.getByText('_generate_response')).toBeInTheDocument();
    expect(screen.getByText('rephrase_chat_to_queue')).toBeInTheDocument();

    // uncheck the "show parents" checkbox
    const showParentsCheckbox = screen.getByRole('menuitemcheckbox', { name: 'Show all parent spans' });
    await userEvent.click(showParentsCheckbox);

    // now that the parents checkbox is unchecked,
    // only the "rephrase" span should be visible
    expect(screen.queryByTestId('timeline-tree-node-document-qa-chain')).not.toBeInTheDocument();
    expect(screen.queryByTestId('timeline-tree-node-_generate_response')).not.toBeInTheDocument();
    expect(screen.getByTestId('timeline-tree-node-rephrase_chat_to_queue')).toBeInTheDocument();
  });

  it('keeps the assessments pane closed by default', () => {
    render(<TestComponent modelTrace={MOCK_V3_TRACE} />);

    expect(screen.queryByRole('heading', { name: 'Assessments' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Assess trace/ })).toBeInTheDocument();
  });

  it('shows the selected span assessment count in the Assess button', async () => {
    render(<TestComponent modelTrace={MOCK_V3_TRACE} />);

    const assessButton = screen.getByRole('button', { name: 'Assess trace (1 assessment)' });
    expect(within(assessButton).getByText('1')).toBeInTheDocument();

    await userEvent.click(assessButton);

    expect(screen.getByRole('heading', { name: 'Assessments' })).toBeInTheDocument();
  });

  it('persists explicit assessment pane open and close choices across traces', async () => {
    const { rerender } = render(<TestComponent modelTrace={MOCK_V3_TRACE} explorerKey="first-trace" />);

    await userEvent.click(screen.getByRole('button', { name: 'Assess trace (1 assessment)' }));

    rerender(<TestComponent modelTrace={MOCK_V4_TRACE_UC_SCHEMA} explorerKey="second-trace" />);
    expect(screen.getByRole('heading', { name: 'Assessments' })).toBeInTheDocument();

    // The existing icon-only close control has no accessible name.
    await userEvent.click(screen.getByTestId('close-assessments-pane-button'));

    rerender(<TestComponent modelTrace={MOCK_V3_TRACE} explorerKey="third-trace" />);

    expect(screen.queryByRole('heading', { name: 'Assessments' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Assess trace (1 assessment)' })).toBeInTheDocument();
  });

  it('should render typeahead when creating a new assessment', async () => {
    const assessments = [MOCK_ASSESSMENT, MOCK_EXPECTATION, MOCK_SPAN_ASSESSMENT];

    render(<TestComponent modelTrace={MOCK_V3_TRACE} collapseAssessmentPane={false} />, {
      wrapper: ({ children }) => (
        <AssessmentSchemaContextProvider assessments={assessments}>{children}</AssessmentSchemaContextProvider>
      ),
    });

    expect(screen.getByTestId('assessments-pane')).toBeInTheDocument();

    const createButton = screen.getByText('Add feedback');
    await userEvent.click(createButton);

    // expect that the default assessment input type is boolean
    expect(screen.getByTestId('assessment-value-boolean-input')).toBeInTheDocument();

    expect(screen.getByTestId('assessment-name-typeahead-input')).toBeInTheDocument();
    const typeahead = screen.getByTestId('assessment-name-typeahead-input');
    await userEvent.click(typeahead);

    // expect that the list of assessment names is rendered
    const assessmentNames = assessments.map((assessment) => assessment.assessment_name);
    for (const name of assessmentNames) {
      expect(screen.getByTestId(`assessment-name-typeahead-item-${name}`)).toBeInTheDocument();
    }

    // when clicking the typeahead item, the input should be updated
    const factsItem = screen.getByTestId(`assessment-name-typeahead-item-expected_facts`);
    await userEvent.click(factsItem);
    expect(typeahead).toHaveValue('expected_facts');
    // JSON data type is clamped to string for feedback assessments
    expect(screen.getByTestId('assessment-value-string-input')).toBeInTheDocument();
  });

  it('should render in-progress traces with multiple top-level spans', async () => {
    // Create an in-progress trace where the root span hasn't been emitted yet
    const inProgressTrace = {
      ...MOCK_V3_TRACE,
      info: {
        ...MOCK_TRACE_INFO_V3,
        state: 'IN_PROGRESS',
      },
      data: {
        spans: [
          // Two spans with parent span ID pointing to the root span not being emitted yet
          {
            ...MOCK_V3_SPANS[1],
            parent_span_id: 'non-existing-parent-span-id',
          },
          {
            ...MOCK_V3_SPANS[2],
            parent_span_id: 'non-existing-parent-span-id',
          },
        ],
      },
    };

    render(<TestComponent modelTrace={inProgressTrace} />);

    // Both top-level spans should be visible in the tree
    const spanElements = await screen.findAllByText('document-qa-chain');
    expect(spanElements.length).toBeGreaterThanOrEqual(1);

    expect(screen.getByText('rephrase_chat_to_queue')).toBeInTheDocument();

    // Should be able to select and view each span independently
    await userEvent.click(screen.getByText('rephrase_chat_to_queue'));
    expect(await screen.findByText('rephrase_chat_to_queue-input')).toBeInTheDocument();

    // Switch to the other top-level span
    await userEvent.click(spanElements[0]);
    expect(await screen.findByText('rephrase_chat_to_queue-input')).toBeInTheDocument();
  });

  it('renders the Graph toggle in settings by default when spans form a workflow', async () => {
    render(<TestComponent modelTrace={MOCK_TRACE} />);

    await userEvent.click(await screen.findByRole('button', { name: 'Trace display settings' }));
    expect(await screen.findByRole('menuitemcheckbox', { name: 'Show graph' })).toBeInTheDocument();
  });

  it('hides the Graph toggle and canvas when enableGraphView is false', async () => {
    render(<TestComponent modelTrace={MOCK_TRACE} enableGraphView={false} />);

    // Span tree still renders
    expect(await screen.findByText('Spans')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Trace display settings' }));
    expect(screen.queryByRole('menuitemcheckbox', { name: 'Show graph' })).not.toBeInTheDocument();

    // The span navigator ("Click a graph node to navigate spans") is rendered inside the same
    // hasGraph block as the canvas — if the toggle button is absent the navigator is also absent.
    expect(screen.queryByText(/click a graph node/i)).not.toBeInTheDocument();

    // GraphViewWorkflowCanvas is React.lazy + Suspense and never resolves in JSDOM, so its
    // absence cannot be directly asserted here. The guard `graphAvailable = enableGraphView && …`
    // ensures it is never mounted when enableGraphView is false.
  });

  it('should render V4 error feedback assessments that have no value field', async () => {
    // V4 API omits feedback.value for error assessments — only error is present.
    // Previously these were silently dropped by the grouping logic.
    const traceWithV4ErrorFeedback: ModelTrace = {
      data: { spans: MOCK_V3_SPANS },
      info: {
        ...MOCK_TRACE_INFO_V3,
        assessments: [
          {
            ...MOCK_ASSESSMENT,
            assessment_id: 'a-error-1',
            assessment_name: 'error_judge',
            feedback: {
              error: { error_code: 'EVALUATION_FAILED', error_message: 'Something went wrong' },
            },
          },
        ],
      },
    };

    render(<TestComponent modelTrace={traceWithV4ErrorFeedback} collapseAssessmentPane={false} />);

    // The assessments pane should open since the trace has assessments
    expect(screen.getByTestId('assessments-pane')).toBeInTheDocument();

    // The error feedback group should be rendered with its assessment name,
    // proving the V4 error feedback was not dropped by groupFeedbacks
    expect(await screen.findByText('error_judge')).toBeInTheDocument();
  });
});
