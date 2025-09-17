import { render, screen, within, waitForElementToBeRemoved } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { cloneDeep } from 'lodash';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

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
} from './ModelTraceExplorer.test-utils';
import { AssessmentSchemaContextProvider } from './contexts/AssessmentSchemaContext';

// increase timeout and it's a heavy test
// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000);

// mock the scrollIntoView function to prevent errors
window.HTMLElement.prototype.scrollIntoView = jest.fn();

jest.mock('./hooks/useGetModelTraceInfoV3', () => ({
  useGetModelTraceInfoV3: jest.fn().mockReturnValue({
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

const TestComponent = ({ modelTrace }: { modelTrace: ModelTrace }) => {
  const queryClient = new QueryClient();

  return (
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={queryClient}>
          <ModelTraceExplorer modelTrace={modelTrace} initialActiveView="detail" />
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>
  );
};

describe('ModelTraceExplorer', () => {
  it.each([MOCK_TRACE, MOCK_V3_TRACE])(
    'renders the component and allows to inspect selected spans',
    async (trace: ModelTrace) => {
      render(<TestComponent modelTrace={trace} />);

      // Assert existence of task column header
      expect(screen.getByText('Trace breakdown')).toBeInTheDocument();

      // Expect timeline view to be closed at first (due to JSDOM's 1024 default screen width)
      expect(screen.queryByTestId('time-marker-area')).not.toBeInTheDocument();
      await userEvent.click(screen.getByTestId('show-timeline-info-button'));

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
    render(<TestComponent modelTrace={MOCK_TRACE} />);

    // de-select "show parents" button so the rest of the test is easier to run
    const filterButton = screen.getByRole('button', { name: /Filter/i });
    await userEvent.click(filterButton);
    const showParentsCheckbox = screen.getByRole('checkbox', { name: /Show all parent spans/i });
    await userEvent.click(showParentsCheckbox);

    // enter search term
    const searchBar = screen.getByPlaceholderText('Search');

    await userEvent.type(searchBar, 'rephrase');
    await waitForElementToBeRemoved(await screen.findByText('document-qa-chain'));

    // Assert that only the filtered span is rendered
    expect(await screen.findByText('rephrase_chat_to_queue')).toBeInTheDocument();

    await userEvent.clear(searchBar);

    // Assert that the tree is reset
    expect(await screen.findByText('document-qa-chain')).toBeInTheDocument();

    await userEvent.type(searchBar, 'string with no match');

    // Assert that no spans are rendered
    expect(await screen.findByText('No results found. Try using a different search term.')).toBeInTheDocument();
  });

  it('rerenders only when a new root span ID is provided', async () => {
    const { rerender } = render(<TestComponent modelTrace={MOCK_TRACE} />);

    // Assert that all spans are expanded
    expect(screen.getByText('document-qa-chain')).toBeInTheDocument();
    expect(screen.getByText('_generate_response')).toBeInTheDocument();
    expect(screen.getByText('rephrase_chat_to_queue')).toBeInTheDocument();

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

    // expect that the new span is rendered
    expect(await screen.findByText('new-span')).toBeInTheDocument();

    // expect that the span selection doesn't change if the previous node is still in the tree
    expect(await screen.findByText('rephrase_chat_to_queue-input')).toBeInTheDocument();
  });

  it('should allow jumping to matches', async () => {
    render(<TestComponent modelTrace={MOCK_TRACE} />);

    // Search for the word "input"
    const searchBar = screen.getByPlaceholderText('Search');
    await userEvent.type(searchBar, 'input');

    // expect 3 matches (one in each span)
    expect(await screen.findByText('1 / 3')).toBeInTheDocument();

    // assert that the first span is selected by checking for the output
    // text (since the input text is broken up by a highlighted span)
    expect(await screen.findByText('document-qa-chain-output')).toBeInTheDocument();

    // next match
    const nextButton = await screen.findByTestId('next-search-match');
    await userEvent.click(nextButton);

    // assert that match label updates, and new span is selected
    expect(await screen.findByText('2 / 3')).toBeInTheDocument();
    expect(await screen.findByText('_generate_response-output')).toBeInTheDocument();

    await userEvent.click(nextButton);
    expect(await screen.findByText('3 / 3')).toBeInTheDocument();
    expect(await screen.findByText('rephrase_chat_to_queue-output')).toBeInTheDocument();

    // user shouldn't be able to progress past the last match
    await userEvent.click(nextButton);
    expect(await screen.findByText('3 / 3')).toBeInTheDocument();
    expect(await screen.findByText('rephrase_chat_to_queue-output')).toBeInTheDocument();

    const prevButton = await screen.findByTestId('prev-search-match');
    await userEvent.click(prevButton);
    expect(await screen.findByText('2 / 3')).toBeInTheDocument();
    expect(await screen.findByText('_generate_response-output')).toBeInTheDocument();

    await userEvent.click(prevButton);
    expect(await screen.findByText('1 / 3')).toBeInTheDocument();
    expect(await screen.findByText('document-qa-chain-output')).toBeInTheDocument();

    // user shouldn't be able to progress past the first match
    await userEvent.click(prevButton);
    expect(await screen.findByText('1 / 3')).toBeInTheDocument();
    expect(await screen.findByText('document-qa-chain-output')).toBeInTheDocument();
  });

  it('should open the correct tabs when searching', async () => {
    const trace = {
      data: {
        spans: [MOCK_EVENTS_SPAN],
      },
      info: {},
    };

    render(<TestComponent modelTrace={trace} />);

    // expect that the content tab is open by default
    expect(await screen.findByText('events_span-input')).toBeInTheDocument();

    // search for an attribute
    const searchBar = screen.getByPlaceholderText('Search');
    await userEvent.type(searchBar, 'top-level-attribute');

    // expect that the attributes tab is open
    expect(await screen.findByText('top-level-attribute')).toBeInTheDocument();

    await userEvent.clear(searchBar);
    await userEvent.type(searchBar, 'event1-attr1');

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

    // expect that the chat tab is open by default
    expect(await screen.findByTestId('model-trace-explorer-chat-tab')).toBeInTheDocument();

    // click the non-chat span
    const eventSpan = screen.getByText('events_span');
    await userEvent.click(eventSpan);

    // expect that the content tab is open
    expect(await screen.findByTestId('model-trace-explorer-content-tab')).toBeInTheDocument();
  });

  it('should correctly filter the tree', async () => {
    render(<TestComponent modelTrace={MOCK_TRACE} />);

    expect(screen.getByText('document-qa-chain')).toBeInTheDocument();
    expect(screen.getByText('_generate_response')).toBeInTheDocument();
    expect(screen.getByText('rephrase_chat_to_queue')).toBeInTheDocument();

    const filterButton = screen.getByRole('button', { name: /Filter/i });
    await userEvent.click(filterButton);

    // de-select the "Chain" and "Chat model" span types
    const chainSelector = await screen.findByRole('checkbox', { name: /Chain/i });
    await userEvent.click(chainSelector);
    const chatModelSelector = await screen.findByRole('checkbox', { name: /Chat model/i });
    await userEvent.click(chatModelSelector);

    // since the "show parents" checkbox is checked by default, all spans should still be visible
    expect(screen.getByText('document-qa-chain')).toBeInTheDocument();
    expect(screen.getByText('_generate_response')).toBeInTheDocument();
    expect(screen.getByText('rephrase_chat_to_queue')).toBeInTheDocument();

    // uncheck the "show parents" checkbox
    const showParentsCheckbox = screen.getByRole('checkbox', { name: /Show all parent spans/i });
    await userEvent.click(showParentsCheckbox);

    // now that the parents checkbox is unchecked,
    // only the "rephrase" span should be visible
    expect(screen.queryByText('document-qa-chain')).not.toBeInTheDocument();
    expect(screen.queryByText('_generate_response')).not.toBeInTheDocument();
    expect(screen.getByText('rephrase_chat_to_queue')).toBeInTheDocument();
  });

  it('should open the assessments pane when the assessment tag is clicked', async () => {
    render(<TestComponent modelTrace={MOCK_V3_TRACE} />);

    // expect that the assessments pane is open by
    // default as the root node has assessments
    expect(screen.getByTestId('assessments-pane')).toBeInTheDocument();

    // close the assessments pane
    await userEvent.click(screen.getByTestId('close-assessments-pane-button'));

    // expect that the assessments pane is closed
    expect(screen.queryByTestId('assessments-pane')).not.toBeInTheDocument();

    // click the assessment tag
    const assessmentTag = screen.getByTestId(/assessment-tag/);
    await userEvent.click(assessmentTag);

    // expect that the assessments pane is open
    expect(screen.getByTestId('assessments-pane')).toBeInTheDocument();
  });

  it('should render typeahead when creating a new assessment', async () => {
    const assessments = [MOCK_ASSESSMENT, MOCK_EXPECTATION, MOCK_SPAN_ASSESSMENT];

    render(<TestComponent modelTrace={MOCK_V3_TRACE} />, {
      wrapper: ({ children }) => (
        <AssessmentSchemaContextProvider assessments={assessments}>{children}</AssessmentSchemaContextProvider>
      ),
    });

    expect(screen.getByTestId('assessments-pane')).toBeInTheDocument();

    const createButton = screen.getByText('Add new assessment');
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
    expect(screen.getByTestId('assessment-value-json-input')).toBeInTheDocument();
  });
});
