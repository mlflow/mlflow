import { jest, beforeAll, afterAll, describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { cloneDeep } from 'lodash';
import type { ComponentProps } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

import type { ModelTrace, ModelTraceSpanV2 } from './ModelTrace.types';
import { ModelTraceExplorer } from './ModelTraceExplorer';
import { MOCK_TRACE } from '../ModelTraceExplorer.test-utils';
import { ModelTraceExplorerPreferencesProvider } from './ModelTraceExplorerPreferencesContext';
import { ModelTraceExplorerContextProvider } from './ModelTraceExplorerContext';

window.HTMLElement.prototype.scrollIntoView = jest.fn();

let originalResizeObserver: typeof ResizeObserver;
beforeAll(() => {
  originalResizeObserver = globalThis.ResizeObserver;
  globalThis.ResizeObserver = class MockResizeObserver {
    observe = jest.fn();
    unobserve = jest.fn();
    disconnect = jest.fn();
  } as unknown as typeof ResizeObserver;
});

afterAll(() => {
  globalThis.ResizeObserver = originalResizeObserver;
});

const TestComponent = ({
  modelTrace,
  isSearchVisible = false,
  comparisonModelTrace,
  comparisonRole,
}: {
  modelTrace: ModelTrace;
  isSearchVisible?: boolean;
  comparisonModelTrace?: ModelTrace;
  comparisonRole?: ComponentProps<typeof ModelTraceExplorer>['comparisonRole'];
}) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>
      <QueryClientProvider client={new QueryClient()}>
        <ModelTraceExplorerPreferencesProvider initialRenderMode="default">
          <ModelTraceExplorerContextProvider isSearchVisible={isSearchVisible}>
            <ModelTraceExplorer
              modelTrace={modelTrace}
              comparisonModelTrace={comparisonModelTrace}
              comparisonRole={comparisonRole}
            />
          </ModelTraceExplorerContextProvider>
        </ModelTraceExplorerPreferencesProvider>
      </QueryClientProvider>
    </DesignSystemProvider>
  </IntlProvider>
);

describe('ModelTraceExplorer trace comparison', () => {
  it('keeps unchanged comparison content expanded and sorts changed attributes first', async () => {
    const baselineTrace = cloneDeep(MOCK_TRACE);
    const candidateTrace = cloneDeep(MOCK_TRACE);
    const baselineRoot = baselineTrace.data.spans[0] as ModelTraceSpanV2;
    const candidateRoot = candidateTrace.data.spans[0] as ModelTraceSpanV2;

    baselineRoot.attributes = {
      ...baselineRoot.attributes,
      unchangedAttribute: 'same',
      changedAttribute: 'before',
      'mlflow.spanInputs': JSON.stringify({ unchangedInput: 'same' }),
      'mlflow.spanOutputs': JSON.stringify({ unchangedOutput: 'same' }),
    };
    candidateRoot.attributes = {
      ...candidateRoot.attributes,
      unchangedAttribute: 'same',
      changedAttribute: 'after',
      'mlflow.spanInputs': JSON.stringify({ unchangedInput: 'same' }),
      'mlflow.spanOutputs': JSON.stringify({ unchangedOutput: 'same' }),
    };

    render(
      <TestComponent modelTrace={baselineTrace} comparisonModelTrace={candidateTrace} comparisonRole="baseline" />,
    );

    expect(screen.getByTestId('model-trace-explorer-input-section-toggle')).toHaveAttribute('aria-expanded', 'true');
    expect(screen.queryByTestId('model-trace-explorer-inputs-expand-hint')).not.toBeInTheDocument();
    expect(screen.getByTestId('model-trace-explorer-output-section-toggle')).toHaveAttribute('aria-expanded', 'true');
    expect(screen.queryByTestId('model-trace-explorer-outputs-expand-hint')).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('tab', { name: /^Attributes/ }));

    expect(screen.getByTestId('model-trace-explorer-attributes-section-toggle')).toHaveAttribute(
      'aria-expanded',
      'true',
    );
    const changedAttribute = screen.getByText('changedAttribute');
    const unchangedAttribute = screen.getByText('unchangedAttribute');
    expect(
      changedAttribute.compareDocumentPosition(unchangedAttribute) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it('shows the union of baseline and candidate spans regardless of the active trace', () => {
    const baselineTrace = cloneDeep(MOCK_TRACE);
    const candidateTrace = cloneDeep(MOCK_TRACE);
    candidateTrace.data.spans = candidateTrace.data.spans.slice(0, 1);

    render(
      <TestComponent modelTrace={candidateTrace} comparisonModelTrace={baselineTrace} comparisonRole="candidate" />,
    );

    expect(screen.getByText('rephrase_chat_to_queue')).toBeInTheDocument();
  });

  it('shows baseline and candidate model names in the selected span header', () => {
    const baselineTrace = cloneDeep(MOCK_TRACE);
    const candidateTrace = cloneDeep(MOCK_TRACE);
    const baselineRoot = baselineTrace.data.spans[0] as ModelTraceSpanV2;
    const candidateRoot = candidateTrace.data.spans[0] as ModelTraceSpanV2;
    baselineRoot.attributes = {
      ...baselineRoot.attributes,
      'mlflow.llm.model': 'baseline-model',
      'mlflow.chat.tokenUsage': JSON.stringify({ input_tokens: 10, output_tokens: 20, total_tokens: 30 }),
      'mlflow.llm.cost': JSON.stringify({ input_cost: 0.01, output_cost: 0.02, total_cost: 0.03 }),
    };
    candidateRoot.attributes = {
      ...candidateRoot.attributes,
      'mlflow.llm.model': 'candidate-model',
      'mlflow.chat.tokenUsage': JSON.stringify({ input_tokens: 20, output_tokens: 30, total_tokens: 50 }),
      'mlflow.llm.cost': JSON.stringify({ input_cost: 0.02, output_cost: 0.03, total_cost: 0.05 }),
    };

    render(
      <TestComponent modelTrace={baselineTrace} comparisonModelTrace={candidateTrace} comparisonRole="baseline" />,
    );

    expect(screen.getByText('baseline-model')).toBeInTheDocument();
    expect(screen.getByText('candidate-model')).toBeInTheDocument();
    expect(screen.queryByLabelText('Duration')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Tokens')).not.toBeInTheDocument();
    expect(screen.queryByText('Estimated cost')).not.toBeInTheDocument();
  });

  it('renders events from both sides of a matched span', async () => {
    const baselineTrace = cloneDeep(MOCK_TRACE);
    const candidateTrace = cloneDeep(MOCK_TRACE);
    const baselineRoot = baselineTrace.data.spans[0] as ModelTraceSpanV2;
    const candidateRoot = candidateTrace.data.spans[0] as ModelTraceSpanV2;
    baselineRoot.events = [{ name: 'log', attributes: { message: 'baseline event value' } }];
    candidateRoot.events = [
      { name: 'log', attributes: { message: 'candidate event value' } },
      { name: 'exception', attributes: { 'exception.message': 'candidate exception value' } },
    ];

    render(
      <TestComponent modelTrace={baselineTrace} comparisonModelTrace={candidateTrace} comparisonRole="baseline" />,
    );
    await userEvent.click(screen.getByRole('tab', { name: /^Events/ }));

    expect(screen.getByText('baseline event value')).toBeInTheDocument();
    expect(screen.getByText('candidate event value')).toBeInTheDocument();
    expect(screen.getByText('candidate exception value')).toBeInTheDocument();
  });

  it('compares conversations stored only in mlflow.chat.messages', () => {
    const baselineTrace = cloneDeep(MOCK_TRACE);
    const candidateTrace = cloneDeep(MOCK_TRACE);
    const baselineRoot = baselineTrace.data.spans[0] as ModelTraceSpanV2;
    const candidateRoot = candidateTrace.data.spans[0] as ModelTraceSpanV2;
    baselineRoot.attributes = {
      'mlflow.spanType': JSON.stringify('CHAIN'),
      'mlflow.chat.messages': JSON.stringify([
        { role: 'user', content: 'baseline fallback user message' },
        { role: 'assistant', content: 'baseline fallback assistant message' },
      ]),
    };
    candidateRoot.attributes = {
      'mlflow.spanType': JSON.stringify('CHAIN'),
      'mlflow.chat.messages': JSON.stringify([
        { role: 'user', content: 'candidate fallback user message' },
        { role: 'assistant', content: 'candidate fallback assistant message' },
      ]),
    };

    render(
      <TestComponent modelTrace={baselineTrace} comparisonModelTrace={candidateTrace} comparisonRole="baseline" />,
    );

    const content = screen.getByTestId('model-trace-explorer-content-tab');
    expect(content).toHaveTextContent('baseline fallback user message');
    expect(content).toHaveTextContent('candidate fallback user message');
    expect(content).toHaveTextContent('baseline fallback assistant message');
    expect(content).toHaveTextContent('candidate fallback assistant message');
  });

  it('finds candidate-only content in a matched span', async () => {
    const user = userEvent.setup();
    const baselineTrace = cloneDeep(MOCK_TRACE);
    const candidateTrace = cloneDeep(MOCK_TRACE);
    const candidateRoot = candidateTrace.data.spans[0] as ModelTraceSpanV2;
    candidateRoot.attributes = {
      ...candidateRoot.attributes,
      'mlflow.spanOutputs': JSON.stringify({ response: 'candidate-only-search-value' }),
    };

    render(
      <TestComponent
        modelTrace={baselineTrace}
        comparisonModelTrace={candidateTrace}
        comparisonRole="baseline"
        isSearchVisible
      />,
    );
    const searchBar = screen.getByPlaceholderText('Search');
    await user.click(searchBar);
    await user.keyboard('candidate-only-search-value');

    expect(await screen.findByText('1 / 1')).toBeInTheDocument();
    expect(screen.getByTestId('timeline-tree-node-document-qa-chain')).toBeInTheDocument();
  });

  it('shows no attribute change count for a one-sided span without attributes', async () => {
    const baselineTrace = cloneDeep(MOCK_TRACE);
    const candidateTrace = cloneDeep(MOCK_TRACE);
    candidateTrace.data.spans = candidateTrace.data.spans.slice(0, 1);
    const baselineOnlySpan = baselineTrace.data.spans.find(
      (span) => (span as ModelTraceSpanV2).name === 'rephrase_chat_to_queue',
    ) as ModelTraceSpanV2;
    baselineOnlySpan.attributes = {};

    render(
      <TestComponent modelTrace={candidateTrace} comparisonModelTrace={baselineTrace} comparisonRole="candidate" />,
    );
    await userEvent.click(screen.getByText('rephrase_chat_to_queue'));

    expect(screen.getByRole('tab', { name: /^Attributes/ })).not.toHaveTextContent(/[−+]\d/);
  });

  it('counts only the field lines of a one-sided span, not the braces around them', async () => {
    const baselineTrace = cloneDeep(MOCK_TRACE);
    const candidateTrace = cloneDeep(MOCK_TRACE);
    candidateTrace.data.spans = candidateTrace.data.spans.slice(0, 1);

    render(
      <TestComponent modelTrace={candidateTrace} comparisonModelTrace={baselineTrace} comparisonRole="candidate" />,
    );
    await userEvent.click(screen.getByText('rephrase_chat_to_queue'));

    // One input field and one output field, one line each.
    expect(screen.getByRole('tab', { name: /^Inputs/ })).toHaveTextContent(/−2(?!\d)/);
  });

  it('handles missing active attributes when the counterpart has attributes', async () => {
    const baselineTrace = cloneDeep(MOCK_TRACE);
    const candidateTrace = cloneDeep(MOCK_TRACE);
    const baselineRoot = baselineTrace.data.spans[0] as ModelTraceSpanV2;
    const candidateRoot = candidateTrace.data.spans[0] as ModelTraceSpanV2;
    baselineRoot.attributes = undefined;
    candidateRoot.attributes = { candidateOnlyAttribute: 'candidate attribute value' };

    render(
      <TestComponent modelTrace={baselineTrace} comparisonModelTrace={candidateTrace} comparisonRole="baseline" />,
    );
    await userEvent.click(screen.getByRole('tab', { name: /^Attributes/ }));

    expect(screen.getByText('candidateOnlyAttribute')).toBeInTheDocument();
    expect(screen.getByText(/candidate attribute value/)).toBeInTheDocument();
  });
});
