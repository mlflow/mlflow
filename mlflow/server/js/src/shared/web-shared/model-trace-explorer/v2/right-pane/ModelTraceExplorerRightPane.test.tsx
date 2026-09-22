import { describe, it, expect } from '@jest/globals';
import { screen, within } from '@testing-library/react';
import { render } from '@databricks/web-shared/test-utils/render';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerContentTab } from './ModelTraceExplorerContentTab';
import { SpanModelCostBadge } from './SpanModelCostBadge';
import type { ModelTraceSpan } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';
import { getDefaultActiveTab, normalizeNewSpanData } from '../ModelTraceExplorer.utils';
import { mockSpans, MOCK_RETRIEVER_SPAN, MOCK_CHAT_SPAN } from '../../ModelTraceExplorer.test-utils';
import { ModelTraceExplorerPreferencesProvider } from '../ModelTraceExplorerPreferencesContext';

const DEFAULT_SPAN: ModelTraceSpan = mockSpans[0];

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>
      <ModelTraceExplorerPreferencesProvider initialRenderMode="default">
        {children}
      </ModelTraceExplorerPreferencesProvider>
    </DesignSystemProvider>
  </IntlProvider>
);

describe('ModelTraceExplorerRightPane', () => {
  it('renders structured Jev inputs and answers with the resolved model', async () => {
    const inputs = {
      state: 'I was charged twice',
      model: 'jev-latest',
      questions: {
        billing: { type: 'noul', instructions: 'Is this about billing?' },
        tone: { type: 'choice', criteria: { calm: null, angry: null } },
        urgency: { type: 'score', criteria: ['low', 'high'] },
      },
    };
    const outputs = {
      model: 'jev-1.13.0',
      answers: {
        billing: { type: 'noul', noul: 0.98 },
        tone: { type: 'choice', choice: 'angry', confidence: 0.96, probabilities: { calm: 0.02, angry: 0.98 } },
        urgency: { type: 'score', score: 0.9, confidence: 0.94, legend: { 0: 'low', 1: 'high' } },
      },
      usage: { input_tokens: 10, output_tokens: 0 },
    };
    const tokenUsage = { input_tokens: 10, output_tokens: 0, total_tokens: 10 };
    const span = normalizeNewSpanData(
      {
        ...DEFAULT_SPAN,
        name: 'TypeSafeClient',
        attributes: {
          'mlflow.spanType': JSON.stringify('LLM'),
          'mlflow.spanInputs': JSON.stringify(inputs),
          'mlflow.spanOutputs': JSON.stringify(outputs),
          'mlflow.llm.model': JSON.stringify('jev-1.13.0'),
          'mlflow.chat.tokenUsage': JSON.stringify(tokenUsage),
        },
      },
      0,
      0,
      [],
      {},
      'jev-trace',
    );

    expect(span.type).toBe(ModelSpanType.LLM);
    expect(span.inputs).toEqual(inputs);
    expect(span.outputs).toEqual(outputs);
    expect(span.modelName).toBe('jev-1.13.0');
    expect(span.tokenUsage).toEqual(tokenUsage);
    expect(span.chatMessages).toBeUndefined();
    expect(getDefaultActiveTab(span)).toBe('content');

    render(
      <>
        <div data-testid="model-badge">
          <SpanModelCostBadge activeSpan={span} />
        </div>
        <ModelTraceExplorerContentTab activeSpan={span} searchFilter="" activeMatch={null} />
      </>,
      { wrapper: Wrapper },
    );

    const modelBadge = within(screen.getByTestId('model-badge'));
    expect(modelBadge.getByText('Model')).toBeInTheDocument();
    expect(modelBadge.getByText('jev-1.13.0')).toBeInTheDocument();
    expect(screen.getByText('Inputs')).toBeInTheDocument();
    expect(screen.getByText('Outputs')).toBeInTheDocument();
    const contentTab = screen.getByTestId('model-trace-explorer-content-tab');
    expect(contentTab).toHaveTextContent('I was charged twice');
    expect(contentTab).toHaveTextContent('Is this about billing?');
    expect(contentTab).toHaveTextContent('answers');
    expect(contentTab).toHaveTextContent('noul');
    expect(contentTab).toHaveTextContent('angry');
    expect(contentTab).toHaveTextContent('probabilities');
    expect(contentTab).toHaveTextContent('0.98');
    expect(contentTab).toHaveTextContent('score');

    await userEvent.click(screen.getAllByText('Pretty')[1]);
    await userEvent.click(screen.getByText('JSON'));
    expect(contentTab).toHaveTextContent('answers');
    expect(contentTab).toHaveTextContent('probabilities');
    expect(contentTab).toHaveTextContent('confidence');
  });

  it('renders selected span payloads with the pretty field renderers by default', () => {
    const { rerender } = render(
      <ModelTraceExplorerContentTab
        activeSpan={{
          ...DEFAULT_SPAN,
          start: DEFAULT_SPAN.start_time,
          end: DEFAULT_SPAN.end_time,
          key: DEFAULT_SPAN.context.span_id,
          assessments: [],
          traceId: DEFAULT_SPAN.context.trace_id,
        }}
        searchFilter=""
        activeMatch={null}
      />,
      { wrapper: Wrapper },
    );

    expect(screen.queryByTestId('model-trace-explorer-retriever-field-renderer')).not.toBeInTheDocument();

    rerender(<ModelTraceExplorerContentTab activeSpan={MOCK_RETRIEVER_SPAN} searchFilter="" activeMatch={null} />);

    expect(screen.queryByTestId('model-trace-explorer-retriever-field-renderer')).toBeInTheDocument();
    expect(screen.getByTestId('model-trace-explorer-content-tab')).toHaveTextContent('Content with metadata');
  });

  it('renders chat-shaped inputs in pretty mode', async () => {
    render(<ModelTraceExplorerContentTab activeSpan={MOCK_CHAT_SPAN} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    // check that the user text renders
    expect(screen.queryByText('User')).toBeInTheDocument();
    expect(screen.queryByText("What's the weather in Singapore and New York?")).toBeInTheDocument();

    // Outputs render as message cards, without the raw LangChain wrapper fields in pretty mode.
    expect(screen.queryByText('The weather in Singapore is hot, while in New York, it is cold.')).toBeInTheDocument();
    const contentTab = screen.getByTestId('model-trace-explorer-content-tab');
    expect(contentTab).not.toHaveTextContent('generations');
    expect(contentTab).not.toHaveTextContent('llm_output');

    // Tool definitions start collapsed and show their count in the section title.
    expect(screen.queryByText('Tools')).toBeInTheDocument();
    expect(screen.queryByText('(1)')).toBeInTheDocument();
    expect(screen.queryAllByTestId('model-trace-explorer-chat-tool')).toHaveLength(0);
    await userEvent.click(screen.getByTestId('model-trace-explorer-tools-section-toggle'));
    expect(screen.queryAllByTestId('model-trace-explorer-chat-tool')).toHaveLength(1);
    expect(screen.queryByText('Tells a joke')).not.toBeInTheDocument();
    // Expand tool definition detail
    const toolDefinitionToggle = within(screen.getByTestId('model-trace-explorer-chat-tool')).getByTestId(
      'model-trace-explorer-chat-tool-toggle',
    );
    await userEvent.click(toolDefinitionToggle);
    expect(screen.queryByText('Tells a joke')).toBeInTheDocument();
  });

  it('shows raw input and output fields after switching render mode', async () => {
    render(<ModelTraceExplorerContentTab activeSpan={MOCK_CHAT_SPAN} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    expect(screen.queryByText('Inputs')).toBeInTheDocument();
    expect(screen.queryByText('Outputs')).toBeInTheDocument();
    expect(screen.getAllByText('Pretty')).toHaveLength(2);
    expect(screen.queryByText('Table')).not.toBeInTheDocument();
    await userEvent.click(screen.getAllByText('Pretty')[1]);
    await userEvent.click(screen.getByText('JSON'));

    const contentTab = screen.getByTestId('model-trace-explorer-content-tab');
    expect(contentTab).toHaveTextContent('generations');
    expect(contentTab).toHaveTextContent('llm_output');
  });

  it('switches a section to YAML rendering', async () => {
    render(<ModelTraceExplorerContentTab activeSpan={MOCK_CHAT_SPAN} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    await userEvent.click(screen.getAllByText('Pretty')[1]);
    await userEvent.click(screen.getByText('YAML'));

    expect(screen.getByRole('button', { name: 'YAML' })).toBeInTheDocument();
    expect(screen.getByTestId('model-trace-explorer-content-tab')).toHaveTextContent('generations:');
  });
});
