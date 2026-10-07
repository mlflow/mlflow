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
  it('renders TypeSafe decision cards while preserving the full trace in JSON', async () => {
    const inputs = {
      state: 'I was charged twice',
      evaluation_context: { customer_tier: 'enterprise' },
      questions: {
        billing: { type: 'noul', instructions: 'Is this about billing?' },
        tone: { type: 'choice', criteria: { calm: null, angry: null } },
        urgency: { type: 'score', criteria: ['low', 'high'] },
      },
      temperature: 0,
      model: 'jev-latest',
    };
    const outputs = {
      model: 'jev-1.13.0',
      usage: { input_tokens: 10, output_tokens: 0 },
      answers: {
        billing: { type: 'noul', noul: 0.98 },
        tone: { type: 'choice', choice: 'angry', confidence: 0.96, probabilities: { calm: 0.02, angry: 0.98 } },
        urgency: {
          type: 'score',
          score: 0.9,
          confidence: 0.94,
          legend: { 0: 'low', 1: 'high' },
          probabilities: { 0: 0.1, 1: 0.9 },
        },
      },
      debug_info: { cache_key: 'billing-refund' },
    };
    const tokenUsage = { input_tokens: 10, output_tokens: 0, total_tokens: 10 };
    const span = normalizeNewSpanData(
      {
        ...DEFAULT_SPAN,
        name: 'typesafe.system_one',
        attributes: {
          'mlflow.spanType': JSON.stringify('LLM'),
          'mlflow.spanInputs': JSON.stringify(inputs),
          'mlflow.spanOutputs': JSON.stringify(outputs),
          'mlflow.message.format': JSON.stringify('typesafe'),
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
    expect(span.chatMessageFormat).toBe('typesafe');
    expect(span.modelName).toBe('jev-1.13.0');
    expect(span.tokenUsage).toEqual(tokenUsage);
    expect(span.chatMessages).toBeUndefined();
    expect(getDefaultActiveTab(span)).toBe('content');

    const renderTrace = (searchFilter: string) => (
      <>
        <div data-testid="model-badge">
          <SpanModelCostBadge activeSpan={span} />
        </div>
        <ModelTraceExplorerContentTab activeSpan={span} searchFilter={searchFilter} activeMatch={null} />
      </>
    );
    const { rerender } = render(renderTrace(''), { wrapper: Wrapper });

    const modelBadge = within(screen.getByTestId('model-badge'));
    expect(modelBadge.getByText('Model')).toBeInTheDocument();
    expect(modelBadge.getByText('jev-1.13.0')).toBeInTheDocument();
    expect(screen.getByText('Inputs')).toBeInTheDocument();
    expect(screen.getByText('Outputs')).toBeInTheDocument();
    const contentTab = screen.getByTestId('model-trace-explorer-content-tab');
    const richQuestions = within(screen.getByTestId('decision-questions'));
    expect(screen.getByText('state', { exact: true })).toBeInTheDocument();
    expect(screen.getByText('questions', { exact: true })).toBeInTheDocument();
    expect(richQuestions.getByText('billing')).toBeInTheDocument();
    expect(richQuestions.getByText('Is this about billing?')).toBeInTheDocument();
    expect(contentTab).toHaveTextContent('I was charged twice');
    expect(contentTab).not.toHaveTextContent('evaluation_context');
    expect(contentTab).not.toHaveTextContent('customer_tier');
    expect(contentTab).not.toHaveTextContent('temperature');
    expect(contentTab).not.toHaveTextContent('jev-latest');
    expect(contentTab).not.toHaveTextContent('Request');
    expect(contentTab).not.toHaveTextContent('Additional inputs');

    const richAnswers = within(screen.getByTestId('decision-answers'));
    expect(richAnswers.queryByText('Is this about billing?')).not.toBeInTheDocument();
    const getAnswerSummary = (id: string) => {
      const summary = richAnswers.getByText(id, { exact: true }).closest('summary');
      expect(summary).not.toBeNull();
      return summary as HTMLElement;
    };
    const noulSummary = getAnswerSummary('billing');
    const choiceSummary = getAnswerSummary('tone');
    const scoreSummary = getAnswerSummary('urgency');
    const noulAnswer = within(noulSummary);
    expect(noulAnswer.getByText('true')).toBeInTheDocument();
    expect(noulAnswer.getByText('98% probability')).toBeInTheDocument();
    expect(noulAnswer.queryByText(/confidence/i)).not.toBeInTheDocument();
    const choiceAnswer = within(choiceSummary);
    expect(choiceAnswer.getByText('angry')).toBeInTheDocument();
    expect(choiceAnswer.getByText('96% confidence')).toBeInTheDocument();
    const scoreAnswer = within(scoreSummary);
    expect(scoreAnswer.getByText('0.9')).toBeInTheDocument();
    expect(scoreAnswer.getByText('Range 0–1')).toBeInTheDocument();
    expect(scoreAnswer.getByText('94% confidence')).toBeInTheDocument();
    expect(within(contentTab).queryByText('answers', { exact: true })).not.toBeInTheDocument();
    expect(contentTab).not.toHaveTextContent('jev-1.13.0');
    expect(contentTab).not.toHaveTextContent('usage');
    expect(contentTab).not.toHaveTextContent('input_tokens');
    expect(contentTab).not.toHaveTextContent('debug_info');
    expect(contentTab).not.toHaveTextContent('billing-refund');

    await userEvent.click(noulSummary);
    expect(noulSummary.closest('details')).toHaveAttribute('open');
    expect(richAnswers.getByRole('progressbar', { name: 'Probability for true' })).toHaveAttribute(
      'aria-valuenow',
      '98',
    );
    expect(richAnswers.getByRole('progressbar', { name: 'Probability for false' })).toHaveAttribute(
      'aria-valuetext',
      '2%',
    );

    await userEvent.click(choiceSummary);
    const choiceDisclosure = choiceSummary.closest('details');
    expect(choiceDisclosure).toHaveAttribute('open');
    const choiceDetails = within(choiceDisclosure as HTMLElement);
    expect(choiceDetails.getByText('Probability distribution')).toBeInTheDocument();
    expect(richAnswers.getByRole('progressbar', { name: 'Probability for angry' })).toHaveAttribute(
      'aria-valuenow',
      '98',
    );
    expect(richAnswers.getByRole('progressbar', { name: 'Probability for calm' })).toHaveAttribute(
      'aria-valuenow',
      '2',
    );
    expect(choiceDetails.getByText('Confidence')).toBeInTheDocument();

    await userEvent.click(scoreSummary);
    expect(scoreSummary.closest('details')).toHaveAttribute('open');
    expect(richAnswers.getByRole('progressbar', { name: 'Probability for score 0' })).toHaveAttribute(
      'aria-valuenow',
      '10',
    );
    expect(richAnswers.getByRole('progressbar', { name: 'Probability for score 1' })).toHaveAttribute(
      'aria-valuenow',
      '90',
    );
    expect(richAnswers.getByText('low')).toBeInTheDocument();
    expect(richAnswers.getByText('high')).toBeInTheDocument();

    rerender(renderTrace('billing-refund'));
    expect(screen.queryByTestId('decision-questions')).not.toBeInTheDocument();
    expect(screen.queryByTestId('decision-answers')).not.toBeInTheDocument();
    expect(contentTab).toHaveTextContent('answers');
    expect(contentTab).toHaveTextContent('debug_info');
    expect(contentTab).toHaveTextContent('billing-refund');

    rerender(renderTrace(''));
    expect(screen.getByTestId('decision-questions')).toBeInTheDocument();
    expect(screen.getByTestId('decision-answers')).toBeInTheDocument();

    await userEvent.click(screen.getAllByText('Pretty')[1]);
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'JSON' }));
    expect(screen.getByTestId('decision-questions')).toBeInTheDocument();
    expect(screen.queryByTestId('decision-answers')).not.toBeInTheDocument();
    expect(contentTab).toHaveTextContent('answers');
    expect(contentTab).toHaveTextContent('probabilities');
    expect(contentTab).toHaveTextContent('confidence');
    expect(contentTab).toHaveTextContent('jev-1.13.0');
    expect(contentTab).toHaveTextContent('input_tokens');
    expect(contentTab).toHaveTextContent('debug_info');

    await userEvent.click(screen.getByText('Pretty'));
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'JSON' }));
    expect(screen.queryByTestId('decision-questions')).not.toBeInTheDocument();
    expect(contentTab).toHaveTextContent('evaluation_context');
    expect(contentTab).toHaveTextContent('customer_tier');
    expect(contentTab).toHaveTextContent('temperature');
    expect(contentTab).toHaveTextContent('jev-latest');
  });

  it('falls back to generic fields for a TypeSafe custom response without standard answers', () => {
    const inputs = {
      state: 'custom response input',
      model: 'jev-custom-alias',
      questions: { safety: { type: 'noul', instructions: 'Is this safe?' } },
      evaluation_context: { policy: 'strict' },
      temperature: 0,
    };
    const outputs = {
      decision: 'allow',
      response_metadata: { request_id: 'req-custom-response' },
    };
    const span = normalizeNewSpanData(
      {
        ...DEFAULT_SPAN,
        name: 'typesafe.system_one',
        attributes: {
          'mlflow.spanType': JSON.stringify('LLM'),
          'mlflow.spanInputs': JSON.stringify(inputs),
          'mlflow.spanOutputs': JSON.stringify(outputs),
          'mlflow.message.format': JSON.stringify('typesafe'),
          'mlflow.llm.model': JSON.stringify('jev-custom-alias'),
        },
      },
      0,
      0,
      [],
      {},
      'custom-response-trace',
    );

    render(<ModelTraceExplorerContentTab activeSpan={span} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    const contentTab = screen.getByTestId('model-trace-explorer-content-tab');
    expect(screen.queryByTestId('decision-questions')).not.toBeInTheDocument();
    expect(screen.queryByTestId('decision-answers')).not.toBeInTheDocument();
    expect(contentTab).toHaveTextContent('custom response input');
    expect(contentTab).toHaveTextContent('jev-custom-alias');
    expect(contentTab).toHaveTextContent('Is this safe?');
    expect(contentTab).toHaveTextContent('evaluation_context');
    expect(contentTab).toHaveTextContent('strict');
    expect(contentTab).toHaveTextContent('temperature');
    expect(contentTab).toHaveTextContent('decision');
    expect(contentTab).toHaveTextContent('allow');
    expect(contentTab).toHaveTextContent('response_metadata');
    expect(contentTab).toHaveTextContent('req-custom-response');
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

  it('does not duplicate top-level OTEL chat messages in pretty mode', () => {
    const span = {
      key: 'span-1',
      inputs: [{ role: 'user', parts: [{ type: 'text', content: 'Gibt es klassifizierte Artikel?' }] }],
      outputs: [
        {
          role: 'assistant',
          parts: [{ type: 'text', content: 'Nein, es gibt aktuell keine klassifizierten Artikel im System.' }],
        },
      ],
      attributes: {},
      assessments: [],
    } as any;

    const { rerender } = render(<ModelTraceExplorerContentTab activeSpan={span} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    expect(screen.getAllByText('Gibt es klassifizierte Artikel?')).toHaveLength(1);
    expect(screen.getAllByText('Nein, es gibt aktuell keine klassifizierten Artikel im System.')).toHaveLength(1);
    expect(screen.getAllByText('User')).toHaveLength(1);
    expect(screen.getAllByText('Assistant')).toHaveLength(1);

    rerender(
      <ModelTraceExplorerContentTab
        activeSpan={span}
        searchFilter="Gibt"
        activeMatch={{
          span,
          section: 'inputs',
          key: '',
          isKeyMatch: false,
          matchIndex: 0,
        }}
      />,
    );

    expect(screen.getAllByText('Gibt es klassifizierte Artikel?')).toHaveLength(1);
    expect(screen.getAllByText('User')).toHaveLength(1);

    rerender(
      <ModelTraceExplorerContentTab
        activeSpan={span}
        searchFilter="Nein"
        activeMatch={{
          span,
          section: 'outputs',
          key: '',
          isKeyMatch: false,
          matchIndex: 0,
        }}
      />,
    );

    expect(screen.getAllByText('Nein, es gibt aktuell keine klassifizierten Artikel im System.')).toHaveLength(1);
    expect(screen.getAllByText('Assistant')).toHaveLength(1);
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
