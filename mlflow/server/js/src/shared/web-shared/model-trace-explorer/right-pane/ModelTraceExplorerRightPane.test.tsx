import { describe, it, expect } from '@jest/globals';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerChatTab } from './ModelTraceExplorerChatTab';
import { ModelTraceExplorerContentTab } from './ModelTraceExplorerContentTab';
import type { ModelTraceSpan, ModelTraceSpanNode } from '../ModelTrace.types';
import {
  mockSpans,
  MOCK_RETRIEVER_SPAN,
  MOCK_CHAT_SPAN,
  MOCK_CHAT_MESSAGES,
  MOCK_CHAT_TOOLS,
} from '../ModelTraceExplorer.test-utils';
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
  it('renders separate TypeSafe System One inputs and answers in the default mode and preserves raw modes', async () => {
    const span: ModelTraceSpanNode = {
      ...DEFAULT_SPAN,
      title: 'typesafe.system_one',
      start: DEFAULT_SPAN.start_time,
      end: DEFAULT_SPAN.end_time,
      key: DEFAULT_SPAN.context.span_id,
      assessments: [],
      traceId: DEFAULT_SPAN.context.trace_id,
      chatMessageFormat: 'typesafe',
      modelProvider: 'typesafe',
      inputs: {
        state: 'I was charged twice',
        evaluation_context: { customer_tier: 'enterprise' },
        questions: {
          category: {
            type: 'choice',
            instructions: 'Which queue should handle this request?',
            criteria: { billing: null, technical: null },
          },
          duplicate_charge: { type: 'noul', instructions: 'Is this a duplicate charge?' },
        },
        temperature: 0,
        model: 'jev-latest',
      },
      outputs: {
        model: 'jev-1.13.0',
        usage: { input_tokens: 10, output_tokens: 0 },
        answers: {
          category: {
            type: 'choice',
            choice: 'billing',
            confidence: 0.61,
            probabilities: { billing: 0.74, technical: 0.26 },
          },
          duplicate_charge: { type: 'noul', noul: 0.99 },
        },
        debug_info: { cache_key: 'billing-refund' },
      },
    };

    render(<ModelTraceExplorerContentTab activeSpan={span} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    const contentTab = screen.getByTestId('model-trace-explorer-content-tab');
    const richQuestions = within(screen.getByTestId('decision-questions'));
    expect(screen.getByText('state', { exact: true })).toBeInTheDocument();
    expect(screen.getByText('questions', { exact: true })).toBeInTheDocument();
    expect(richQuestions.getByText('category')).toBeInTheDocument();
    expect(richQuestions.getByText('Which queue should handle this request?')).toBeInTheDocument();
    expect(contentTab).toHaveTextContent('I was charged twice');
    expect(contentTab).not.toHaveTextContent('evaluation_context');
    expect(contentTab).not.toHaveTextContent('customer_tier');
    expect(contentTab).not.toHaveTextContent('temperature');
    expect(contentTab).not.toHaveTextContent('jev-latest');
    expect(contentTab).not.toHaveTextContent('Request');
    expect(contentTab).not.toHaveTextContent('Additional inputs');

    const richAnswers = within(screen.getByTestId('decision-answers'));
    expect(richAnswers.getByRole('button', { name: /^category\b/ })).toHaveTextContent('billing');
    expect(richAnswers.queryByText('Which queue should handle this request?')).not.toBeInTheDocument();
    const noulAnswer = within(richAnswers.getByRole('button', { name: /^duplicate_charge\b/ }));
    expect(noulAnswer.getByText('true')).toBeInTheDocument();
    expect(noulAnswer.getByText('99% probability')).toBeInTheDocument();
    expect(noulAnswer.queryByText(/confidence/i)).not.toBeInTheDocument();
    expect(within(contentTab).queryByText('answers', { exact: true })).not.toBeInTheDocument();
    expect(contentTab).not.toHaveTextContent('jev-1.13.0');
    expect(contentTab).not.toHaveTextContent('usage');
    expect(contentTab).not.toHaveTextContent('input_tokens');
    expect(contentTab).not.toHaveTextContent('debug_info');
    expect(contentTab).not.toHaveTextContent('billing-refund');

    await userEvent.click(screen.getByText('JSON'));
    expect(screen.queryByTestId('decision-questions')).not.toBeInTheDocument();
    expect(screen.queryByTestId('decision-answers')).not.toBeInTheDocument();
    expect(contentTab).toHaveTextContent('probabilities');
    expect(contentTab).toHaveTextContent('evaluation_context');
    expect(contentTab).toHaveTextContent('temperature');
    expect(contentTab).toHaveTextContent('jev-1.13.0');
    expect(contentTab).toHaveTextContent('input_tokens');
  });

  it('does not select the decision renderer from the TypeSafe span title and provider alone', () => {
    const span: ModelTraceSpanNode = {
      ...DEFAULT_SPAN,
      title: 'typesafe.system_one',
      start: DEFAULT_SPAN.start_time,
      end: DEFAULT_SPAN.end_time,
      key: DEFAULT_SPAN.context.span_id,
      assessments: [],
      traceId: DEFAULT_SPAN.context.trace_id,
      modelProvider: 'typesafe',
      inputs: {
        state: 'I was charged twice',
        questions: { duplicate_charge: { type: 'noul', instructions: 'Is this a duplicate charge?' } },
      },
      outputs: { answers: { duplicate_charge: { type: 'noul', noul: 0.99 } } },
    };

    render(<ModelTraceExplorerContentTab activeSpan={span} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    expect(screen.queryByTestId('decision-questions')).not.toBeInTheDocument();
    expect(screen.queryByTestId('decision-answers')).not.toBeInTheDocument();
    expect(screen.getByTestId('model-trace-explorer-content-tab')).toHaveTextContent('answers');
  });

  it('falls back to generic fields for a TypeSafe custom response without standard answers', () => {
    const span: ModelTraceSpanNode = {
      ...DEFAULT_SPAN,
      title: 'typesafe.system_one',
      start: DEFAULT_SPAN.start_time,
      end: DEFAULT_SPAN.end_time,
      key: DEFAULT_SPAN.context.span_id,
      assessments: [],
      traceId: DEFAULT_SPAN.context.trace_id,
      chatMessageFormat: 'typesafe',
      modelProvider: 'typesafe',
      inputs: {
        state: 'custom response input',
        model: 'jev-custom-alias',
        questions: { safety: { type: 'noul', instructions: 'Is this safe?' } },
        evaluation_context: { policy: 'strict' },
        temperature: 0,
      },
      outputs: {
        decision: 'allow',
        response_metadata: { request_id: 'req-custom-response' },
      },
    };

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

  it('switches between span renderers appropriately', () => {
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
  });

  it('should render conversations if possible', async () => {
    const MOCK_SPAN: ModelTraceSpanNode = {
      ...DEFAULT_SPAN,
      start: DEFAULT_SPAN.start_time,
      end: DEFAULT_SPAN.end_time,
      key: DEFAULT_SPAN.context.span_id,
      assessments: [],
      traceId: DEFAULT_SPAN.context.trace_id,
      chatMessages: MOCK_CHAT_MESSAGES,
      chatTools: MOCK_CHAT_TOOLS,
    };
    render(<ModelTraceExplorerChatTab activeSpan={MOCK_SPAN} />, {
      wrapper: Wrapper,
    });

    // check that the user text renders
    expect(screen.queryByText('User')).toBeInTheDocument();
    expect(screen.queryByText('tell me a joke in 50 words')).toBeInTheDocument();

    // check that the tool calls render
    expect(screen.queryByText('Assistant')).toBeInTheDocument();
    expect(screen.queryAllByText('tell_joke')).toHaveLength(2); // one in input, one in tool definition

    // check that the tool result render
    expect(screen.queryByText('Tool')).toBeInTheDocument();
    expect(
      screen.queryByText('Why did the scarecrow win an award? Because he was outstanding in his field!'),
    ).toBeInTheDocument();

    // check that the tool definition render
    expect(screen.queryAllByTestId('model-trace-explorer-chat-tool')).toHaveLength(1);
    expect(screen.queryByText('Tells a joke')).not.toBeInTheDocument();
    // Expand tool definition detail
    const toolDefinitionToggle = screen.queryAllByTestId('model-trace-explorer-chat-tool-toggle')[0];
    await userEvent.click(toolDefinitionToggle);
    expect(screen.queryByText('Tells a joke')).toBeInTheDocument();
  });

  it('shows raw input and output of spans', async () => {
    render(<ModelTraceExplorerContentTab activeSpan={MOCK_CHAT_SPAN} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    expect(screen.queryByText('Inputs')).toBeInTheDocument();
    expect(screen.queryByText('Outputs')).toBeInTheDocument();
    expect(screen.queryByText('generations')).toBeInTheDocument();
    expect(screen.queryByText('llm_output')).toBeInTheDocument();
    expect(screen.queryAllByText('See more').length).toBeGreaterThanOrEqual(1);
  });
});
