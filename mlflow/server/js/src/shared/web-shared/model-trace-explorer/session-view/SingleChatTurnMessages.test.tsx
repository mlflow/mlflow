import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { SingleChatTurnMessages, extractSimpleChatMessages } from './SingleChatTurnMessages';
import type { ModelTrace, ModelTraceSpanV3 } from '../ModelTrace.types';

const TestWrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

const createSpan = (
  inputs: unknown,
  outputs: unknown,
  extraAttributes: Record<string, unknown> = {},
): ModelTraceSpanV3 => ({
  trace_id: 'trace-1',
  span_id: 'span-1',
  trace_state: '',
  parent_span_id: null,
  name: 'root',
  start_time_unix_nano: '1000000000',
  end_time_unix_nano: '2000000000',
  status: { code: 'STATUS_CODE_OK' },
  attributes: {
    'mlflow.spanType': JSON.stringify('UNKNOWN'),
    'mlflow.spanInputs': JSON.stringify(inputs),
    'mlflow.spanOutputs': JSON.stringify(outputs),
    ...extraAttributes,
  },
});

const createTrace = (span: ModelTraceSpanV3): ModelTrace => ({
  data: { spans: [span] },
  info: {
    trace_id: 'trace-1',
    request_time: '2025-04-19T09:04:07.875Z',
    state: 'OK',
    tags: {},
    assessments: [],
    trace_location: {
      type: 'MLFLOW_EXPERIMENT',
      mlflow_experiment: { experiment_id: 'exp-1' },
    },
  },
});

describe('extractSimpleChatMessages', () => {
  it('extracts messages from object input with query field and string output', () => {
    const result = extractSimpleChatMessages(
      { query: 'What is MLflow?', thread: { messages: ['old msg 1', 'old msg 2'] } },
      'MLflow is an open-source platform.',
    );
    expect(result).toEqual([
      { role: 'user', content: 'What is MLflow?' },
      { role: 'assistant', content: 'MLflow is an open-source platform.' },
    ]);
  });

  it('extracts messages from plain string input and string output', () => {
    const result = extractSimpleChatMessages('Hello there', 'Hi! How can I help?');
    expect(result).toEqual([
      { role: 'user', content: 'Hello there' },
      { role: 'assistant', content: 'Hi! How can I help?' },
    ]);
  });

  it('returns null when output is not a string', () => {
    expect(extractSimpleChatMessages({ query: 'test', messages: [] }, { result: 'object output' })).toBeNull();
  });

  it('returns null when output is empty string', () => {
    expect(extractSimpleChatMessages({ query: 'test', messages: [] }, '')).toBeNull();
  });

  it('returns null when object input has no messages key', () => {
    expect(extractSimpleChatMessages({ query: 'test', config: {} }, 'response')).toBeNull();
  });

  it('returns null when no recognizable query field in inputs', () => {
    expect(extractSimpleChatMessages({ thread: { messages: [] }, config: {} }, 'response')).toBeNull();
  });

  it('returns null when inputs is null', () => {
    expect(extractSimpleChatMessages(null, 'response')).toBeNull();
  });

  it('recognizes all supported query field names', () => {
    for (const field of ['query', 'input', 'message', 'question', 'prompt', 'content']) {
      const result = extractSimpleChatMessages({ [field]: 'user text', messages: [] }, 'assistant text');
      expect(result).toEqual([
        { role: 'user', content: 'user text' },
        { role: 'assistant', content: 'assistant text' },
      ]);
    }
  });

  it('picks the first matching field by priority order', () => {
    const result = extractSimpleChatMessages({ query: 'from query', input: 'from input', messages: [] }, 'response');
    expect(result).toEqual([
      { role: 'user', content: 'from query' },
      { role: 'assistant', content: 'response' },
    ]);
  });
});

describe('SingleChatTurnMessages', () => {
  it('renders chat bubbles for LangGraph-style inputs with string output', () => {
    const span = createSpan(
      { query: 'What is MLflow?', thread: { messages: ['old'] } },
      'MLflow is an open-source platform.',
    );

    render(
      <TestWrapper>
        <SingleChatTurnMessages trace={createTrace(span)} />
      </TestWrapper>,
    );

    expect(screen.getByText('What is MLflow?')).toBeInTheDocument();
    expect(screen.getByText('MLflow is an open-source platform.')).toBeInTheDocument();
  });

  it('renders chat bubbles for simple string input field and string output', () => {
    const span = createSpan({ input: 'Tell me about tracing', messages: [] }, 'Tracing helps you debug.');

    render(
      <TestWrapper>
        <SingleChatTurnMessages trace={createTrace(span)} />
      </TestWrapper>,
    );

    expect(screen.getByText('Tell me about tracing')).toBeInTheDocument();
    expect(screen.getByText('Tracing helps you debug.')).toBeInTheDocument();
  });

  it('uses existing chatMessages path when available', () => {
    const span = createSpan({ query: 'should not appear' }, 'should not appear either', {
      'mlflow.spanType': JSON.stringify('CHAT_MODEL'),
      'mlflow.chat.messages': JSON.stringify([
        { role: 'user', content: 'parsed question' },
        { role: 'assistant', content: 'parsed answer' },
      ]),
    });

    render(
      <TestWrapper>
        <SingleChatTurnMessages trace={createTrace(span)} />
      </TestWrapper>,
    );

    expect(screen.getByText('parsed question')).toBeInTheDocument();
    expect(screen.getByText('parsed answer')).toBeInTheDocument();
    expect(screen.queryByText('should not appear')).not.toBeInTheDocument();
  });

  it('falls through to raw display when output is an object', () => {
    const span = createSpan({ query: 'test query' }, { result: 'structured output' });

    render(
      <TestWrapper>
        <SingleChatTurnMessages trace={createTrace(span)} />
      </TestWrapper>,
    );

    expect(screen.getByText('Inputs')).toBeInTheDocument();
    expect(screen.getByText('Outputs')).toBeInTheDocument();
  });

  it('falls through to raw display when no recognizable query field exists', () => {
    const span = createSpan({ thread: { messages: [] }, config: { model: 'gpt-4' } }, 'some response');

    render(
      <TestWrapper>
        <SingleChatTurnMessages trace={createTrace(span)} />
      </TestWrapper>,
    );

    expect(screen.getByText('Inputs')).toBeInTheDocument();
    expect(screen.getByText('Outputs')).toBeInTheDocument();
  });
});
