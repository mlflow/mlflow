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
  it('extracts user/assistant from LangGraph messages format', () => {
    const result = extractSimpleChatMessages(
      {
        messages: [
          { type: 'system', content: 'You are helpful.' },
          { type: 'human', content: 'What is MLflow?' },
        ],
      },
      {
        messages: [
          { type: 'system', content: 'You are helpful.' },
          { type: 'human', content: 'What is MLflow?' },
          { type: 'ai', content: 'MLflow is an open-source platform.' },
        ],
      },
    );
    expect(result).toEqual([
      { role: 'user', content: 'What is MLflow?' },
      { role: 'assistant', content: 'MLflow is an open-source platform.' },
    ]);
  });

  it('skips intermediate tool-calling assistant messages in outputs', () => {
    const result = extractSimpleChatMessages(
      {
        messages: [{ type: 'human', content: 'What is RLM?' }],
      },
      {
        messages: [
          { type: 'human', content: 'What is RLM?' },
          {
            type: 'ai',
            content: '',
            tool_calls: [{ name: 'web_search', args: { query: 'RLM' }, id: 'call_1' }],
          },
          { type: 'tool', content: 'Search results...', tool_call_id: 'call_1' },
          { type: 'ai', content: 'RLM stands for Recursive Language Models.' },
        ],
      },
    );
    expect(result).toEqual([
      { role: 'user', content: 'What is RLM?' },
      { role: 'assistant', content: 'RLM stands for Recursive Language Models.' },
    ]);
  });

  it('handles LangGraph input messages with string output fallback', () => {
    const result = extractSimpleChatMessages(
      {
        messages: [{ type: 'human', content: 'Hello' }],
      },
      'Hi! How can I help?',
    );
    expect(result).toEqual([
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi! How can I help?' },
    ]);
  });

  it('extracts messages from plain string input and string output', () => {
    const result = extractSimpleChatMessages('Hello there', 'Hi! How can I help?');
    expect(result).toEqual([
      { role: 'user', content: 'Hello there' },
      { role: 'assistant', content: 'Hi! How can I help?' },
    ]);
  });

  it('returns null when outputs have no assistant message', () => {
    expect(
      extractSimpleChatMessages(
        { messages: [{ type: 'human', content: 'test' }] },
        { messages: [{ type: 'human', content: 'test' }] },
      ),
    ).toBeNull();
  });

  it('returns null when inputs have no user message', () => {
    expect(
      extractSimpleChatMessages(
        { messages: [{ type: 'system', content: 'You are helpful.' }] },
        { messages: [{ type: 'ai', content: 'response' }] },
      ),
    ).toBeNull();
  });

  it('returns null when inputs is null', () => {
    expect(extractSimpleChatMessages(null, 'response')).toBeNull();
  });

  it('returns null when inputs is non-messages object and output is non-string', () => {
    expect(extractSimpleChatMessages({ config: {} }, { result: 'object output' })).toBeNull();
  });

  it('returns null when string input but non-string output', () => {
    expect(extractSimpleChatMessages('hello', { result: 'object' })).toBeNull();
  });
});

describe('SingleChatTurnMessages', () => {
  it('renders chat bubbles for LangGraph-style messages format', () => {
    const span = createSpan(
      {
        messages: [
          { type: 'system', content: 'You are helpful.' },
          { type: 'human', content: 'What is MLflow?' },
        ],
      },
      {
        messages: [
          { type: 'system', content: 'You are helpful.' },
          { type: 'human', content: 'What is MLflow?' },
          { type: 'ai', content: 'MLflow is an open-source platform.' },
        ],
      },
    );

    render(
      <TestWrapper>
        <SingleChatTurnMessages trace={createTrace(span)} />
      </TestWrapper>,
    );

    expect(screen.getByText('What is MLflow?')).toBeInTheDocument();
    expect(screen.getByText('MLflow is an open-source platform.')).toBeInTheDocument();
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

  it('falls through to raw display when no messages format is detected', () => {
    const span = createSpan({ config: { model: 'gpt-4' } }, { result: 'structured output' });

    render(
      <TestWrapper>
        <SingleChatTurnMessages trace={createTrace(span)} />
      </TestWrapper>,
    );

    expect(screen.getByText('Inputs')).toBeInTheDocument();
    expect(screen.getByText('Outputs')).toBeInTheDocument();
  });
});
