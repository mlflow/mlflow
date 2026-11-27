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

const createSpanWithId = ({
  spanId,
  parentSpanId,
  name,
  inputs,
  outputs,
  extraAttributes = {},
}: {
  spanId: string;
  parentSpanId: string | null;
  name: string;
  inputs?: unknown;
  outputs?: unknown;
  extraAttributes?: Record<string, unknown>;
}): ModelTraceSpanV3 => ({
  trace_id: 'trace-1',
  span_id: spanId,
  trace_state: '',
  parent_span_id: parentSpanId,
  name,
  start_time_unix_nano: '1000000000',
  end_time_unix_nano: '2000000000',
  status: { code: 'STATUS_CODE_OK' },
  attributes: {
    'mlflow.spanType': JSON.stringify('UNKNOWN'),
    ...(inputs === undefined ? {} : { 'mlflow.spanInputs': JSON.stringify(inputs) }),
    ...(outputs === undefined ? {} : { 'mlflow.spanOutputs': JSON.stringify(outputs) }),
    ...extraAttributes,
  },
});

const createTraceFromSpans = (spans: ModelTraceSpanV3[]): ModelTrace => ({
  data: { spans },
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

  it('falls back to a child span when the root has no inputs or outputs', () => {
    // OpenTelemetry GenAI instrumentation puts the messages on the LLM span,
    // which is a child of the agent span that starts the trace.
    const root = createSpanWithId({ spanId: 'span-root', parentSpanId: null, name: 'invoke_agent' });
    const child = createSpanWithId({
      spanId: 'span-chat',
      parentSpanId: 'span-root',
      name: 'chat',
      inputs: [{ role: 'user', content: 'This is an example Question?' }],
      outputs: [{ role: 'assistant', content: 'This is an example answer.' }],
    });

    render(
      <TestWrapper>
        <SingleChatTurnMessages trace={createTraceFromSpans([root, child])} />
      </TestWrapper>,
    );

    expect(screen.getByText('This is an example Question?')).toBeInTheDocument();
    expect(screen.getByText('This is an example answer.')).toBeInTheDocument();
  });

  it('prefers the root span over a child that also has content', () => {
    const root = createSpanWithId({
      spanId: 'span-root',
      parentSpanId: null,
      name: 'invoke_agent',
      inputs: [{ role: 'user', content: 'root question' }],
      outputs: [{ role: 'assistant', content: 'root answer' }],
    });
    const child = createSpanWithId({
      spanId: 'span-chat',
      parentSpanId: 'span-root',
      name: 'chat',
      inputs: [{ role: 'user', content: 'child question' }],
      outputs: [{ role: 'assistant', content: 'child answer' }],
    });

    render(
      <TestWrapper>
        <SingleChatTurnMessages trace={createTraceFromSpans([root, child])} />
      </TestWrapper>,
    );

    expect(screen.getByText('root question')).toBeInTheDocument();
    expect(screen.queryByText('child question')).not.toBeInTheDocument();
  });

  it('renders the empty root sections when no span in the trace has content', () => {
    const root = createSpanWithId({ spanId: 'span-root', parentSpanId: null, name: 'invoke_agent' });
    const child = createSpanWithId({ spanId: 'span-child', parentSpanId: 'span-root', name: 'noop' });

    render(
      <TestWrapper>
        <SingleChatTurnMessages trace={createTraceFromSpans([root, child])} />
      </TestWrapper>,
    );

    expect(screen.getByText('Inputs')).toBeInTheDocument();
    expect(screen.getByText('Outputs')).toBeInTheDocument();
  });

  it('picks the shallowest span with content', () => {
    const root = createSpanWithId({ spanId: 'span-root', parentSpanId: null, name: 'invoke_agent' });
    const middle = createSpanWithId({
      spanId: 'span-middle',
      parentSpanId: 'span-root',
      name: 'retrieve',
      inputs: [{ role: 'user', content: 'shallow question' }],
      outputs: [{ role: 'assistant', content: 'shallow answer' }],
    });
    const deep = createSpanWithId({
      spanId: 'span-deep',
      parentSpanId: 'span-middle',
      name: 'chat',
      inputs: [{ role: 'user', content: 'deep question' }],
      outputs: [{ role: 'assistant', content: 'deep answer' }],
    });

    render(
      <TestWrapper>
        <SingleChatTurnMessages trace={createTraceFromSpans([root, middle, deep])} />
      </TestWrapper>,
    );

    expect(screen.getByText('shallow question')).toBeInTheDocument();
    expect(screen.queryByText('deep question')).not.toBeInTheDocument();
  });
});
