import { describe, it, expect } from '@jest/globals';

import { normalizeConversation } from '../ModelTraceExplorer.utils';
import { formatDspySections } from './dspy';

const MOCK_DSPY_INPUT = {
  messages: [
    {
      role: 'system',
      content:
        'Your input fields are:\n1. `passage` (str): a passage to summarize\n\nYour output fields are:\n1. `reasoning` (str)\n2. `summary` (str): a one-line summary of the passage\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## passage ## ]]\n{passage}\n\n[[ ## reasoning ## ]]\n{reasoning}\n\n[[ ## summary ## ]]\n{summary}\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is: \n        Given a passage, generate a summary.',
    },
    {
      role: 'user',
      content:
        "[[ ## passage ## ]]\nMLflow Tracing is a feature that enhances LLM observability in your Generative AI (GenAI) applications by capturing detailed information about the execution of your application's services. Tracing provides a way to record the inputs, outputs, and metadata associated with each intermediate step of a request, enabling you to easily pinpoint the source of bugs and unexpected behaviors.\n\nRespond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## summary ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
    },
  ],
  prompt: null,
};

const MOCK_DSPY_OUTPUT = [
  '[[ ## reasoning ## ]]\nThe passage explains the functionality of MLflow Tracing in the context of Generative AI applications. It highlights how this feature improves observability by recording detailed information about the execution process, which aids in identifying bugs and unexpected behaviors. The emphasis is on the benefits of capturing inputs, outputs, and metadata for each step of a request.\n\n[[ ## summary ## ]]\nMLflow Tracing enhances observability in Generative AI applications by recording detailed execution information, helping to identify bugs and unexpected behaviors.\n\n[[ ## completed ## ]]',
];

describe('normalizeConversation', () => {
  it('should handle dspy input', () => {
    const conv = normalizeConversation(MOCK_DSPY_INPUT, 'dspy');
    expect(conv).toEqual([
      expect.objectContaining({
        role: 'system',
        content: expect.stringContaining('Your input fields are:'),
      }),
      expect.objectContaining({
        role: 'user',
        content: expect.stringContaining('#### Passage'),
      }),
    ]);
    // Standalone markers should be converted to headings (first one has no rule above)
    expect(conv?.[0].content).toContain('#### Passage');
    expect(conv?.[0].content).toContain('#### Reasoning');
    expect(conv?.[0].content).toContain('#### Summary');
    // The completed marker should be removed
    expect(conv?.[0].content).not.toContain('[[ ## completed ## ]]');
    // Inline references inside backticks should be preserved
    expect(conv?.[1].content).toContain('`[[ ## reasoning ## ]]`');
    expect(conv?.[1].content).toContain('`[[ ## summary ## ]]`');
    expect(conv?.[1].content).toContain('`[[ ## completed ## ]]`');
    // Ensure single newlines are converted to hard breaks for markdown rendering
    expect(conv?.[0].content).toContain('  \n');
  });

  it('should handle dspy output', () => {
    const conv = normalizeConversation(MOCK_DSPY_OUTPUT, 'dspy');
    expect(conv).toEqual([
      expect.objectContaining({
        content: expect.stringContaining('#### Reasoning'),
        role: 'assistant',
      }),
    ]);
    expect(conv?.[0].content).toContain('#### Summary');
    // completed marker should be removed
    expect(conv?.[0].content).not.toContain('[[ ## completed ## ]]');
    expect(conv?.[0].content).toContain('  \n');
  });

  it('should render JSON output as a formatted code block', () => {
    const jsonOutput = ['{ "summary": "MLflow is great." }'];
    const conv = normalizeConversation(jsonOutput, 'dspy');
    expect(conv).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: '```json\n{\n  "summary": "MLflow is great."\n}\n```',
      }),
    ]);
  });
});

// LM span payloads recorded by DSPy 3.5, which calls LMs with lm15 `Request`/`Response` objects
const MOCK_DSPY_LM15_INPUT = {
  prompt: {
    model: 'openai/gpt-4o-mini',
    messages: [
      {
        role: 'user',
        parts: [
          {
            text: '<question>\nWhat is the capital of France?\n</question>\n\nOutput format (replace placeholders):\n<answer>\n{answer}\n</answer>',
            continuation: [],
            type: 'text',
          },
        ],
        continuation: [],
      },
    ],
    system:
      'Your input fields are:\n1. `question` (str):\nYour output fields are:\n1. `answer` (str):\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `answer`.',
    tools: [],
    config: { max_tokens: null, temperature: null, stop: [] },
  },
  options: {},
  cache: null,
  rollout_id: null,
};

const MOCK_DSPY_LM15_OUTPUT = {
  id: 'resp_1',
  model: 'openai/gpt-4o-mini',
  message: {
    role: 'assistant',
    parts: [
      { text: 'The user wants a capital city.', continuation: [], type: 'thinking' },
      {
        text: '[[ ## plan ## ]]\nRecall the capital.\n\n[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]',
        continuation: [],
        type: 'text',
      },
    ],
    continuation: [],
  },
  finish_reason: 'stop',
  usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
  logprobs: null,
  provider_data: null,
  adaptations: [],
  logprobs_complete: true,
};

// Second turn of a ReAct program with native function calling
const MOCK_DSPY_LM15_TOOL_CALLING_INPUT = {
  prompt: {
    model: 'openai/gpt-4o-mini',
    messages: [
      {
        role: 'user',
        parts: [
          {
            text: '[[ ## question ## ]]\nWeather in Paris?\n\n[[ ## turn_budget ## ]]\nTurn 1 of 3 (2 more after this one).',
            continuation: [],
            type: 'text',
          },
        ],
        continuation: [],
      },
      {
        role: 'assistant',
        parts: [
          {
            text: '[[ ## next_thought ## ]]\nI should look up the weather first.\n\n[[ ## completed ## ]]\n',
            continuation: [],
            type: 'text',
          },
          { id: 'call_1', name: 'get_weather', input: { city: 'Paris' }, continuation: [], type: 'tool_call' },
        ],
        continuation: [],
      },
      {
        role: 'tool',
        parts: [
          {
            id: 'call_1',
            content: [{ text: 'The weather in Paris is sunny.', continuation: [], type: 'text' }],
            name: 'get_weather',
            is_error: false,
            continuation: [],
            type: 'tool_result',
          },
        ],
        continuation: [],
      },
      {
        role: 'user',
        parts: [
          { text: '[[ ## turn_budget ## ]]\nTurn 2 of 3 (1 more after this one).', continuation: [], type: 'text' },
        ],
        continuation: [],
      },
    ],
    system: 'Your input fields are:\n1. `question` (Union[str, NoneType]):',
    tools: [
      {
        name: 'get_weather',
        description: 'Get the weather for a city.',
        parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
        type: 'function',
      },
    ],
    config: { max_tokens: null, temperature: null, stop: [] },
  },
  options: {},
  cache: null,
  rollout_id: null,
};

const MOCK_DSPY_LM15_TOOL_CALLING_OUTPUT = {
  id: 'resp_2',
  model: 'openai/gpt-4o-mini',
  message: {
    role: 'assistant',
    parts: [
      { text: '[[ ## next_thought ## ]]\nI have the answer.\n\n[[ ## completed ## ]]', continuation: [], type: 'text' },
      { id: 'call_2', name: 'submit', input: { answer: 'It is sunny in Paris.' }, continuation: [], type: 'tool_call' },
    ],
    continuation: [],
  },
  finish_reason: 'tool_call',
};

const makeDspyLm15Input = (parts: unknown[]) => ({
  prompt: { model: 'openai/gpt-4o-mini', messages: [{ role: 'user', parts, continuation: [] }], system: null },
  options: {},
  cache: null,
  rollout_id: null,
});

describe('normalizeConversation with DSPy 3.5 lm15 payloads', () => {
  it('should handle a request with a system prompt', () => {
    const conv = normalizeConversation(MOCK_DSPY_LM15_INPUT, 'dspy');
    expect(conv).toEqual([
      expect.objectContaining({ role: 'system', content: expect.stringContaining('Your input fields are:') }),
      expect.objectContaining({ role: 'user', content: expect.stringContaining('What is the capital of France?') }),
    ]);
    expect(conv?.[1].content).toContain('  \n');
  });

  it('should handle a response with reasoning', () => {
    const conv = normalizeConversation(MOCK_DSPY_LM15_OUTPUT, 'dspy');
    expect(conv).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: expect.stringContaining('#### Plan'),
        reasoning: 'The user wants a capital city.',
      }),
    ]);
    expect(conv?.[0].content).toContain('#### Answer');
    expect(conv?.[0].content).not.toContain('[[ ## completed ## ]]');
  });

  it('should handle tool calls and tool results in the request history', () => {
    const conv = normalizeConversation(MOCK_DSPY_LM15_TOOL_CALLING_INPUT, 'dspy');
    expect(conv).toEqual([
      expect.objectContaining({ role: 'system' }),
      expect.objectContaining({ role: 'user', content: expect.stringContaining('#### Question') }),
      expect.objectContaining({
        role: 'assistant',
        content: expect.stringContaining('I should look up the weather first.'),
        tool_calls: [{ id: 'call_1', function: { name: 'get_weather', arguments: '{\n  "city": "Paris"\n}' } }],
      }),
      expect.objectContaining({
        role: 'tool',
        tool_call_id: 'call_1',
        name: 'get_weather',
        content: 'The weather in Paris is sunny.',
      }),
      expect.objectContaining({ role: 'user', content: expect.stringContaining('#### Turn Budget') }),
    ]);
  });

  it('should handle tool calls in the response', () => {
    const conv = normalizeConversation(MOCK_DSPY_LM15_TOOL_CALLING_OUTPUT, 'dspy');
    expect(conv).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: expect.stringContaining('I have the answer.'),
        tool_calls: [
          { id: 'call_2', function: { name: 'submit', arguments: '{\n  "answer": "It is sunny in Paris."\n}' } },
        ],
      }),
    ]);
  });

  it('should render image parts', () => {
    const conv = normalizeConversation(
      makeDspyLm15Input([
        { text: '<image>\n', continuation: [], type: 'text' },
        { media_type: 'image/png', url: 'https://example.com/cat.png', continuation: [], type: 'image' },
        { media_type: 'image/png', data: 'iVBORw0KGgo=', continuation: [], type: 'image' },
      ]),
      'dspy',
    );
    expect(conv).toHaveLength(1);
    expect(conv?.[0].content).toContain('![](https://example.com/cat.png)');
    expect(conv?.[0].content).toContain('![](data:image/png;base64,iVBORw0KGgo=)');
  });

  it('should render a JSON response as a formatted code block', () => {
    const conv = normalizeConversation(
      { message: { role: 'assistant', parts: [{ text: '{"answer": "Paris"}', type: 'text' }] }, finish_reason: 'stop' },
      'dspy',
    );
    expect(conv).toEqual([
      expect.objectContaining({ role: 'assistant', content: '```json\n{\n  "answer": "Paris"\n}\n```' }),
    ]);
  });

  it('should handle a string prompt', () => {
    const conv = normalizeConversation(
      { prompt: 'Say hello', options: { temperature: 0.2 }, cache: null, rollout_id: null },
      'dspy',
    );
    expect(conv).toEqual([expect.objectContaining({ role: 'user', content: 'Say hello' })]);
  });

  it('should not treat messages without parts as an lm15 request', () => {
    expect(normalizeConversation({ prompt: { messages: [{ role: 'user', content: 'hi' }] } }, 'dspy')).toBeNull();
  });

  it('should handle a system prompt given as parts', () => {
    const conv = normalizeConversation(
      {
        prompt: {
          messages: [{ role: 'user', parts: [{ text: 'Hi', type: 'text' }] }],
          system: [{ text: 'Be brief.', type: 'text' }],
        },
      },
      'dspy',
    );
    expect(conv).toEqual([
      expect.objectContaining({ role: 'system', content: 'Be brief.' }),
      expect.objectContaining({ role: 'user', content: 'Hi' }),
    ]);
  });

  it.each([
    [
      'mlflow-attachment:// data',
      { media_type: 'image/png', data: 'mlflow-attachment://abc' },
      '![](mlflow-attachment://abc)',
    ],
    ['no url or data', { media_type: 'image/png', file_id: 'file_1' }, '[image]'],
  ])('should render an image part with %s', (_name, image, expected) => {
    const conv = normalizeConversation(makeDspyLm15Input([{ ...image, type: 'image' }]), 'dspy');
    expect(conv).toEqual([expect.objectContaining({ role: 'user', content: expected })]);
  });

  it.each(['audio', 'video', 'document', 'binary', 'citation', 'data'])(
    'should render a placeholder for %s parts',
    (type) => {
      const conv = normalizeConversation(makeDspyLm15Input([{ type }]), 'dspy');
      expect(conv).toEqual([expect.objectContaining({ role: 'user', content: `[${type}]` })]);
    },
  );

  it.each([
    ['user', 'text'],
    ['developer', 'text'],
    ['assistant', 'refusal'],
  ])('should render the %s role with a %s part', (role, type) => {
    const conv = normalizeConversation({ message: { role, parts: [{ text: 'Hello', type }] } }, 'dspy');
    expect(conv).toEqual([expect.objectContaining({ role, content: 'Hello' })]);
  });

  it('should handle a response with only reasoning', () => {
    const conv = normalizeConversation(
      { message: { role: 'assistant', parts: [{ text: 'Thinking it through.', type: 'thinking' }] } },
      'dspy',
    );
    expect(conv).toEqual([
      expect.objectContaining({ role: 'assistant', content: undefined, reasoning: 'Thinking it through.' }),
    ]);
  });

  it('should handle a response with only a tool call', () => {
    const conv = normalizeConversation(
      { message: { role: 'assistant', parts: [{ id: 'call_1', name: 'search', input: {}, type: 'tool_call' }] } },
      'dspy',
    );
    expect(conv).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: undefined,
        tool_calls: [{ id: 'call_1', function: { name: 'search', arguments: '{}' } }],
      }),
    ]);
  });

  // The chat view parses JSON tool responses itself, and shows other tool responses as plain text
  it.each(['{"temperature": 20}', 'Sunny\nWarm', '[[ ## result ## ]]\nSunny'])(
    'should keep the tool result text %j unformatted',
    (text) => {
      const conv = normalizeConversation(
        {
          message: {
            role: 'tool',
            parts: [{ id: 'call_1', name: 'get_weather', content: [{ text, type: 'text' }], type: 'tool_result' }],
          },
        },
        'dspy',
      );
      expect(conv).toEqual([
        expect.objectContaining({ role: 'tool', tool_call_id: 'call_1', name: 'get_weather', content: text }),
      ]);
    },
  );
});

describe('formatDspySections', () => {
  it('should convert standalone markers to markdown headings', () => {
    const input = '[[ ## reasoning ## ]]\nSome reasoning text';
    expect(formatDspySections(input)).toBe('#### Reasoning\nSome reasoning text');
  });

  it('should handle markers with leading/trailing whitespace on the line', () => {
    const input = '  [[ ## passage ## ]]  \nSome text';
    expect(formatDspySections(input)).toBe('#### Passage\nSome text');
  });

  it('should remove the completed marker entirely', () => {
    const input = 'Some text\n\n[[ ## completed ## ]]';
    expect(formatDspySections(input)).toBe('Some text\n\n');
  });

  it('should leave inline references unchanged', () => {
    const input = 'Start with `[[ ## reasoning ## ]]` then `[[ ## summary ## ]]`.';
    expect(formatDspySections(input)).toBe(input);
  });

  it('should title-case snake_case variable names', () => {
    const input = '[[ ## tool_name_0 ## ]]';
    expect(formatDspySections(input)).toBe('#### Tool Name 0');
  });

  it('should handle multiple sections', () => {
    const input = '[[ ## reasoning ## ]]\nThinking...\n\n[[ ## summary ## ]]\nDone.\n\n[[ ## completed ## ]]';
    const result = formatDspySections(input);
    expect(result).toContain('#### Reasoning');
    expect(result).toContain('#### Summary');
    expect(result).not.toContain('[[ ## completed ## ]]');
    expect(result).not.toContain('#### Completed');
  });

  it('should return text unchanged when there are no markers', () => {
    const input = 'Just regular text with no markers.';
    expect(formatDspySections(input)).toBe(input);
  });

  it('should return empty string unchanged', () => {
    expect(formatDspySections('')).toBe('');
  });
});
