import { describe, expect, it } from '@jest/globals';

import { normalizeToolDefinitions } from './tools';

const WEATHER_SCHEMA = {
  type: 'object',
  properties: { city: { type: 'string', description: 'The city name' } },
  required: ['city'],
};

describe('normalizeToolDefinitions', () => {
  // One row per framework request shape. The extraction is shape-based, so each row documents
  // the real wire format an integration records in span inputs.
  it.each([
    [
      'OpenAI-compatible (OpenAI / Mistral / DeepSeek / Groq / LiteLLM / MLflow gateway)',
      {
        model: 'gpt-4o',
        messages: [{ role: 'user', content: 'weather?' }],
        tools: [
          {
            type: 'function',
            function: { name: 'get_weather', description: 'Get weather', parameters: WEATHER_SCHEMA },
          },
        ],
      },
      [{ type: 'function', function: { name: 'get_weather', description: 'Get weather', parameters: WEATHER_SCHEMA } }],
    ],
    [
      'Anthropic (input_schema tools)',
      {
        model: 'claude-sonnet-4-6',
        messages: [{ role: 'user', content: 'weather?' }],
        tools: [{ name: 'get_weather', description: 'Get weather', input_schema: WEATHER_SCHEMA }],
      },
      [{ type: 'function', function: { name: 'get_weather', description: 'Get weather', parameters: WEATHER_SCHEMA } }],
    ],
    [
      'Google Gemini (function_declarations)',
      {
        contents: [{ role: 'user', parts: [{ text: 'weather?' }] }],
        tools: [
          { function_declarations: [{ name: 'get_weather', description: 'Get weather', parameters: WEATHER_SCHEMA }] },
        ],
      },
      [{ type: 'function', function: { name: 'get_weather', description: 'Get weather', parameters: WEATHER_SCHEMA } }],
    ],
    [
      'Amazon Bedrock Converse (toolSpec with nested inputSchema.json)',
      {
        messages: [{ role: 'user', content: [{ text: 'weather?' }] }],
        toolConfig: {
          tools: [
            { toolSpec: { name: 'get_weather', description: 'Get weather', inputSchema: { json: WEATHER_SCHEMA } } },
          ],
        },
      },
      [{ type: 'function', function: { name: 'get_weather', description: 'Get weather', parameters: WEATHER_SCHEMA } }],
    ],
    [
      'PydanticAI (model_request_parameters.function_tools, as recorded by autolog)',
      {
        messages: [{ parts: [{ content: 'weather?', part_kind: 'user-prompt' }] }],
        model_request_parameters: {
          function_tools: [
            {
              name: 'get_weather',
              description: 'Get weather',
              parameters_json_schema: { ...WEATHER_SCHEMA, additionalProperties: false },
              outer_typed_dict_key: null,
              strict: null,
              kind: 'function',
            },
          ],
          output_tools: [],
        },
      },
      [
        {
          type: 'function',
          function: {
            name: 'get_weather',
            description: 'Get weather',
            parameters: { ...WEATHER_SCHEMA, additionalProperties: false },
          },
        },
      ],
    ],
    [
      'unknown wrapper objects (e.g. LangChain invocation params carrying OpenAI-shaped tools)',
      {
        messages: [[{ role: 'user', content: 'weather?' }]],
        invocation_params: {
          _type: 'openai-chat',
          tools: [{ type: 'function', function: { name: 'get_weather', parameters: WEATHER_SCHEMA } }],
        },
      },
      [{ type: 'function', function: { name: 'get_weather', parameters: WEATHER_SCHEMA } }],
    ],
  ])('extracts tool definitions from %s inputs', (_format, inputs, expected) => {
    expect(normalizeToolDefinitions(inputs)).toEqual(expected);
  });

  it.each([
    [
      'OpenAI assistant tool_calls (calls, not definitions)',
      {
        messages: [
          { role: 'user', content: 'weather?' },
          {
            role: 'assistant',
            content: null,
            tool_calls: [
              { id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } },
            ],
          },
          { role: 'tool', tool_call_id: 'call_1', content: '20C' },
        ],
      },
    ],
    [
      'Anthropic tool_use / tool_result content blocks',
      {
        messages: [
          {
            role: 'assistant',
            content: [{ type: 'tool_use', id: 'toolu_1', name: 'get_weather', input: { city: 'Paris' } }],
          },
          { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: '20C' }] },
        ],
      },
    ],
    [
      'Gemini functionCall parts',
      {
        contents: [{ role: 'model', parts: [{ functionCall: { name: 'get_weather', args: { city: 'Paris' } } }] }],
      },
    ],
    [
      'PydanticAI tool-call / tool-return message parts',
      {
        messages: [
          {
            parts: [
              { tool_name: 'get_weather', args: '{"city":"Paris"}', tool_call_id: 't1', part_kind: 'tool-call' },
              { tool_name: 'get_weather', content: '20C', part_kind: 'tool-return' },
            ],
          },
        ],
      },
    ],
    [
      'response_format json_schema envelopes',
      {
        messages: [{ role: 'user', content: 'hi' }],
        response_format: {
          type: 'json_schema',
          json_schema: { name: 'response_schema', schema: WEATHER_SCHEMA, strict: true },
        },
      },
    ],
    [
      'plain named objects without a schema (messages, configs)',
      {
        messages: [{ role: 'user', content: 'hi', name: 'joel' }],
        metadata: { name: 'my-run', parameters: 'not-a-schema' },
      },
    ],
    ['null inputs', null],
    ['primitive inputs', 'What is the weather?'],
  ])('does not misdetect %s as tool definitions', (_case, inputs) => {
    expect(normalizeToolDefinitions(inputs)).toBeUndefined();
  });

  it('de-duplicates repeated definitions by tool name', () => {
    const inputs = {
      tools: [
        { type: 'function', function: { name: 'get_weather', parameters: WEATHER_SCHEMA } },
        { type: 'function', function: { name: 'get_weather', parameters: WEATHER_SCHEMA } },
        { name: 'get_lat_lng', input_schema: WEATHER_SCHEMA },
      ],
    };
    expect(normalizeToolDefinitions(inputs)?.map((tool) => tool.function.name)).toEqual(['get_weather', 'get_lat_lng']);
  });

  it('keeps the tool but drops the schema when the schema is malformed for the UI', () => {
    const inputs = {
      tools: [
        {
          type: 'function',
          // `required` must be a string list — a malformed schema should not lose the tool itself.
          function: { name: 'bad_schema', parameters: { type: 'object', properties: {}, required: 'city' } },
        },
      ],
    };
    expect(normalizeToolDefinitions(inputs)).toEqual([{ type: 'function', function: { name: 'bad_schema' } }]);
  });

  it('accepts OpenAI-style definitions without parameters', () => {
    expect(normalizeToolDefinitions({ tools: [{ type: 'function', function: { name: 'noop' } }] })).toEqual([
      { type: 'function', function: { name: 'noop' } },
    ]);
  });
});
