// Track mock spans
let spanCounter = 0;
const mockSpans: Record<string, any> = {};
const mockTraceInfo: {
  traceMetadata: Record<string, string>;
} = {
  traceMetadata: {},
};

jest.mock('@mlflow/core', () => {
  return {
    init: jest.fn(),
    startSpan: jest.fn((options: any) => {
      const id = `span-${++spanCounter}`;
      const parentId = options.parent ? options.parent.spanId : null;
      const span = {
        name: options.name,
        traceId: 'mock-trace-id',
        spanId: id,
        parentId,
        spanType: options.spanType ?? 'UNKNOWN',
        inputs: options.inputs ?? {},
        outputs: {},
        attributes: { ...(options.attributes ?? {}) },
        startTimeNs: options.startTimeNs ?? null,
        endTimeNs: null,
        statusCode: null as string | null,
        statusMessage: null as string | null,
        setAttribute: jest.fn((key: string, value: any) => {
          span.attributes[key] = value;
        }),
        setStatus: jest.fn((code: string, message?: string) => {
          span.statusCode = code;
          span.statusMessage = message ?? null;
        }),
        end: jest.fn((opts?: any) => {
          if (opts?.outputs) {
            span.outputs = opts.outputs;
          }
          if (opts?.endTimeNs != null) {
            span.endTimeNs = opts.endTimeNs;
          }
        }),
      };
      mockSpans[id] = span;
      return span;
    }),
    flushTraces: jest.fn().mockResolvedValue(undefined),
    SpanStatusCode: {
      OK: 'STATUS_CODE_OK',
      ERROR: 'STATUS_CODE_ERROR',
      UNSET: 'STATUS_CODE_UNSET',
    },
    SpanType: {
      LLM: 'LLM',
      AGENT: 'AGENT',
      TOOL: 'TOOL',
      UNKNOWN: 'UNKNOWN',
    },
    SpanAttributeKey: {
      TOKEN_USAGE: 'mlflow.chat.tokenUsage',
      MESSAGE_FORMAT: 'mlflow.message.format',
    },
    TraceMetadataKey: {
      TRACE_SESSION: 'mlflow.trace.session',
      TRACE_USER: 'mlflow.trace.user',
    },
    TokenUsageKey: {
      INPUT_TOKENS: 'input_tokens',
      OUTPUT_TOKENS: 'output_tokens',
      TOTAL_TOKENS: 'total_tokens',
      CACHE_READ_INPUT_TOKENS: 'cache_read_input_tokens',
      CACHE_CREATION_INPUT_TOKENS: 'cache_creation_input_tokens',
    },
    InMemoryTraceManager: {
      getInstance: jest.fn(() => ({
        getTrace: jest.fn(() => ({
          info: mockTraceInfo,
        })),
      })),
    },
  };
});

import { resolve } from 'path';

import {
  buildToolStatuses,
  createChildSpans,
  findTaskCompleteNs,
  findTaskStartedNs,
  processNotify,
  reconstructMessages,
} from '../src/tracing';
import { readTranscript, getLastTurnRecords } from '../src/transcript';
import { flushTraces } from '@mlflow/core';
import type { NotifyPayload, RolloutLine } from '../src/types';

const FIXTURES_DIR = resolve(__dirname, 'fixtures');

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getSpans(): any[] {
  return Object.values(mockSpans);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getSpansByType(type: string): any[] {
  return getSpans().filter((s) => s.spanType === type);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getRootSpan(): any {
  return getSpans().find((s) => s.parentId == null);
}

function makeNotifyPayload(overrides?: Partial<NotifyPayload>): NotifyPayload {
  return {
    type: 'agent-turn-complete',
    'thread-id': 'test-thread-001',
    'turn-id': 'test-turn-001',
    cwd: '/tmp/test',
    client: 'codex-tui',
    'input-messages': ['what is 2+2'],
    'last-assistant-message': '4',
    ...overrides,
  };
}

describe('processNotify', () => {
  beforeEach(() => {
    spanCounter = 0;
    Object.keys(mockSpans).forEach((key) => delete mockSpans[key]);
    mockTraceInfo.traceMetadata = {};
    jest.clearAllMocks();
  });

  it('creates an AGENT root span with LLM child', async () => {
    await processNotify(makeNotifyPayload());

    const root = getRootSpan();
    expect(root).toBeDefined();
    expect(root.name).toBe('codex_conversation');
    expect(root.spanType).toBe('AGENT');

    const llmSpans = getSpansByType('LLM');
    expect(llmSpans.length).toBe(1);
    expect(llmSpans[0].parentId).toBe(root.spanId);
  });

  it('passes the user prompt as a raw string on the root span', async () => {
    await processNotify(makeNotifyPayload());

    const root = getRootSpan();
    expect(root.inputs).toBe('what is 2+2');
  });

  it('sets session metadata', async () => {
    await processNotify(makeNotifyPayload());

    expect(mockTraceInfo.traceMetadata['mlflow.trace.session']).toBe('test-thread-001');
    expect(mockTraceInfo.traceMetadata['mlflow.trace.user']).toBeDefined();
  });

  it('sets the OpenTelemetry service name on the root span', async () => {
    await processNotify(makeNotifyPayload());

    expect(getRootSpan().attributes['service.name']).toBe('codex');
  });

  it('uses last input message only', async () => {
    await processNotify(
      makeNotifyPayload({
        'input-messages': ['first prompt', 'second prompt', 'third prompt'],
        'last-assistant-message': 'response to third',
      }),
    );

    const root = getRootSpan();
    expect(root.inputs).toBe('third prompt');
    const endCall = (root.end as jest.Mock).mock.calls[0][0];
    expect(endCall.outputs).toBe('response to third');
  });

  it('skips when no user prompt', async () => {
    await processNotify(makeNotifyPayload({ 'input-messages': [] }));

    expect(getSpans().length).toBe(0);
    expect(flushTraces).not.toHaveBeenCalled();
  });

  it('calls flushTraces after processing', async () => {
    await processNotify(makeNotifyPayload());
    expect(flushTraces).toHaveBeenCalled();
  });

  it('sets the assistant response as the raw root span output', async () => {
    await processNotify(makeNotifyPayload());

    const root = getRootSpan();
    expect(root.end).toHaveBeenCalled();
    const endCall = (root.end as jest.Mock).mock.calls[0][0];
    expect(endCall.outputs).toBe('4');
  });

  it('uses OpenAI chat format for the fallback LLM span', async () => {
    await processNotify(makeNotifyPayload());

    const [llm] = getSpansByType('LLM');
    expect(llm.name).toBe('llm_call');
    expect(llm.inputs).toEqual({
      model: 'unknown',
      messages: [{ role: 'user', content: 'what is 2+2' }],
    });

    const endCall = (llm.end as jest.Mock).mock.calls[0][0];
    expect(endCall.outputs).toEqual({
      choices: [{ message: { role: 'assistant', content: '4' } }],
    });
  });
});

describe('reconstructMessages', () => {
  function responseItem(payload: Record<string, unknown>): RolloutLine {
    return {
      timestamp: '2026-04-05T10:00:00Z',
      type: 'response_item',
      payload: payload as RolloutLine['payload'],
    };
  }

  it('maps user and assistant messages to chat format', () => {
    const items = [
      responseItem({
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'hi' }],
      }),
      responseItem({
        type: 'message',
        role: 'assistant',
        content: [{ type: 'output_text', text: 'hello' }],
      }),
    ];
    expect(reconstructMessages(items, items.length)).toEqual([
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
    ]);
  });

  it('maps developer messages to system role', () => {
    const items = [
      responseItem({
        type: 'message',
        role: 'developer',
        content: [{ type: 'input_text', text: 'you are a helpful assistant' }],
      }),
    ];
    expect(reconstructMessages(items, items.length)).toEqual([
      { role: 'system', content: 'you are a helpful assistant' },
    ]);
  });

  it('maps function_call to assistant message with tool_calls', () => {
    const items = [
      responseItem({
        type: 'function_call',
        name: 'exec_command',
        call_id: 'call_1',
        arguments: '{"cmd":"ls"}',
      }),
    ];
    expect(reconstructMessages(items, items.length)).toEqual([
      {
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            id: 'call_1',
            type: 'function',
            function: { name: 'exec_command', arguments: '{"cmd":"ls"}' },
          },
        ],
      },
    ]);
  });

  it('maps function_call_output to tool message', () => {
    const items = [
      responseItem({
        type: 'function_call_output',
        call_id: 'call_1',
        output: 'file1.txt\nfile2.txt',
      }),
    ];
    expect(reconstructMessages(items, items.length)).toEqual([
      { role: 'tool', tool_call_id: 'call_1', content: 'file1.txt\nfile2.txt' },
    ]);
  });

  it('maps current Codex custom tool records to chat format', () => {
    const items = [
      responseItem({
        type: 'custom_tool_call',
        name: 'exec',
        call_id: 'call_custom_1',
        input: 'await tools.exec_command({"cmd":"pwd"});',
      }),
      responseItem({
        type: 'custom_tool_call_output',
        call_id: 'call_custom_1',
        output: [
          { type: 'input_text', text: 'Script completed\nOutput:\n' },
          { type: 'input_text', text: '/tmp/test\n' },
        ],
      }),
    ];

    expect(reconstructMessages(items, items.length)).toEqual([
      {
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            id: 'call_custom_1',
            type: 'function',
            function: {
              name: 'exec',
              arguments: 'await tools.exec_command({"cmd":"pwd"});',
            },
          },
        ],
      },
      {
        role: 'tool',
        tool_call_id: 'call_custom_1',
        content: 'Script completed\nOutput:\n/tmp/test\n',
      },
    ]);
  });

  it('stops at uptoIndex and preserves order across a tool-use turn', () => {
    const items = [
      responseItem({
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'list files' }],
      }),
      responseItem({
        type: 'function_call',
        name: 'exec',
        call_id: 'c1',
        arguments: '{"cmd":"ls"}',
      }),
      responseItem({
        type: 'function_call_output',
        call_id: 'c1',
        output: 'a.txt',
      }),
      responseItem({
        type: 'message',
        role: 'assistant',
        content: [{ type: 'output_text', text: 'there is one file' }],
      }),
    ];
    // uptoIndex=3 stops before the final assistant message
    expect(reconstructMessages(items, 3)).toEqual([
      { role: 'user', content: 'list files' },
      {
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            id: 'c1',
            type: 'function',
            function: { name: 'exec', arguments: '{"cmd":"ls"}' },
          },
        ],
      },
      { role: 'tool', tool_call_id: 'c1', content: 'a.txt' },
    ]);
  });

  it('skips empty-text messages', () => {
    const items = [
      responseItem({
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: '   ' }],
      }),
    ];
    expect(reconstructMessages(items, items.length)).toEqual([]);
  });
});

describe('createChildSpans (integration with real transcript fixture)', () => {
  // Reset the mock span tracking between tests to avoid pollution from
  // processNotify tests in the same file.
  beforeEach(() => {
    spanCounter = 0;
    Object.keys(mockSpans).forEach((key) => delete mockSpans[key]);
    jest.clearAllMocks();
  });

  it('creates the expected span tree from the tool-call fixture', () => {
    // Fixture conversation:
    //   user "list files in current directory"
    //   assistant "I'll list the files for you."
    //   function_call exec_command({"cmd":"ls"})  -> call_abc123
    //   function_call_output "file1.txt\nfile2.txt\nfile3.txt"
    //   assistant "There are 3 files: ..."
    const records = readTranscript(resolve(FIXTURES_DIR, 'with-tool-call.jsonl'));
    const turn = getLastTurnRecords(records);

    // Build a fake parent span that startSpan() can parent children to
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const parent = { spanId: 'root' } as any;

    createChildSpans(parent, turn, 'gpt-4');

    const llmSpans = getSpansByType('LLM');
    const toolSpans = getSpansByType('TOOL');

    // 2 model responses -> 2 LLM spans; 1 function_call -> 1 TOOL span
    expect(llmSpans.length).toBe(2);
    expect(toolSpans.length).toBe(1);

    // All children parented to the fake root
    for (const span of [...llmSpans, ...toolSpans]) {
      expect(span.parentId).toBe('root');
    }

    // First LLM response contains both commentary and a tool call.
    expect(llmSpans[0].name).toBe('llm_call');
    expect(llmSpans[0].inputs).toEqual({
      model: 'gpt-4',
      messages: [{ role: 'user', content: 'list files in current directory' }],
    });
    const firstLlmEnd = (llmSpans[0].end as jest.Mock).mock.calls[0][0];
    expect(firstLlmEnd.outputs).toEqual({
      choices: [
        {
          message: {
            role: 'assistant',
            content: "I'll list the files for you.",
            tool_calls: [
              {
                id: 'call_abc123',
                type: 'function',
                function: { name: 'exec_command', arguments: '{"cmd":"ls"}' },
              },
            ],
          },
        },
      ],
    });

    // TOOL span
    expect(toolSpans[0].name).toBe('tool_exec_command');
    expect(toolSpans[0].inputs).toEqual({ cmd: 'ls' });
    expect(toolSpans[0].attributes).toEqual({
      tool_name: 'exec_command',
      tool_id: 'call_abc123',
    });
    const toolEnd = (toolSpans[0].end as jest.Mock).mock.calls[0][0];
    expect(toolEnd.outputs).toEqual({
      result: 'file1.txt\nfile2.txt\nfile3.txt',
    });

    // Second LLM span: full conversation history including the tool-call and
    // tool-result should be reconstructed in inputs.messages
    expect(llmSpans[1].inputs).toEqual({
      model: 'gpt-4',
      messages: [
        { role: 'user', content: 'list files in current directory' },
        { role: 'assistant', content: "I'll list the files for you." },
        {
          role: 'assistant',
          content: null,
          tool_calls: [
            {
              id: 'call_abc123',
              type: 'function',
              function: { name: 'exec_command', arguments: '{"cmd":"ls"}' },
            },
          ],
        },
        {
          role: 'tool',
          tool_call_id: 'call_abc123',
          content: 'file1.txt\nfile2.txt\nfile3.txt',
        },
      ],
    });
    const secondLlmEnd = (llmSpans[1].end as jest.Mock).mock.calls[0][0];
    expect(secondLlmEnd.outputs).toEqual({
      choices: [
        {
          message: {
            role: 'assistant',
            content: 'There are 3 files: file1.txt, file2.txt, file3.txt',
          },
        },
      ],
    });
  });

  it('derives span timestamps from turn boundaries, not next-record chaining', () => {
    // Fixture timestamps are spaced 1 second apart:
    //   10:00:00Z task_started
    //   10:00:00Z user "list files in current directory"
    //   10:00:01Z assistant "I'll list the files for you."
    //   10:00:02Z function_call exec_command
    //   10:00:03Z function_call_output
    //   10:00:04Z assistant "There are 3 files: ..."
    //   10:00:05Z task_complete
    //
    // Expected:
    //   LLM #1: task_started (00) -> function_call (02)          = 2s
    //   TOOL:   function_call (02) -> function_call_output (03)  = 1s
    //   LLM #2: function_call_output (03) -> assistant #2 (04)   = 1s
    const records = readTranscript(resolve(FIXTURES_DIR, 'with-tool-call.jsonl'));
    const turn = getLastTurnRecords(records);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const parent = { spanId: 'root' } as any;
    createChildSpans(parent, turn, 'gpt-4');

    const NS_PER_SEC = 1_000_000_000;
    const parseNs = (iso: string) => new Date(iso).getTime() * 1_000_000;

    const llmSpans = getSpansByType('LLM');
    const toolSpans = getSpansByType('TOOL');

    // LLM #1: task_started -> final output record in the response
    expect(llmSpans[0].startTimeNs).toBe(parseNs('2026-04-05T10:00:00Z'));
    expect(llmSpans[0].endTimeNs).toBe(parseNs('2026-04-05T10:00:02Z'));
    expect(llmSpans[0].endTimeNs - llmSpans[0].startTimeNs).toBe(2 * NS_PER_SEC);

    // TOOL: function_call -> matching function_call_output (by call_id)
    expect(toolSpans[0].startTimeNs).toBe(parseNs('2026-04-05T10:00:02Z'));
    expect(toolSpans[0].endTimeNs).toBe(parseNs('2026-04-05T10:00:03Z'));
    expect(toolSpans[0].endTimeNs - toolSpans[0].startTimeNs).toBe(NS_PER_SEC);

    // LLM #2: previous function_call_output -> second assistant message
    // (NOT from the previous assistant message — the LLM was waiting on the
    // tool during that time)
    expect(llmSpans[1].startTimeNs).toBe(parseNs('2026-04-05T10:00:03Z'));
    expect(llmSpans[1].endTimeNs).toBe(parseNs('2026-04-05T10:00:04Z'));
    expect(llmSpans[1].endTimeNs - llmSpans[1].startTimeNs).toBe(NS_PER_SEC);

    // Sanity: spans don't overlap
    expect(llmSpans[0].endTimeNs).toBeLessThanOrEqual(toolSpans[0].startTimeNs);
    expect(toolSpans[0].endTimeNs).toBeLessThanOrEqual(llmSpans[1].startTimeNs);
  });

  it('creates custom TOOL spans and records usage on each LLM span', () => {
    const records = readTranscript(resolve(FIXTURES_DIR, 'with-custom-tool-call.jsonl'));
    const turn = getLastTurnRecords(records);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const parent = { spanId: 'root' } as any;
    createChildSpans(parent, turn, 'gpt-5.6-sol');

    const [tool] = getSpansByType('TOOL');
    expect(tool.name).toBe('tool_exec');
    expect(tool.inputs).toEqual({
      input: 'await tools.exec_command({"cmd":"sed -n \'1,5p\' package.json"});',
    });
    expect((tool.end as jest.Mock).mock.calls[0][0].outputs).toEqual({
      result: 'Script completed\nOutput:\n  "version": "0.4.0"\n',
    });

    const [firstLlm, secondLlm] = getSpansByType('LLM');
    expect(firstLlm.attributes['mlflow.chat.tokenUsage']).toEqual({
      input_tokens: 25025,
      output_tokens: 102,
      total_tokens: 25127,
      cache_read_input_tokens: 11904,
      cache_creation_input_tokens: 0,
    });
    expect(secondLlm.attributes['mlflow.chat.tokenUsage']).toEqual({
      input_tokens: 25180,
      output_tokens: 22,
      total_tokens: 25202,
      cache_read_input_tokens: 24832,
      cache_creation_input_tokens: 0,
    });
  });

  it('creates token-counted LLM spans for tool-only model responses', () => {
    const records = readTranscript(resolve(FIXTURES_DIR, 'with-tool-only-calls.jsonl'));
    const turn = getLastTurnRecords(records);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const parent = { spanId: 'root' } as any;
    createChildSpans(parent, turn, 'gpt-5.6-sol');

    const llmSpans = getSpansByType('LLM');
    const toolSpans = getSpansByType('TOOL');
    expect(llmSpans).toHaveLength(3);
    expect(toolSpans).toHaveLength(2);

    expect((llmSpans[0].end as jest.Mock).mock.calls[0][0].outputs).toEqual({
      choices: [
        {
          message: {
            role: 'assistant',
            content: null,
            tool_calls: [
              {
                id: 'call_1',
                type: 'function',
                function: {
                  name: 'exec',
                  arguments: 'await tools.exec_command({"cmd":"pwd"});',
                },
              },
            ],
          },
        },
      ],
    });
    expect((llmSpans[1].end as jest.Mock).mock.calls[0][0].outputs).toEqual({
      choices: [
        {
          message: {
            role: 'assistant',
            content: null,
            tool_calls: [
              {
                id: 'call_2',
                type: 'function',
                function: {
                  name: 'exec',
                  arguments: 'await tools.exec_command({"cmd":"git status --short"});',
                },
              },
            ],
          },
        },
      ],
    });
    expect((llmSpans[2].end as jest.Mock).mock.calls[0][0].outputs).toEqual({
      choices: [{ message: { role: 'assistant', content: 'The repository is clean.' } }],
    });

    expect(llmSpans.map((span) => span.attributes['mlflow.chat.tokenUsage'] as unknown)).toEqual([
      {
        input_tokens: 100,
        output_tokens: 10,
        total_tokens: 110,
        cache_read_input_tokens: 80,
      },
      {
        input_tokens: 120,
        output_tokens: 15,
        total_tokens: 135,
        cache_read_input_tokens: 96,
      },
      {
        input_tokens: 140,
        output_tokens: 20,
        total_tokens: 160,
        cache_read_input_tokens: 112,
      },
    ]);
  });
});

describe('findTaskStartedNs / findTaskCompleteNs', () => {
  it('extract the turn boundary timestamps from the fixture', () => {
    const records = readTranscript(resolve(FIXTURES_DIR, 'with-tool-call.jsonl'));
    const turn = getLastTurnRecords(records);

    const parseNs = (iso: string) => new Date(iso).getTime() * 1_000_000;
    expect(findTaskStartedNs(turn)).toBe(parseNs('2026-04-05T10:00:00Z'));
    expect(findTaskCompleteNs(turn)).toBe(parseNs('2026-04-05T10:00:05Z'));
  });

  it('returns null when the turn lacks a task_started or task_complete event', () => {
    // A turn that only contains a response_item (no event_msg markers)
    const records: RolloutLine[] = [
      {
        timestamp: '2026-04-05T10:00:00Z',
        type: 'response_item',
        payload: { type: 'message', role: 'user', content: [] },
      },
    ];
    expect(findTaskStartedNs(records)).toBeNull();
    expect(findTaskCompleteNs(records)).toBeNull();
  });

  it('returned values bracket all child spans from createChildSpans', () => {
    // Invariant: if processNotify passes these timestamps as the root span's
    // startTimeNs/endTimeNs, the root span will correctly enclose every
    // child created by createChildSpans.
    const records = readTranscript(resolve(FIXTURES_DIR, 'with-tool-call.jsonl'));
    const turn = getLastTurnRecords(records);
    const startNs = findTaskStartedNs(turn)!;
    const completeNs = findTaskCompleteNs(turn)!;

    // Reset the span tracker populated by earlier describe blocks
    spanCounter = 0;
    Object.keys(mockSpans).forEach((key) => delete mockSpans[key]);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const parent = { spanId: 'root' } as any;
    createChildSpans(parent, turn, 'gpt-4');

    for (const span of Object.values(mockSpans)) {
      expect(span.startTimeNs).toBeGreaterThanOrEqual(startNs);
      const endNs = (span.end as jest.Mock).mock.calls[0]?.[0]?.endTimeNs;
      expect(endNs).toBeLessThanOrEqual(completeNs);
    }
  });
});

describe('tool span failure status', () => {
  beforeEach(() => {
    spanCounter = 0;
    Object.keys(mockSpans).forEach((key) => delete mockSpans[key]);
    jest.clearAllMocks();
  });

  it('buildToolStatuses flags failed exec_command_end by status and exit_code', () => {
    const records = readTranscript(resolve(FIXTURES_DIR, 'with-failed-tool.jsonl'));
    const turn = getLastTurnRecords(records);

    const statuses = buildToolStatuses(turn);
    expect(statuses).toEqual({
      call_fail_1: { failed: true, exitCode: 127 },
      call_ok_1: { failed: false, exitCode: 0 },
    });
  });

  it('sets ERROR status on TOOL spans whose exec_command_end reports failure', () => {
    const records = readTranscript(resolve(FIXTURES_DIR, 'with-failed-tool.jsonl'));
    const turn = getLastTurnRecords(records);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const parent = { spanId: 'root' } as any;
    createChildSpans(parent, turn, 'gpt-4');

    const toolSpans = getSpansByType('TOOL');
    expect(toolSpans.length).toBe(2);

    const failed = toolSpans.find((s) => s.attributes.tool_id === 'call_fail_1');
    const ok = toolSpans.find((s) => s.attributes.tool_id === 'call_ok_1');

    expect(failed.statusCode).toBe('STATUS_CODE_ERROR');
    expect(failed.statusMessage).toContain('127');
    expect(failed.setStatus).toHaveBeenCalled();

    // OK tool: setStatus should NOT have been called
    expect(ok.statusCode).toBeNull();
    expect(ok.setStatus).not.toHaveBeenCalled();
  });
});
