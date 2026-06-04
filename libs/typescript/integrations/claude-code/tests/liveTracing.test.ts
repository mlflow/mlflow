/**
 * Unit tests for LiveTracingContext — exercise the streaming-path skill-scope
 * state machine directly. Mocks @mlflow/core so spans are captured in-memory
 * rather than shipped to a tracking server.
 *
 * Counterpart to tracedClaudeAgent.test.ts (which spins up a real server).
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

interface MockSpan {
  name: string;
  spanType: string;
  inputs: any;
  outputs: any;
  attributes: Record<string, any>;
  parentId: string | null;
  ended: boolean;
}

const mockSpans: MockSpan[] = [];
let spanCounter = 0;

function resetMocks() {
  mockSpans.length = 0;
  spanCounter = 0;
}

jest.mock('@mlflow/core', () => ({
  startSpan: jest.fn((opts: any) => {
    spanCounter += 1;
    const span: MockSpan = {
      name: opts.name,
      spanType: opts.spanType ?? 'UNKNOWN',
      inputs: opts.inputs ?? {},
      outputs: {},
      attributes: { ...(opts.attributes ?? {}) },
      parentId: opts.parent ? (opts.parent as { spanId: string }).spanId : null,
      ended: false,
    };
    const handle: any = {
      spanId: `span-${spanCounter}`,
      setAttribute: jest.fn((k: string, v: any) => {
        span.attributes[k] = v;
      }),
      setOutputs: jest.fn((o: any) => {
        span.outputs = o;
      }),
      setStatus: jest.fn(),
      setInputs: jest.fn(),
      end: jest.fn(() => {
        span.ended = true;
      }),
    };
    mockSpans.push(span);
    return handle;
  }),
  flushTraces: jest.fn().mockResolvedValue(undefined),
  SpanType: { LLM: 'LLM', TOOL: 'TOOL', AGENT: 'AGENT', CHAIN: 'CHAIN', UNKNOWN: 'UNKNOWN' },
  SpanAttributeKey: {
    TOKEN_USAGE: 'mlflow.chat.tokenUsage',
    MESSAGE_FORMAT: 'mlflow.message.format',
    SKILL_NAME: 'mlflow.skill.name',
  },
  SpanStatusCode: { OK: 'OK', ERROR: 'ERROR', UNSET: 'UNSET' },
  TraceMetadataKey: {
    TRACE_SESSION: 'mlflow.trace.session',
    TRACE_USER: 'mlflow.trace.user',
  },
  InMemoryTraceManager: {
    getInstance: jest.fn(() => ({
      getTrace: jest.fn(() => ({ info: { traceMetadata: {} } })),
    })),
  },
}));

import { LiveTracingContext } from '../src/liveTracing';

const SKILL_NAME = 'mlflow.skill.name';
const skillSpans = () => mockSpans.filter((s) => s.attributes[SKILL_NAME]);
const findByToolId = (id: string) =>
  mockSpans.find((s) => s.spanType === 'TOOL' && s.attributes.tool_id === id);
const llmSpans = () => mockSpans.filter((s) => s.spanType === 'LLM');

// Helper: assistant message that calls a single tool by id+name.
const toolUseMsg = (id: string, name: string, input: any = {}) =>
  ({
    type: 'assistant' as const,
    parent_tool_use_id: null,
    message: {
      role: 'assistant' as const,
      model: 'claude-test',
      content: [{ type: 'tool_use' as const, id, name, input }],
    },
  }) as any;

// Helper: assistant message containing only text — produces an LLM span.
const textMsg = (text: string) =>
  ({
    type: 'assistant' as const,
    parent_tool_use_id: null,
    message: {
      role: 'assistant' as const,
      model: 'claude-test',
      content: [{ type: 'text' as const, text }],
    },
  }) as any;

// Helper: user tool_result message; opts can attach tool_use_result for skills.
const toolResultMsg = (
  toolUseId: string,
  resultContent: string,
  opts: { commandName?: string } = {},
) => ({
  type: 'user' as const,
  parent_tool_use_id: null,
  message: {
    role: 'user' as const,
    content: [
      { type: 'tool_result' as const, tool_use_id: toolUseId, content: resultContent },
    ],
  },
  ...(opts.commandName
    ? { tool_use_result: { success: true, commandName: opts.commandName } }
    : {}),
});

// Helper: a plain text user prompt (multi-prompt AsyncIterable scenario).
const userPromptMsg = (text: string, extra: Record<string, unknown> = {}) => ({
  type: 'user' as const,
  parent_tool_use_id: null,
  message: { role: 'user' as const, content: text },
  ...extra,
});

describe('LiveTracingContext — skill scope propagation', () => {
  beforeEach(() => resetMocks());

  it('stamps SKILL_NAME on the Skill TOOL span itself when commandName arrives', () => {
    const ctx = new LiveTracingContext('do the thing');
    ctx.onAssistantMessage(toolUseMsg('skill_1', 'Skill', { skill: 'my-skill' }));
    ctx.onUserMessage(toolResultMsg('skill_1', 'launched', { commandName: 'my-skill' }));

    expect(findByToolId('skill_1')?.attributes[SKILL_NAME]).toBe('my-skill');
  });

  it('propagates SKILL_NAME to child LLM spans emitted after the skill', () => {
    const ctx = new LiveTracingContext('do the thing');
    ctx.onAssistantMessage(toolUseMsg('skill_1', 'Skill'));
    ctx.onUserMessage(toolResultMsg('skill_1', 'launched', { commandName: 'my-skill' }));
    ctx.onAssistantMessage(textMsg('Working inside the skill.'));

    const taggedLlm = llmSpans().filter((s) => s.attributes[SKILL_NAME] === 'my-skill');
    expect(taggedLlm).toHaveLength(1);
  });

  it('propagates SKILL_NAME to child TOOL spans emitted after the skill', () => {
    const ctx = new LiveTracingContext('do the thing');
    ctx.onAssistantMessage(toolUseMsg('skill_1', 'Skill'));
    ctx.onUserMessage(toolResultMsg('skill_1', 'launched', { commandName: 'my-skill' }));
    ctx.onAssistantMessage(toolUseMsg('bash_1', 'Bash'));

    expect(findByToolId('bash_1')?.attributes[SKILL_NAME]).toBe('my-skill');
  });

  it('does NOT tag spans created before any skill was invoked', () => {
    const ctx = new LiveTracingContext('do the thing');
    ctx.onAssistantMessage(toolUseMsg('bash_pre', 'Bash'));
    ctx.onUserMessage(toolResultMsg('bash_pre', 'ok'));

    expect(findByToolId('bash_pre')?.attributes[SKILL_NAME]).toBeUndefined();
  });
});

describe('LiveTracingContext — isRealUserPrompt clears skill scope', () => {
  beforeEach(() => resetMocks());

  function buildContextWithActiveSkill() {
    const ctx = new LiveTracingContext('start');
    ctx.onAssistantMessage(toolUseMsg('skill_1', 'Skill'));
    ctx.onUserMessage(toolResultMsg('skill_1', 'launched', { commandName: 'my-skill' }));
    return ctx;
  }

  it('clears scope when origin.kind === "human" arrives mid-stream (AsyncIterable case)', () => {
    const ctx = buildContextWithActiveSkill();
    ctx.onUserMessage(userPromptMsg('a new prompt', { origin: { kind: 'human' } }));
    ctx.onAssistantMessage(textMsg('after the new prompt'));

    const afterPromptLlm = llmSpans().slice(-1)[0];
    expect(afterPromptLlm.attributes[SKILL_NAME]).toBeUndefined();
  });

  it('clears scope on a plain user prompt when origin is absent (older SDK fallback)', () => {
    // After the skill finishes, there's intervening assistant work (a Bash
    // call + its tool_result) before the user types a new prompt. The Bash
    // tool_result has no commandName so prevUserHadCommandName resets to
    // false, letting the next plain user prompt be recognized as real.
    const ctx = buildContextWithActiveSkill();
    ctx.onAssistantMessage(toolUseMsg('bash_inside', 'Bash'));
    ctx.onUserMessage(toolResultMsg('bash_inside', 'ok'));
    ctx.onUserMessage(userPromptMsg('another prompt'));
    ctx.onAssistantMessage(textMsg('plain'));

    expect(llmSpans().slice(-1)[0].attributes[SKILL_NAME]).toBeUndefined();
  });

  it('does NOT clear on tool_result echoes (tool_use_result set)', () => {
    const ctx = buildContextWithActiveSkill();
    ctx.onAssistantMessage(toolUseMsg('bash_1', 'Bash'));
    ctx.onUserMessage(toolResultMsg('bash_1', 'ok'));
    ctx.onAssistantMessage(textMsg('still in skill'));

    expect(llmSpans().slice(-1)[0].attributes[SKILL_NAME]).toBe('my-skill');
  });

  it('does NOT clear on skill content injection (msg right after Skill tool_result)', () => {
    // Skill content injections look like a plain text user msg but always
    // follow a tool_result with commandName. prevUserHadCommandName catches it.
    const ctx = buildContextWithActiveSkill();
    ctx.onUserMessage(userPromptMsg('Base directory for this skill: /path...'));
    ctx.onAssistantMessage(textMsg('reading skill body'));

    expect(llmSpans().slice(-1)[0].attributes[SKILL_NAME]).toBe('my-skill');
  });

  it('does NOT clear on sub-agent (parent_tool_use_id set)', () => {
    const ctx = buildContextWithActiveSkill();
    ctx.onUserMessage({
      ...userPromptMsg('subagent inner'),
      parent_tool_use_id: 'some_parent',
    });
    ctx.onAssistantMessage(textMsg('inside subagent'));

    expect(llmSpans().slice(-1)[0].attributes[SKILL_NAME]).toBe('my-skill');
  });

  it('does NOT clear on non-human origin (e.g. coordinator)', () => {
    const ctx = buildContextWithActiveSkill();
    ctx.onUserMessage(userPromptMsg('coordinator msg', { origin: { kind: 'coordinator' } }));
    ctx.onAssistantMessage(textMsg('still in skill'));

    expect(llmSpans().slice(-1)[0].attributes[SKILL_NAME]).toBe('my-skill');
  });

  it('does NOT clear on isSynthetic messages', () => {
    const ctx = buildContextWithActiveSkill();
    ctx.onUserMessage(userPromptMsg('synthetic', { isSynthetic: true }));
    ctx.onAssistantMessage(textMsg('still in skill'));

    expect(llmSpans().slice(-1)[0].attributes[SKILL_NAME]).toBe('my-skill');
  });

  it('does NOT clear on shouldQuery=false', () => {
    const ctx = buildContextWithActiveSkill();
    ctx.onUserMessage(userPromptMsg('appended', { shouldQuery: false }));
    ctx.onAssistantMessage(textMsg('still in skill'));

    expect(llmSpans().slice(-1)[0].attributes[SKILL_NAME]).toBe('my-skill');
  });

  it('most-recent-skill wins when a second skill is invoked', () => {
    const ctx = buildContextWithActiveSkill();
    ctx.onAssistantMessage(toolUseMsg('skill_2', 'Skill'));
    ctx.onUserMessage(toolResultMsg('skill_2', 'launched', { commandName: 'other-skill' }));
    ctx.onAssistantMessage(textMsg('inside other skill'));

    expect(llmSpans().slice(-1)[0].attributes[SKILL_NAME]).toBe('other-skill');
  });
});

afterAll(() => expect(skillSpans().length).toBeGreaterThan(0)); // smoke: tests actually exercised the attribute
