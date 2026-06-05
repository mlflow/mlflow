/**
 * Unit tests for LiveTracingContext — exercise the streaming-path skill-scope
 * state machine directly. Mocks @mlflow/core so spans are captured in-memory
 * rather than shipped to a tracking server.
 *
 * Counterpart to tracedClaudeAgent.test.ts (which spins up a real server).
 *
 * ----------------------------------------------------------------------------
 * Canonical Skill wire format (shared with the Python and TS-transcript tests)
 * ----------------------------------------------------------------------------
 *
 * When the user invokes a Skill, the Claude Code CLI emits TWO distinct
 * user-role stream events on the happy path:
 *
 *   [assistant] tool_use(id=X, name="Skill", input={skill: "<name>"})
 *   [user]      tool_result(tool_use_id=X)            ← success result
 *                 + tool_use_result.commandName = "<name>"
 *                 + tool_use_result.success    = true
 *   [user]      text "<skill body>"                   ← synthetic body injection
 *   [assistant] ... continuation (work inside the skill body) ...
 *
 * At the API layer the SDK merges the two user entries into one message with
 * two content blocks; at the stream layer they remain two consecutive user
 * events. The `prevUserHadCommandName` look-back exists to handle exactly
 * that two-event pattern without clobbering the active skill scope.
 *
 * On the FAILED path (is_error=true), the CLI does NOT emit a body injection
 * (the skill never boots), so failed fixtures stop at the tool_result and
 * the assistant's next event is recovery work.
 *
 * Most tests below deliberately use the minimal sequence needed to drive one
 * state-machine transition — adding the body-injection user event is a no-op
 * for many of them and was elided for clarity. The integration-style
 * propagation tests at the top of the file DO include the body injection to
 * stay wire-format-accurate; treat those as the canonical reference shape.
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
  opts: { commandName?: string; isError?: boolean } = {},
) => ({
  type: 'user' as const,
  parent_tool_use_id: null,
  message: {
    role: 'user' as const,
    content: [
      {
        type: 'tool_result' as const,
        tool_use_id: toolUseId,
        content: resultContent,
        ...(opts.isError ? { is_error: true } : {}),
      },
    ],
  },
  ...(opts.commandName
    ? { tool_use_result: { success: !opts.isError, commandName: opts.commandName } }
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
    // Synthetic body injection — canonical happy-path wire shape.
    ctx.onUserMessage(userPromptMsg('<my-skill body>'));
    ctx.onAssistantMessage(textMsg('Working inside the skill.'));

    const taggedLlm = llmSpans().filter((s) => s.attributes[SKILL_NAME] === 'my-skill');
    expect(taggedLlm).toHaveLength(1);
  });

  it('propagates SKILL_NAME to child TOOL spans emitted after the skill', () => {
    const ctx = new LiveTracingContext('do the thing');
    ctx.onAssistantMessage(toolUseMsg('skill_1', 'Skill'));
    ctx.onUserMessage(toolResultMsg('skill_1', 'launched', { commandName: 'my-skill' }));
    // Synthetic body injection — canonical happy-path wire shape.
    ctx.onUserMessage(userPromptMsg('<my-skill body>'));
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

describe('LiveTracingContext — failed Skill should not propagate', () => {
  beforeEach(() => resetMocks());

  it('does NOT propagate skill name to subsequent spans when the Skill tool_result is_error=true', () => {
    const ctx = new LiveTracingContext('start');
    ctx.onAssistantMessage(toolUseMsg('skill_failed', 'Skill'));
    ctx.onUserMessage(
      toolResultMsg('skill_failed', 'launch failed', {
        commandName: 'broken-skill',
        isError: true,
      }),
    );
    ctx.onAssistantMessage(textMsg('recovering after failure'));

    // The Skill TOOL span itself is still stamped with its commandName for
    // identification ("which skill failed?").
    expect(findByToolId('skill_failed')?.attributes[SKILL_NAME]).toBe('broken-skill');
    // But subsequent spans must NOT inherit the failed skill's name.
    expect(llmSpans().slice(-1)[0].attributes[SKILL_NAME]).toBeUndefined();
  });

  it('successful skill after a failed one still propagates correctly', () => {
    const ctx = new LiveTracingContext('start');
    ctx.onAssistantMessage(toolUseMsg('skill_failed', 'Skill'));
    ctx.onUserMessage(
      toolResultMsg('skill_failed', 'failed', {
        commandName: 'broken-skill',
        isError: true,
      }),
    );
    ctx.onAssistantMessage(toolUseMsg('skill_ok', 'Skill'));
    ctx.onUserMessage(toolResultMsg('skill_ok', 'launched', { commandName: 'good-skill' }));
    ctx.onAssistantMessage(textMsg('inside good skill'));

    expect(llmSpans().slice(-1)[0].attributes[SKILL_NAME]).toBe('good-skill');
  });

  it('a failed Skill clears the prior skill scope (no leak to recovery spans)', () => {
    // Skill A succeeds, then Skill B fails. Recovery work after B must NOT
    // inherit Skill A's name.
    const ctx = new LiveTracingContext('start');
    ctx.onAssistantMessage(toolUseMsg('skill_a', 'Skill'));
    ctx.onUserMessage(toolResultMsg('skill_a', 'launched', { commandName: 'skill-a' }));
    ctx.onAssistantMessage(toolUseMsg('skill_b', 'Skill'));
    ctx.onUserMessage(
      toolResultMsg('skill_b', 'failed', { commandName: 'skill-b', isError: true }),
    );
    ctx.onAssistantMessage(textMsg('recovering'));

    expect(findByToolId('skill_a')?.attributes[SKILL_NAME]).toBe('skill-a');
    expect(findByToolId('skill_b')?.attributes[SKILL_NAME]).toBe('skill-b');
    // Recovery LLM span must NOT carry skill-a (the prior skill).
    expect(llmSpans().slice(-1)[0].attributes[SKILL_NAME]).toBeUndefined();
  });
});

describe('LiveTracingContext — onSystemInit resets all skill-scope state', () => {
  beforeEach(() => resetMocks());

  it('clears prevUserHadCommandName in addition to activeSkillName', () => {
    const ctx = new LiveTracingContext('start');
    ctx.onAssistantMessage(toolUseMsg('skill_1', 'Skill'));
    ctx.onUserMessage(toolResultMsg('skill_1', 'launched', { commandName: 'my-skill' }));
    // Now prevUserHadCommandName=true, activeSkillName='my-skill'.
    ctx.onSystemInit({ type: 'system', subtype: 'init' } as any);
    // A plain user prompt right after init must NOT be misclassified as a
    // skill body injection — both state fields must be reset.
    ctx.onUserMessage({
      type: 'user',
      message: { role: 'user', content: 'fresh prompt' },
    } as any);
    ctx.onAssistantMessage(textMsg('responding fresh'));

    expect(llmSpans().slice(-1)[0].attributes[SKILL_NAME]).toBeUndefined();
  });
});

describe('LiveTracingContext — isRealUserPrompt empty content handling', () => {
  beforeEach(() => resetMocks());

  it('does NOT treat empty content array as a real prompt (preserves skill scope)', () => {
    const ctx = new LiveTracingContext('start');
    ctx.onAssistantMessage(toolUseMsg('skill_1', 'Skill'));
    ctx.onUserMessage(toolResultMsg('skill_1', 'launched', { commandName: 'my-skill' }));
    // An empty-array content message should not be classified as a real
    // user prompt — i.e., it should not clear active skill scope.
    ctx.onUserMessage({
      type: 'user',
      message: { role: 'user', content: [] },
    } as any);
    ctx.onAssistantMessage(textMsg('still in skill'));

    expect(llmSpans().slice(-1)[0].attributes[SKILL_NAME]).toBe('my-skill');
  });
});

describe('LiveTracingContext — nested skills (KNOWN LIMITATION)', () => {
  beforeEach(() => resetMocks());

  // KNOWN LIMITATION (documented): The single-slot state machine doesn't
  // preserve the outer skill's scope across nested invocations.
  //
  // Ideal attribution: the user invoked skill-A, so every span produced
  // while A is running — including B's TOOL span, C's TOOL span, and all of
  // B/C's reasoning LLMs — should be tagged skill-A. B and C are
  // implementation details of A. Actual behavior: each nested Skill
  // overwrites the active scope, so B's work is tagged skill-B, C's work
  // skill-C, and the trailing unwind LLMs keep skill-C because there's no
  // stack to pop.
  //
  // Set a breakpoint in `liveTracing.ts` at the `this.activeSkillName = ...`
  // assignment (around line 377) to step through this.
  it('nested skill overwrites the outer skill scope', () => {
    const ctx = new LiveTracingContext('start');

    // Skill A invoked.
    ctx.onAssistantMessage(toolUseMsg('ta', 'Skill', { skill: 'A' }));
    ctx.onUserMessage(toolResultMsg('ta', 'launch A', { commandName: 'skill-A' }));
    // A body injection — must NOT clear scope (caught by prevUserHadCommandName).
    ctx.onUserMessage({
      type: 'user',
      parent_tool_use_id: null,
      message: { role: 'user', content: [{ type: 'text', text: 'A body' }] },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);
    ctx.onAssistantMessage(textMsg('A reasoning'));

    // Skill B invoked from within A.
    ctx.onAssistantMessage(toolUseMsg('tb', 'Skill', { skill: 'B' }));
    ctx.onUserMessage(toolResultMsg('tb', 'launch B', { commandName: 'skill-B' }));
    ctx.onUserMessage({
      type: 'user',
      parent_tool_use_id: null,
      message: { role: 'user', content: [{ type: 'text', text: 'B body' }] },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);
    ctx.onAssistantMessage(textMsg('B reasoning'));

    // Skill C invoked from within B (deepest).
    ctx.onAssistantMessage(toolUseMsg('tc', 'Skill', { skill: 'C' }));
    ctx.onUserMessage(toolResultMsg('tc', 'launch C', { commandName: 'skill-C' }));
    ctx.onUserMessage({
      type: 'user',
      parent_tool_use_id: null,
      message: { role: 'user', content: [{ type: 'text', text: 'C body' }] },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);
    ctx.onAssistantMessage(textMsg('C reasoning (deepest)'));

    // Unwind: C finishes, control returns to B → A → user. The semantic
    // skill scope should pop on each, but the state machine has no stack.
    ctx.onAssistantMessage(textMsg('back to B'));
    ctx.onAssistantMessage(textMsg('back to A'));
    ctx.onAssistantMessage(textMsg('final reply'));

    // Each Skill's OWN TOOL span IS stamped with its own commandName via the
    // own-span pass. Ideally tb/tc would also report skill-A (the outer
    // scope), but the single slot can only hold one value.
    expect(findByToolId('ta')?.attributes[SKILL_NAME]).toBe('skill-A');
    // ↓ THE LIMITATION (own-span): tb/tc tagged with themselves; the outer
    //   skill-A scope is lost. Ideally both would be skill-A.
    expect(findByToolId('tb')?.attributes[SKILL_NAME]).toBe('skill-B');
    expect(findByToolId('tc')?.attributes[SKILL_NAME]).toBe('skill-C');

    const llmByText = (text: string) =>
      llmSpans().find((s) => JSON.stringify(s.outputs?.content ?? '').includes(text));

    // Inside-A reasoning: tagged skill-A (correct).
    expect(llmByText('A reasoning')?.attributes[SKILL_NAME]).toBe('skill-A');

    // ↓ THE LIMITATION (propagation): once a nested skill is invoked, the
    //   slot is overwritten and all descendants get the inner skill's name.
    //   Ideally every LLM below should still be tagged skill-A — A is the
    //   user-invoked skill and B/C are implementation details of A.
    expect(llmByText('B reasoning')?.attributes[SKILL_NAME]).toBe('skill-B');
    expect(llmByText('C reasoning (deepest)')?.attributes[SKILL_NAME]).toBe('skill-C');
    expect(llmByText('back to B')?.attributes[SKILL_NAME]).toBe('skill-C');
    expect(llmByText('back to A')?.attributes[SKILL_NAME]).toBe('skill-C');
    expect(llmByText('final reply')?.attributes[SKILL_NAME]).toBe('skill-C');
  });
});

afterAll(() => expect(skillSpans().length).toBeGreaterThan(0)); // smoke: tests actually exercised the attribute
