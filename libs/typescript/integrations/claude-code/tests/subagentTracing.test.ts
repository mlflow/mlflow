import { resolve } from 'node:path';
import { appendFileSync, copyFileSync, mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';

import type { SubagentStopHookInput, TranscriptEntry } from '../src/types';

// Keep unit tests offline and deterministic (see tracing.test.ts).
const ORIGINAL_CATALOG_URI = process.env.MLFLOW_MODEL_CATALOG_URI;

beforeAll(() => {
  process.env.MLFLOW_MODEL_CATALOG_URI = '';
});

afterAll(() => {
  if (ORIGINAL_CATALOG_URI === undefined) {
    delete process.env.MLFLOW_MODEL_CATALOG_URI;
  } else {
    process.env.MLFLOW_MODEL_CATALOG_URI = ORIGINAL_CATALOG_URI;
  }
});

jest.mock('@mlflow/core', () =>
  jest
    .requireActual<typeof import('./helpers/mlflowCoreMock')>('./helpers/mlflowCoreMock')
    .createMlflowCoreMock(),
);

// Import after mock
import {
  findAgentLaunch,
  isBackgroundSubagent,
  processSubagentTranscript,
} from '../src/subagentTracing';
import { processTranscript } from '../src/tracing';
import { isBackgroundLaunch } from '../src/toolResults';
import { readTranscript } from '../src/transcript';
import { startSpan, flushTraces } from '@mlflow/core';
import {
  getChildSpans,
  getSpans,
  getSpansByName,
  getSpansByType,
  mockTraceInfo,
  resetMlflowCoreMock,
} from './helpers/mlflowCoreMock';

const FIXTURES_DIR = resolve(__dirname, 'fixtures');
const BACKGROUND_AGENT_ID = 'bg5678';

/**
 * Lay the fixtures out the way Claude Code does on disk:
 * `<dir>/<session>.jsonl` and `<dir>/<session>/subagents/agent-<id>.jsonl`.
 */
function layOutSession(
  mainFixture: string,
  agentFixture: string,
  agentId: string,
): { mainPath: string; agentPath: string } {
  const dir = mkdtempSync(resolve(tmpdir(), 'cc-subagent-'));
  const mainPath = resolve(dir, 'session.jsonl');
  const subagentsDir = resolve(dir, 'session', 'subagents');
  const agentPath = resolve(subagentsDir, `agent-${agentId}.jsonl`);
  mkdirSync(subagentsDir, { recursive: true });
  copyFileSync(resolve(FIXTURES_DIR, mainFixture), mainPath);
  copyFileSync(resolve(FIXTURES_DIR, agentFixture), agentPath);
  return { mainPath, agentPath };
}

function backgroundSession() {
  return layOutSession(
    'with-background-subagent.jsonl',
    'subagent-bg5678.jsonl',
    BACKGROUND_AGENT_ID,
  );
}

function appendEntries(path: string, entries: TranscriptEntry[]): void {
  appendFileSync(path, entries.map((e) => JSON.stringify(e)).join('\n') + '\n');
}

/** A later turn in which the parent resumes the agent with SendMessage. */
const SEND_MESSAGE_TURN: TranscriptEntry[] = [
  {
    type: 'assistant',
    message: {
      id: 'msg_main_3',
      role: 'assistant',
      content: [
        {
          type: 'tool_use',
          id: 'toolu_send_001',
          name: 'SendMessage',
          input: { to: BACKGROUND_AGENT_ID, message: 'Also check the tests' },
        },
      ],
    },
    timestamp: '2025-02-01T09:00:04.000Z',
  },
  {
    type: 'user',
    message: {
      role: 'user',
      content: [{ type: 'tool_result', tool_use_id: 'toolu_send_001', content: 'Message sent.' }],
    },
    toolUseResult: { agentId: BACKGROUND_AGENT_ID, status: 'completed' },
    timestamp: '2025-02-01T09:00:05.000Z',
  },
];

function subagentStopInput(overrides: Partial<SubagentStopHookInput>): SubagentStopHookInput {
  return {
    session_id: 'bg-session',
    transcript_path: '',
    agent_id: BACKGROUND_AGENT_ID,
    agent_type: 'Explore',
    agent_transcript_path: '',
    ...overrides,
  };
}

let consoleError: jest.SpyInstance;

beforeEach(() => {
  resetMlflowCoreMock();
  jest.clearAllMocks();
  consoleError = jest.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  consoleError.mockRestore();
});

describe('processSubagentTranscript (SubagentStop hook)', () => {
  it('traces a finished background agent as its own trace', async () => {
    const { mainPath, agentPath } = backgroundSession();

    await processSubagentTranscript(
      subagentStopInput({ transcript_path: mainPath, agent_transcript_path: agentPath }),
    );

    const roots = getSpans().filter((s) => s.parentId == null);
    expect(roots).toHaveLength(1);
    const root = roots[0];
    expect(root.name).toBe('subagent_Explore');
    expect(root.spanType).toBe('AGENT');
    expect(root.inputs).toEqual({ prompt: 'Find the auth module' });
    expect(root.outputs.response).toBe('The auth module is auth.py.');

    const llms = getSpansByType('LLM');
    expect(llms).toHaveLength(2);
    expect(getSpansByName('tool_Grep')).toHaveLength(1);
    expect(getChildSpans(root.spanId)).toHaveLength(3);

    const usage = llms.map((s) => s.attributes['mlflow.chat.tokenUsage'] as Record<string, number>);
    const sum = (key: string) => usage.reduce((acc, u) => acc + u[key], 0);
    expect(sum('input_tokens')).toBe(250);
    expect(sum('output_tokens')).toBe(50);
    expect(sum('total_tokens')).toBe(300);
    expect(mockTraceInfo.traceMetadata['mlflow.trace.cost']).toBeDefined();

    expect(mockTraceInfo.traceMetadata['mlflow.trace.session']).toBe('bg-session');
    expect(mockTraceInfo.tags).toEqual({
      'mlflow.claude_code.agent_id': BACKGROUND_AGENT_ID,
      'mlflow.claude_code.agent_type': 'Explore',
      'mlflow.claude_code.parent_tool_use_id': 'toolu_bg_001',
    });
    expect(flushTraces).toHaveBeenCalledTimes(1);
    expect(consoleError).not.toHaveBeenCalled();
  });

  it('traces only the latest window of a resumed agent', async () => {
    const { mainPath, agentPath } = layOutSession(
      'with-background-subagent.jsonl',
      'subagent-bg5678-resumed.jsonl',
      BACKGROUND_AGENT_ID,
    );

    await processSubagentTranscript(
      subagentStopInput({ transcript_path: mainPath, agent_transcript_path: agentPath }),
    );

    const roots = getSpans().filter((s) => s.parentId == null);
    expect(roots).toHaveLength(1);
    expect(roots[0].inputs).toEqual({ prompt: 'Also list the auth tests' });
    expect(roots[0].outputs.response).toBe('The auth tests are in tests/test_auth.py.');
    expect(getSpansByName('tool_Glob')).toHaveLength(1);
    expect(getSpansByName('tool_Grep')).toHaveLength(0);

    const llms = getSpansByType('LLM');
    expect(llms).toHaveLength(2);
    const usage = llms.map((s) => s.attributes['mlflow.chat.tokenUsage'] as Record<string, number>);
    expect(usage.reduce((acc, u) => acc + u.input_tokens, 0)).toBe(820);
    expect(usage.reduce((acc, u) => acc + u.output_tokens, 0)).toBe(20);
    expect(mockTraceInfo.tags['mlflow.claude_code.agent_id']).toBe(BACKGROUND_AGENT_ID);
  });

  it('names the root span "subagent" when the agent type is unknown', async () => {
    const { mainPath, agentPath } = backgroundSession();

    await processSubagentTranscript(
      subagentStopInput({
        transcript_path: mainPath,
        agent_transcript_path: agentPath,
        agent_type: undefined,
      }),
    );

    expect(getSpans().filter((s) => s.parentId == null)[0].name).toBe('subagent');
    expect(mockTraceInfo.tags['mlflow.claude_code.agent_type']).toBeUndefined();
  });

  it('traces nothing for a sync agent', async () => {
    const { mainPath, agentPath } = layOutSession(
      'with-subagent-file.jsonl',
      'subagent-abc1234.jsonl',
      'abc1234',
    );

    await processSubagentTranscript(
      subagentStopInput({
        transcript_path: mainPath,
        agent_id: 'abc1234',
        agent_transcript_path: agentPath,
      }),
    );

    const launch = findAgentLaunch(readTranscript(mainPath), 'abc1234');
    expect(launch?.toolUseId).toBe('toolu_task_file_001');
    expect(isBackgroundLaunch(launch!.result)).toBe(false);
    expect(startSpan).not.toHaveBeenCalled();
    expect(consoleError).not.toHaveBeenCalled();
  });

  it('traces nothing for an agent the parent never launched', async () => {
    const { mainPath, agentPath } = backgroundSession();
    expect(findAgentLaunch(readTranscript(mainPath), 'internal-agent')).toBeUndefined();

    await processSubagentTranscript(
      subagentStopInput({
        transcript_path: mainPath,
        agent_id: 'internal-agent',
        agent_transcript_path: agentPath,
      }),
    );

    expect(startSpan).not.toHaveBeenCalled();
    expect(consoleError).not.toHaveBeenCalled();
  });

  it('reports a missing agent transcript on stderr without throwing', async () => {
    const { mainPath, agentPath } = backgroundSession();

    await expect(
      processSubagentTranscript(
        subagentStopInput({
          transcript_path: mainPath,
          agent_transcript_path: `${agentPath}.missing`,
        }),
      ),
    ).resolves.toBeUndefined();

    expect(startSpan).not.toHaveBeenCalled();
    expect(consoleError).toHaveBeenCalledTimes(1);
    expect(consoleError.mock.calls[0][0]).toMatch(/^\[mlflow\] Sub-agent transcript not found/);
  });

  it('never treats a SendMessage result as a launch', () => {
    const dir = mkdtempSync(resolve(tmpdir(), 'cc-subagent-'));
    const mainPath = resolve(dir, 'session.jsonl');
    writeFileSync(mainPath, '');
    const [sendMessageUse, sendMessageResult] = SEND_MESSAGE_TURN;
    appendEntries(mainPath, [
      sendMessageUse,
      {
        ...sendMessageResult,
        toolUseResult: { agentId: BACKGROUND_AGENT_ID, status: 'async_launched' },
      },
    ]);

    expect(findAgentLaunch(readTranscript(mainPath), BACKGROUND_AGENT_ID)).toBeUndefined();
  });

  it('keeps the background launch classification after a SendMessage resume', async () => {
    const { mainPath, agentPath } = backgroundSession();
    appendEntries(mainPath, SEND_MESSAGE_TURN);

    const launch = findAgentLaunch(readTranscript(mainPath), BACKGROUND_AGENT_ID);
    expect(launch?.toolUseId).toBe('toolu_bg_001');

    await processSubagentTranscript(
      subagentStopInput({ transcript_path: mainPath, agent_transcript_path: agentPath }),
    );

    expect(getSpans().filter((s) => s.parentId == null)[0].name).toBe('subagent_Explore');
    expect(mockTraceInfo.tags['mlflow.claude_code.parent_tool_use_id']).toBe('toolu_bg_001');
  });
});

describe('isBackgroundSubagent', () => {
  it('is true for an agent the parent launched in the background', () => {
    const { mainPath } = backgroundSession();
    expect(isBackgroundSubagent({ transcript_path: mainPath, agent_id: BACKGROUND_AGENT_ID })).toBe(
      true,
    );
  });

  it('is false for a sync agent', () => {
    const { mainPath } = layOutSession(
      'with-subagent-file.jsonl',
      'subagent-abc1234.jsonl',
      'abc1234',
    );
    expect(isBackgroundSubagent({ transcript_path: mainPath, agent_id: 'abc1234' })).toBe(false);
  });

  it('is false for an agent the parent never launched', () => {
    const { mainPath } = backgroundSession();
    expect(isBackgroundSubagent({ transcript_path: mainPath, agent_id: 'internal-agent' })).toBe(
      false,
    );
  });

  it('is false, without stderr, when the main transcript is missing', () => {
    const missing = resolve(tmpdir(), 'cc-subagent-missing', 'session.jsonl');
    expect(isBackgroundSubagent({ transcript_path: missing, agent_id: BACKGROUND_AGENT_ID })).toBe(
      false,
    );
    expect(consoleError).not.toHaveBeenCalled();
  });
});

describe('processTranscript (Stop hook) with a background agent', () => {
  it('marks the Agent tool span as background and does not nest the agent', async () => {
    // The agent file exists on disk, so only the background rule keeps it out.
    const { mainPath } = backgroundSession();

    await processTranscript(mainPath, 'bg-session');

    const agentTools = getSpansByName('tool_Agent');
    expect(agentTools).toHaveLength(1);
    expect(agentTools[0].attributes.background).toBe(true);
    expect(agentTools[0].attributes.agent_id).toBe(BACKGROUND_AGENT_ID);
    expect(agentTools[0].outputs).toEqual({ result: 'Async agent launched successfully.' });
    expect(getChildSpans(agentTools[0].spanId)).toHaveLength(0);
    expect(getSpansByType('AGENT')).toHaveLength(1);
    expect(getSpansByName('tool_Grep')).toHaveLength(0);
    // Agent tags belong to the sub-agent's own trace, never to the parent.
    expect(mockTraceInfo.tags).toEqual({});
  });

  it('nests nothing under a SendMessage result that carries the agentId', async () => {
    // The agent file exists on disk, so only the launch-tool rule keeps it out.
    const { mainPath } = backgroundSession();
    appendEntries(mainPath, SEND_MESSAGE_TURN);

    await processTranscript(mainPath, 'bg-session');

    const sendMessages = getSpansByName('tool_SendMessage');
    expect(sendMessages).toHaveLength(1);
    expect(getChildSpans(sendMessages[0].spanId)).toHaveLength(0);
    expect(sendMessages[0].attributes.background).toBeUndefined();
    expect(getSpansByName('tool_Agent')[0].attributes.background).toBe(true);
    expect(getSpansByType('AGENT')).toHaveLength(1);
  });

  it('reports a background receipt without agentId and still marks it background', async () => {
    const dir = mkdtempSync(resolve(tmpdir(), 'cc-subagent-'));
    const mainPath = resolve(dir, 'session.jsonl');
    writeFileSync(mainPath, '');
    appendEntries(mainPath, [
      {
        type: 'user',
        message: { role: 'user', content: 'Run it in the background' },
        timestamp: '2025-02-01T09:00:00.000Z',
      },
      {
        type: 'assistant',
        message: {
          role: 'assistant',
          content: [
            {
              type: 'tool_use',
              id: 'toolu_bg_noid',
              name: 'Agent',
              input: { prompt: 'Work', run_in_background: true },
            },
          ],
        },
        timestamp: '2025-02-01T09:00:01.000Z',
      },
      {
        type: 'user',
        message: {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'toolu_bg_noid',
              content: 'Async agent launched successfully.',
            },
          ],
        },
        toolUseResult: { status: 'async_launched' },
        timestamp: '2025-02-01T09:00:02.000Z',
      },
    ]);

    await processTranscript(mainPath, 'bg-session');

    const agentTool = getSpansByName('tool_Agent')[0];
    expect(agentTool.attributes.background).toBe(true);
    expect(agentTool.attributes.agent_id).toBeUndefined();
    expect(consoleError).toHaveBeenCalledTimes(1);
    expect(consoleError.mock.calls[0][0]).toMatch(
      /^\[mlflow\] Background agent receipt .* no agentId/,
    );
  });
});
