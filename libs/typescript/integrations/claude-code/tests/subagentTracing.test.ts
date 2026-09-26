import { resolve } from 'node:path';
import { copyFileSync, mkdirSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';

import type { SubagentStopHookInput } from '../src/types';

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
import { processSubagentTranscript } from '../src/subagentTracing';
import { processTranscript } from '../src/tracing';
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
    expect(root.inputs).toEqual({
      description: 'Research auth',
      prompt: 'Find the auth module',
      subagent_type: 'Explore',
      run_in_background: true,
    });
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

    expect(startSpan).not.toHaveBeenCalled();
    expect(consoleError).not.toHaveBeenCalled();
  });

  it('traces nothing for an agent the parent never launched', async () => {
    const { mainPath, agentPath } = backgroundSession();

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
  });
});
