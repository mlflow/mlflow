import type { OpenClawPluginApi } from 'openclaw/plugin-sdk/plugin-entry';
import { startSpan } from '@mlflow/core';
import { createMLflowService } from '../src/service';

type MockSpan = {
  setAttribute: jest.Mock;
  setOutputs: jest.Mock;
  setStatus: jest.Mock;
  end: jest.Mock;
};

jest.mock('@mlflow/core', () => ({
  init: jest.fn(),
  flushTraces: jest.fn(),
  SpanStatusCode: { ERROR: 'ERROR' },
  startSpan: jest.fn(() => ({
    setAttribute: jest.fn(),
    setOutputs: jest.fn(),
    setStatus: jest.fn(),
    end: jest.fn(),
  })),
}));

type Handler = (event: unknown, ctx: unknown) => void;

const PARENT = 'agent:main:main';
const CHILD = 'agent:main:subagent:1f0e';

function registerHooks(): Map<string, Handler> {
  const handlers = new Map<string, Handler>();
  const api = {
    on: (name: string, handler: Handler) => handlers.set(name, handler),
  } as unknown as OpenClawPluginApi;
  createMLflowService(api, {
    trackingUri: 'http://localhost:5000',
    experimentId: '0',
  }).registerHooks();
  return handlers;
}

function subagentSpans(): { options: Record<string, unknown>; span: MockSpan }[] {
  const mock = startSpan as unknown as jest.Mock;
  return mock.mock.calls
    .map(([options], i) => ({
      options: options as Record<string, unknown>,
      span: mock.mock.results[i].value as MockSpan,
    }))
    .filter(({ options }) => String(options.name).startsWith('subagent_'));
}

// Payloads below mirror what OpenClaw passes to `subagent_spawned` and
// `subagent_ended`: the parent session is on the context, the child on the event.
describe('subagent hooks', () => {
  let handlers: Map<string, Handler>;

  beforeEach(() => {
    jest.clearAllMocks();
    handlers = registerHooks();
    handlers.get('llm_input')!({ prompt: 'hi' }, { sessionKey: PARENT });
  });

  function spawn(childSessionKey = CHILD, requesterSessionKey = PARENT): void {
    handlers.get('subagent_spawned')!(
      {
        runId: 'run-1',
        childSessionKey,
        agentId: 'main',
        label: 'research',
        mode: 'run',
        threadRequested: false,
      },
      { runId: 'run-1', childSessionKey, requesterSessionKey },
    );
  }

  function end(event: Record<string, unknown>, childSessionKey = CHILD): void {
    handlers.get('subagent_ended')!(
      { targetSessionKey: childSessionKey, targetKind: 'subagent', ...event },
      { runId: 'run-1', childSessionKey, requesterSessionKey: PARENT },
    );
  }

  it('does not subscribe to the removed subagent_spawning hook', () => {
    expect(handlers.has('subagent_spawning')).toBe(false);
  });

  it('opens a span under the requester trace on spawn', () => {
    spawn();

    const [sub] = subagentSpans();
    expect(sub.options.name).toBe('subagent_research');
    expect(sub.options.inputs).toEqual({
      agent_id: 'main',
      child_session_key: CHILD,
      label: 'research',
    });
    expect(sub.span.end).not.toHaveBeenCalled();
  });

  it('ignores a spawn whose requester has no active trace', () => {
    spawn(CHILD, 'agent:other:main');
    expect(subagentSpans()).toHaveLength(0);
  });

  it('closes the span with the outcome on end', () => {
    spawn();
    end({ reason: 'subagent-complete', outcome: 'ok' });

    const [sub] = subagentSpans();
    expect(sub.span.setOutputs).toHaveBeenCalledWith({
      outcome: 'ok',
      reason: 'subagent-complete',
    });
    expect(sub.span.setStatus).not.toHaveBeenCalled();
    expect(sub.span.end).toHaveBeenCalledTimes(1);
  });

  it('marks the span as failed when the subagent errors', () => {
    spawn();
    end({ reason: 'spawn-failed', outcome: 'error', error: 'Session failed to start' });

    const [sub] = subagentSpans();
    expect(sub.span.setOutputs).toHaveBeenCalledWith({
      outcome: 'error',
      reason: 'spawn-failed',
      error: 'Session failed to start',
    });
    expect(sub.span.setStatus).toHaveBeenCalledWith('ERROR', 'Session failed to start');
    expect(sub.span.end).toHaveBeenCalledTimes(1);
  });

  it('tracks concurrent subagents of the same agent separately', () => {
    const other = 'agent:main:subagent:2a9c';
    spawn(CHILD);
    spawn(other);
    end({ reason: 'subagent-complete', outcome: 'ok' }, other);

    const [first, second] = subagentSpans();
    expect(first.span.end).not.toHaveBeenCalled();
    expect(second.span.end).toHaveBeenCalledTimes(1);
  });

  it('ignores an end event without a requester session', () => {
    spawn();
    handlers.get('subagent_ended')!(
      { targetSessionKey: CHILD, targetKind: 'subagent', reason: 'session-reset' },
      { childSessionKey: CHILD },
    );
    expect(subagentSpans()[0].span.end).not.toHaveBeenCalled();
  });
});
