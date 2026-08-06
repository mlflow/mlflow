import { describe, expect, it } from '@jest/globals';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';
import type { CustomViewData } from './customViewBuilders';
import {
  isKnownSource,
  isSourceMarker,
  isSpanRefMarker,
  isValidSpanFieldMarker,
  isValidSpanRefSelector,
  resolveScalarSource,
  resolveSpanFieldSource,
  resolveSpanRef,
  unwrapSpanRefSelector,
} from './customViewSources';

const makeNode = (node: Partial<ModelTraceSpanNode> & { key: string; start: number }): ModelTraceSpanNode =>
  node as unknown as ModelTraceSpanNode;

const nodeMap = {
  root: makeNode({ key: 'root', start: 0, title: 'agent', type: ModelSpanType.AGENT }),
  tool0: makeNode({
    key: 'tool0',
    start: 10,
    parentId: 'root',
    title: 'run_sql_query',
    type: ModelSpanType.TOOL,
    inputs: { q: 'select 1' },
    outputs: { rows: 1 },
  }),
  tool1: makeNode({ key: 'tool1', start: 20, parentId: 'root', title: 'fetch', type: ModelSpanType.TOOL }),
} satisfies Record<string, ModelTraceSpanNode>;

const viewData: CustomViewData = {
  metrics: { status: 'OK', latency: '275ms', totalTokens: '1,024', assessments: '3' } as CustomViewData['metrics'],
  assessmentItems: [],
};

describe('marker guards', () => {
  it('recognizes $source and $spanRef markers', () => {
    expect(isSourceMarker({ $source: 'metrics.latency' })).toBe(true);
    expect(isSourceMarker({ source: 'metrics.latency' })).toBe(false);
    expect(isSpanRefMarker({ $spanRef: 'root' })).toBe(true);
    expect(isSpanRefMarker({ spanRef: 'root' })).toBe(false);
  });

  it('classifies known source names across all categories', () => {
    expect(isKnownSource('metrics.latency')).toBe(true);
    expect(isKnownSource('assessments')).toBe(true);
    expect(isKnownSource('spanField')).toBe(true);
    expect(isKnownSource('nonsense')).toBe(false);
    // Removed alongside the TreeView catalog primitive.
    expect(isKnownSource('spanTree')).toBe(false);
    // Removed alongside the DataTable catalog primitive (the only array source).
    expect(isKnownSource('toolRows')).toBe(false);
  });
});

describe('isValidSpanRefSelector', () => {
  it('accepts root, type, name, and type+nth selectors', () => {
    expect(isValidSpanRefSelector('root')).toBe(true);
    expect(isValidSpanRefSelector({ type: 'TOOL' })).toBe(true);
    expect(isValidSpanRefSelector({ name: 'run_sql_query' })).toBe(true);
    expect(isValidSpanRefSelector({ type: 'TOOL', nth: 1 })).toBe(true);
  });

  it('rejects empty selectors, bad nth, and raw indices', () => {
    expect(isValidSpanRefSelector({})).toBe(false);
    expect(isValidSpanRefSelector({ type: 'TOOL', nth: '1' })).toBe(false);
    expect(isValidSpanRefSelector({ type: 'TOOL', nth: -1 })).toBe(false);
    expect(isValidSpanRefSelector({ type: 'TOOL', nth: 1.5 })).toBe(false);
    expect(isValidSpanRefSelector({ type: 'TOOL', nth: Number.NaN })).toBe(false);
    expect(isValidSpanRefSelector(0)).toBe(false);
    expect(isValidSpanRefSelector('first')).toBe(false);
  });
});

describe('isValidSpanFieldMarker', () => {
  it('accepts a valid selector + known field (bare and wrapped spanRef)', () => {
    expect(isValidSpanFieldMarker({ spanRef: { type: 'TOOL', nth: 0 }, field: 'outputs' })).toBe(true);
    expect(isValidSpanFieldMarker({ spanRef: { $spanRef: 'root' }, field: 'name' })).toBe(true);
  });

  it('rejects unknown fields or invalid selectors', () => {
    expect(isValidSpanFieldMarker({ spanRef: 'root', field: 'cost' })).toBe(false);
    expect(isValidSpanFieldMarker({ spanRef: {}, field: 'outputs' })).toBe(false);
  });

  it('accepts a nested path (keys + indices) on structural fields', () => {
    expect(
      isValidSpanFieldMarker({ spanRef: 'root', field: 'outputs', path: ['choices', 0, 'message', 'content'] }),
    ).toBe(true);
    expect(isValidSpanFieldMarker({ spanRef: 'root', field: 'inputs', path: ['query'] })).toBe(true);
    expect(isValidSpanFieldMarker({ spanRef: 'root', field: 'attributes', path: ['usage', 'total_tokens'] })).toBe(
      true,
    );
  });

  it('rejects a path on non-structural fields or a malformed path', () => {
    expect(isValidSpanFieldMarker({ spanRef: 'root', field: 'name', path: ['content'] })).toBe(false);
    expect(isValidSpanFieldMarker({ spanRef: 'root', field: 'spanId', path: ['x'] })).toBe(false);
    expect(isValidSpanFieldMarker({ spanRef: 'root', field: 'outputs', path: [] })).toBe(false);
    expect(isValidSpanFieldMarker({ spanRef: 'root', field: 'outputs', path: 'content' })).toBe(false);
    expect(isValidSpanFieldMarker({ spanRef: 'root', field: 'outputs', path: [{ k: 1 }] })).toBe(false);
  });
});

describe('unwrapSpanRefSelector', () => {
  it('unwraps the marker form and leaves bare selectors untouched', () => {
    expect(unwrapSpanRefSelector({ $spanRef: { type: 'TOOL' } })).toEqual({ type: 'TOOL' });
    expect(unwrapSpanRefSelector('root')).toBe('root');
  });
});

describe('resolveScalarSource', () => {
  it('resolves metrics.* to the current trace value', () => {
    expect(resolveScalarSource('metrics.latency', viewData)).toBe('275ms');
    expect(resolveScalarSource('metrics.status', viewData)).toBe('OK');
  });

  it('returns undefined for non-metric scalar names', () => {
    expect(resolveScalarSource('assessments', viewData)).toBeUndefined();
  });
});

describe('resolveSpanRef', () => {
  it('resolves root to the parentless span', () => {
    expect(resolveSpanRef('root', nodeMap)).toBe('root');
  });

  it('resolves type + nth in deterministic start order', () => {
    expect(resolveSpanRef({ type: 'TOOL' }, nodeMap)).toBe('tool0');
    expect(resolveSpanRef({ type: 'TOOL', nth: 1 }, nodeMap)).toBe('tool1');
  });

  it('resolves by name and returns undefined when nothing matches', () => {
    expect(resolveSpanRef({ name: 'run_sql_query' }, nodeMap)).toBe('tool0');
    expect(resolveSpanRef({ type: 'TOOL', nth: 5 }, nodeMap)).toBeUndefined();
    expect(resolveSpanRef('root', {})).toBeUndefined();
  });
});

describe('resolveSpanFieldSource', () => {
  it('serializes the selected span field for the current trace', () => {
    expect(resolveSpanFieldSource({ spanRef: { type: 'TOOL', nth: 0 }, field: 'outputs' }, nodeMap)).toBe('{"rows":1}');
    expect(resolveSpanFieldSource({ spanRef: 'root', field: 'name' }, nodeMap)).toBe('agent');
    expect(resolveSpanFieldSource({ spanRef: { type: 'TOOL', nth: 0 }, field: 'spanId' }, nodeMap)).toBe('tool0');
  });

  it('falls back to empty/null when the span is absent in this trace', () => {
    expect(resolveSpanFieldSource({ spanRef: { type: 'TOOL', nth: 9 }, field: 'outputs' }, nodeMap)).toBe('null');
    expect(resolveSpanFieldSource({ spanRef: { type: 'TOOL', nth: 9 }, field: 'name' }, nodeMap)).toBe('');
  });

  it('extracts a nested scalar leaf as raw text (renders as prose, not JSON)', () => {
    const nestedNodeMap = {
      root: makeNode({
        key: 'root',
        start: 0,
        title: 'agent',
        type: ModelSpanType.LLM,
        outputs: { choices: [{ message: { content: 'Hello world' } }] },
        inputs: { query: 'What is the weather?' },
      }),
    } satisfies Record<string, ModelTraceSpanNode>;
    expect(
      resolveSpanFieldSource(
        { spanRef: 'root', field: 'outputs', path: ['choices', 0, 'message', 'content'] },
        nestedNodeMap,
      ),
    ).toBe('Hello world');
    expect(resolveSpanFieldSource({ spanRef: 'root', field: 'inputs', path: ['query'] }, nestedNodeMap)).toBe(
      'What is the weather?',
    );
  });

  it('serializes a nested object/array leaf as JSON', () => {
    const nestedNodeMap = {
      root: makeNode({
        key: 'root',
        start: 0,
        title: 'agent',
        type: ModelSpanType.LLM,
        outputs: { choices: [{ message: { content: 'hi' } }] },
      }),
    } satisfies Record<string, ModelTraceSpanNode>;
    expect(resolveSpanFieldSource({ spanRef: 'root', field: 'outputs', path: ['choices', 0] }, nestedNodeMap)).toBe(
      '{"message":{"content":"hi"}}',
    );
  });

  it('returns empty string when a nested path is missing in this trace', () => {
    const nestedNodeMap = {
      root: makeNode({ key: 'root', start: 0, title: 'agent', type: ModelSpanType.LLM, outputs: { rows: 1 } }),
    } satisfies Record<string, ModelTraceSpanNode>;
    expect(
      resolveSpanFieldSource({ spanRef: 'root', field: 'outputs', path: ['missing', 'deep'] }, nestedNodeMap),
    ).toBe('');
    // Missing span + a path also degrades to '' rather than 'null'.
    expect(
      resolveSpanFieldSource({ spanRef: { type: 'TOOL', nth: 9 }, field: 'outputs', path: ['content'] }, nodeMap),
    ).toBe('');
  });

  it('preserves whole-object JSON behavior when no path is given', () => {
    expect(resolveSpanFieldSource({ spanRef: { type: 'TOOL', nth: 0 }, field: 'outputs' }, nodeMap)).toBe('{"rows":1}');
  });

  it('does not traverse the prototype chain (fails closed to empty)', () => {
    const nestedNodeMap = {
      root: makeNode({ key: 'root', start: 0, title: 'agent', type: ModelSpanType.LLM, outputs: { content: 'hi' } }),
    } satisfies Record<string, ModelTraceSpanNode>;
    // Inherited keys must NOT resolve to prototype objects/functions.
    expect(resolveSpanFieldSource({ spanRef: 'root', field: 'outputs', path: ['__proto__'] }, nestedNodeMap)).toBe('');
    expect(resolveSpanFieldSource({ spanRef: 'root', field: 'outputs', path: ['constructor'] }, nestedNodeMap)).toBe(
      '',
    );
    expect(
      resolveSpanFieldSource({ spanRef: 'root', field: 'outputs', path: ['__proto__', 'constructor'] }, nestedNodeMap),
    ).toBe('');
    // A real own key alongside those still works.
    expect(resolveSpanFieldSource({ spanRef: 'root', field: 'outputs', path: ['content'] }, nestedNodeMap)).toBe('hi');
  });

  it('only indexes real in-bounds array positions', () => {
    const nestedNodeMap = {
      root: makeNode({
        key: 'root',
        start: 0,
        title: 'agent',
        type: ModelSpanType.LLM,
        outputs: { items: ['a', 'b'] },
      }),
    } satisfies Record<string, ModelTraceSpanNode>;
    expect(resolveSpanFieldSource({ spanRef: 'root', field: 'outputs', path: ['items', 1] }, nestedNodeMap)).toBe('b');
    // Out-of-bounds and non-index keys (e.g. "length") degrade to ''.
    expect(resolveSpanFieldSource({ spanRef: 'root', field: 'outputs', path: ['items', 5] }, nestedNodeMap)).toBe('');
    expect(
      resolveSpanFieldSource({ spanRef: 'root', field: 'outputs', path: ['items', 'length'] }, nestedNodeMap),
    ).toBe('');
  });

  it('does not throw when the span field is not JSON-serializable', () => {
    const circular = { rows: 1 } satisfies Record<string, unknown>;
    (circular as Record<string, unknown>).self = circular;
    const cyclicNodeMap = {
      ...nodeMap,
      tool0: makeNode({
        key: 'tool0',
        start: 10,
        parentId: 'root',
        title: 'run_sql_query',
        type: ModelSpanType.TOOL,
        outputs: circular,
      }),
    } satisfies Record<string, ModelTraceSpanNode>;
    expect(resolveSpanFieldSource({ spanRef: { type: 'TOOL', nth: 0 }, field: 'outputs' }, cyclicNodeMap)).toBe(
      '"[object Object]"',
    );
  });
});
