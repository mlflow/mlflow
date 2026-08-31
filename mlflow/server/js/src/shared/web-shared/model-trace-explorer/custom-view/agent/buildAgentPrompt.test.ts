import { describe, test, expect, it } from '@jest/globals';
import {
  RENDER_CUSTOM_VIEW_TOOL_NAME,
  buildCustomViewAuthoringGuide,
  buildAgentDataSnapshot,
} from './buildAgentPrompt';
import type { AgentNode, AgentTraceData } from './buildAgentPrompt';

describe('buildCustomViewAuthoringGuide', () => {
  test('instructs the agent to deliver the view by calling the render_custom_view tool', () => {
    const guide = buildCustomViewAuthoringGuide();
    expect(guide).toContain(`CALL the \`${RENDER_CUSTOM_VIEW_TOOL_NAME}\` tool`);
  });

  test('provides a schema-envelope contract for structured-output providers', () => {
    const guide = buildCustomViewAuthoringGuide('structured');
    expect(guide).toContain('structured final response');
    expect(guide).toContain('"type": "render_custom_view"');
    expect(guide).toContain('"type" to "message"');
    expect(guide).not.toContain(`CALL the \`${RENDER_CUSTOM_VIEW_TOOL_NAME}\` tool`);
  });

  test('includes the catalog/binding/layout/example building blocks', () => {
    const guide = buildCustomViewAuthoringGuide();
    for (const shared of [
      '"Row": horizontal layout',
      'Data binding (CRITICAL',
      'Layout & visual polish',
      'Trace Summary',
    ]) {
      expect(guide).toContain(shared);
    }
  });

  test('documents immediate and staged feedback authoring', () => {
    const guide = buildCustomViewAuthoringGuide();
    for (const primitive of ['FeedbackThumbsUpDownButtons', 'RadioGroup', 'FeedbackInputText', 'FeedbackSubmit']) {
      expect(guide).toContain(`"${primitive}"`);
    }
    expect(guide).toContain('"$spanRef" markers');
    expect(guide).toContain('MUST carry a "formId"');
    expect(guide).toContain('Example — a multi-dimension human-feedback form');
  });

  // A StatCard's `icon`/`tone` are static enums the host never re-resolves, while
  // its `value` is always a bound marker. An example that asserts a verdict —
  // a success check on a status, a warning tint on a latency — teaches the model
  // a view whose styling contradicts the value on some trace.
  test('styles every example StatCard neutrally', () => {
    const guide = buildCustomViewAuthoringGuide();
    const tiles = guide.split('\n').filter((line) => line.includes('"component": "StatCard"'));

    expect(tiles.length).toBeGreaterThan(0);
    for (const tile of tiles) {
      expect(tile).toContain('"tone": "info"');
      expect(tile).not.toMatch(/"icon": "(checkCircle|xCircle)"/);
    }
  });
});

const makeNode = (over: Partial<AgentNode> = {}): AgentNode => ({
  name: 'span',
  type: 'TOOL',
  startMs: 0,
  endMs: 1,
  durationMs: 1,
  inputs: {},
  outputs: {},
  ...over,
});

const makeNodeMap = (count: number): Record<string, AgentNode> =>
  Object.fromEntries(Array.from({ length: count }, (_, i) => [`s${i}`, makeNode({ name: `span-${i}` })]));

const baseData = (over: Partial<AgentTraceData> = {}): AgentTraceData => ({
  metrics: { status: 'OK' },
  ...over,
});

describe('buildAgentDataSnapshot', () => {
  it('passes through metrics and assessments unchanged', () => {
    const snapshot = buildAgentDataSnapshot(
      baseData({
        metrics: { status: 'OK', latency: '1.2s' },
        assessments: [{ name: 'correctness', value: true, source: 'LLM_JUDGE' }],
      }),
    );
    expect(snapshot.metrics).toEqual({ status: 'OK', latency: '1.2s' });
    expect(snapshot.assessments).toEqual([{ name: 'correctness', value: true, source: 'LLM_JUDGE' }]);
  });

  it('defaults nodeMap and assessments to empty when absent', () => {
    const snapshot = buildAgentDataSnapshot(baseData());
    expect(snapshot.nodeMap).toEqual({});
    expect(snapshot.nodeMapTruncated).toBe(0);
    expect(snapshot.assessments).toEqual([]);
  });

  describe('nodeMap cap (limit 400)', () => {
    it('keeps all entries and reports 0 truncated at exactly the limit', () => {
      const snapshot = buildAgentDataSnapshot(baseData({ nodeMap: makeNodeMap(400) }));
      expect(Object.keys(snapshot.nodeMap as Record<string, unknown>)).toHaveLength(400);
      expect(snapshot.nodeMapTruncated).toBe(0);
    });

    it('caps to 400 entries and reports the overflow count when over the limit', () => {
      const snapshot = buildAgentDataSnapshot(baseData({ nodeMap: makeNodeMap(430) }));
      expect(Object.keys(snapshot.nodeMap as Record<string, unknown>)).toHaveLength(400);
      expect(snapshot.nodeMapTruncated).toBe(30);
    });
  });

  describe('per-span input/output truncation (limit 2000 serialized chars)', () => {
    it('leaves small values structured and untouched', () => {
      const nodeMap = { s0: makeNode({ inputs: { q: 'select 1' }, outputs: { rows: 1 } }) };
      const snapshot = buildAgentDataSnapshot(baseData({ nodeMap }));
      const node = (snapshot.nodeMap as Record<string, AgentNode>).s0;
      expect(node.inputs).toEqual({ q: 'select 1' });
      expect(node.outputs).toEqual({ rows: 1 });
    });

    it('truncates large values to a string with a truncation marker', () => {
      const bigInputs = { blob: 'x'.repeat(5000) };
      const bigOutputs = { blob: 'y'.repeat(5000) };
      const nodeMap = { s0: makeNode({ inputs: bigInputs, outputs: bigOutputs }) };
      const snapshot = buildAgentDataSnapshot(baseData({ nodeMap }));
      const node = (snapshot.nodeMap as Record<string, AgentNode>).s0;

      expect(typeof node.inputs).toBe('string');
      expect(node.inputs as string).toMatch(/… \(truncated\)$/);
      // Marker aside, the retained slice is exactly the 2000-char cap.
      expect((node.inputs as string).replace('… (truncated)', '')).toHaveLength(2000);
      expect(typeof node.outputs).toBe('string');
      expect(node.outputs as string).toMatch(/… \(truncated\)$/);
    });

    it('preserves null/undefined field values without stringifying them', () => {
      const nodeMap = { s0: makeNode({ inputs: null, outputs: undefined }) };
      const snapshot = buildAgentDataSnapshot(baseData({ nodeMap }));
      const node = (snapshot.nodeMap as Record<string, AgentNode>).s0;
      expect(node.inputs).toBeNull();
      expect(node.outputs).toBeUndefined();
    });
  });
});
