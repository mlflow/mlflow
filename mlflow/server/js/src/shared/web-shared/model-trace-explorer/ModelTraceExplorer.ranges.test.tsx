import { describe, expect, it } from '@jest/globals';

import type { ModelTraceSpanV2 } from './ModelTrace.types';
import {
  getMaxModelTraceSpanNodeDepth as getLegacyMaxDepth,
  getModelTraceSpanNodeDepth as getLegacyDepth,
} from './timeline-tree/TimelineTree.utils';
import {
  getMaxModelTraceSpanNodeDepth as getMaxDepth,
  getModelTraceSpanNodeDepth as getDepth,
} from './v2/timeline-tree/TimelineTree.utils';
import { MOCK_TRACE } from './ModelTraceExplorer.test-utils';
import {
  getModelTraceTimelineRange as getLegacyTimelineRange,
  parseModelTraceToTreeWithMultipleRoots as parseLegacyTree,
} from './ModelTraceExplorer.utils';
import {
  getModelTraceTimelineRange,
  parseModelTraceToTreeWithMultipleRoots as parseTree,
} from './v2/ModelTraceExplorer.utils';

const SPAN_COUNT = 150000;

// Flat, minimal spans exercise wide traces without a deep recursive fixture.
const spans: ModelTraceSpanV2[] = Array.from({ length: SPAN_COUNT }, (_, index) => ({
  name: `span-${index}`,
  context: { span_id: index.toString(16).padStart(16, '0'), trace_id: 'trace' },
  start_time: 1000 + index,
  end_time: 1001 + index,
  attributes: {},
}));

describe('large trace time ranges', () => {
  it('preserves empty and non-finite timeline boundaries', () => {
    expect(getModelTraceTimelineRange([], {})).toEqual({ traceStartTime: 0, traceEndTime: 0 });
    const node = {
      key: 'invalid',
      title: 'invalid',
      start: NaN,
      end: Infinity,
      attributes: {},
      traceId: 'trace',
      assessments: [],
    };
    expect(getModelTraceTimelineRange([node], {})).toEqual({ traceStartTime: NaN, traceEndTime: Infinity });
  });

  it.each([
    ['legacy', parseLegacyTree],
    ['drawer', parseTree],
  ] as const)('parses %s trace spans without passing every timestamp as a call argument', (_version, parse) => {
    const nodes = parse({ ...MOCK_TRACE, data: { spans } });
    expect(nodes).toHaveLength(SPAN_COUNT);
    expect(nodes[0].start).toBe(0);
    expect(nodes[SPAN_COUNT - 1].end).toBe(SPAN_COUNT);
  });

  it.each([
    ['legacy', parseLegacyTree],
    ['drawer', parseTree],
  ] as const)('normalizes a wide %s V2 trace with explicit parent IDs', (_version, parse) => {
    const normalizedSpans = spans.map((span, index) => ({
      ...span,
      parent_id: index === 0 ? '' : spans[0].context.span_id,
    }));
    const nodes = parse({ ...MOCK_TRACE, data: { spans: normalizedSpans } });
    expect(nodes).toHaveLength(1);
    expect(nodes[0].start).toBe(0);
    expect(nodes[0].end).toBe(0.001);
    expect(nodes[0].children).toHaveLength(SPAN_COUNT - 1);
    expect(nodes[0].children?.[SPAN_COUNT - 2].end).toBe(SPAN_COUNT / 1000);
  });

  it('computes the legacy detail timeline range without spreading every root', () => {
    const node = { key: 'wide', start: -5, end: 10 };
    expect(getLegacyTimelineRange(Array.from({ length: SPAN_COUNT }, () => node))).toEqual({
      traceStartTime: -5,
      traceEndTime: 10,
    });
  });

  it('retains wider baseline bounds and roots without a candidate', () => {
    const nodes = [
      { key: 'paired', start: -20, end: 30 },
      { key: 'unpaired', start: -40, end: 50 },
    ];
    expect(getModelTraceTimelineRange(nodes, { paired: { key: 'paired', start: -10, end: 20 } })).toEqual({
      traceStartTime: -40,
      traceEndTime: 50,
    });
    expect(getModelTraceTimelineRange(nodes.slice(0, 1), { paired: { key: 'paired', start: -10, end: 20 } })).toEqual({
      traceStartTime: -20,
      traceEndTime: 30,
    });
  });

  it('includes all baseline and candidate top-level spans in a wide comparison range', () => {
    const node = { key: 'wide', title: 'wide', start: -5, end: 10, attributes: {}, traceId: 'trace', assessments: [] };
    const nodes = Array.from({ length: SPAN_COUNT }, () => node);
    expect(getModelTraceTimelineRange(nodes, { wide: { ...node, start: -10, end: 20 } })).toEqual({
      traceStartTime: -10,
      traceEndTime: 20,
    });
  });
});

describe.each([
  ['legacy', getLegacyDepth, getLegacyMaxDepth],
  ['drawer', getDepth, getMaxDepth],
] as const)('%s wide tree depths', (_version, depth, maxDepth) => {
  const leaf = { key: 'leaf', title: 'leaf', start: 0, end: 1, attributes: {}, traceId: 'trace', assessments: [] };
  const leaves = Array.from({ length: SPAN_COUNT }, () => leaf);
  it('finds the depth of a node with many direct children', () => {
    expect(depth({ ...leaf, children: leaves })).toBe(1);
  });
  it('finds the maximum depth across many roots without a spread', () => {
    expect(maxDepth([...leaves, { ...leaf, children: [{ ...leaf, children: [leaf] }] }])).toBe(2);
    expect(maxDepth([])).toBe(-Infinity);
  });
});
