import { describe, expect, it } from '@jest/globals';

import type { ModelTraceSpanV2 } from './ModelTrace.types';
import { MOCK_TRACE } from './ModelTraceExplorer.test-utils';
import { parseModelTraceToTreeWithMultipleRoots as parseLegacyTree } from './ModelTraceExplorer.utils';
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

  it('includes all baseline and candidate top-level spans in a wide comparison range', () => {
    const node = { key: 'wide', title: 'wide', start: -5, end: 10, attributes: {}, traceId: 'trace', assessments: [] };
    const nodes = Array.from({ length: SPAN_COUNT }, () => node);
    expect(getModelTraceTimelineRange(nodes, { wide: { ...node, start: -10, end: 20 } })).toEqual({
      traceStartTime: -10,
      traceEndTime: 20,
    });
  });
});
