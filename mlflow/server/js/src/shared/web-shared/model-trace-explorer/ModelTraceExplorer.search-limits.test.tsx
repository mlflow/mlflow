import { describe, expect, it } from '@jest/globals';

import { ModelSpanType, type ModelTraceSpanNode } from './ModelTrace.types';
import { getMatchesFromSpan, searchTree } from './ModelTraceExplorer.utils';
import { TEST_SPAN_FILTER_STATE } from './timeline-tree/TimelineTree.test-utils';

const node = (key: string): ModelTraceSpanNode => ({
  key,
  title: key,
  children: [],
  attributes: {},
  start: 0,
  end: 1,
  type: ModelSpanType.CHAIN,
  assessments: [],
  traceId: 'test',
});
const MATCH_COUNT = 150_000;

describe('shared trace search result aggregation', () => {
  it('keeps all event matches above the argument limit', () => {
    const span = node('event');
    span.events = [{ name: 'event', timestamp: 0, attributes: { value: 'x'.repeat(MATCH_COUNT) } }];
    const matches = getMatchesFromSpan(span, 'x');
    expect(matches).toHaveLength(MATCH_COUNT);
    expect(matches[MATCH_COUNT - 1]).toMatchObject({ span, section: 'events', matchIndex: MATCH_COUNT - 1 });
  });

  it('keeps all link matches above the argument limit', () => {
    const span = node('link');
    span.links = [{ trace_id: 'other', span_id: 'other', attributes: { value: 'x'.repeat(MATCH_COUNT) } }];
    const matches = getMatchesFromSpan(span, 'x');
    expect(matches).toHaveLength(MATCH_COUNT);
    expect(matches[MATCH_COUNT - 1]).toMatchObject({ span, section: 'links', matchIndex: MATCH_COUNT - 1 });
  });

  it('retains parent-first order while aggregating a child event above the argument limit', () => {
    const root = node('root');
    root.inputs = { value: 'x' };
    const child = node('child');
    child.events = [{ name: 'event', timestamp: 0, attributes: { value: 'x'.repeat(MATCH_COUNT) } }];
    root.children = [child];
    const { matches, filteredTreeNodes } = searchTree(root, 'x', TEST_SPAN_FILTER_STATE);
    expect(matches).toHaveLength(MATCH_COUNT + 1);
    expect(matches[0]).toMatchObject({ span: root, section: 'inputs', matchIndex: 0 });
    expect(matches[MATCH_COUNT]).toMatchObject({ span: child, section: 'events', matchIndex: MATCH_COUNT - 1 });
    expect(filteredTreeNodes[0].children?.[0].key).toBe('child');
  });
});
