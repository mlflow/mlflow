import { afterEach, describe, expect, it, jest } from '@jest/globals';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { getMatchesFromSpan } from '../ModelTraceExplorer.utils';
import { collectModelTraceSearchMatches } from './collectModelTraceSearchMatches';

const span = (key: string, value = 'needle'): ModelTraceSpanNode => ({
  key,
  title: key,
  attributes: { value },
  start: 0,
  end: 1,
  assessments: [],
  traceId: 'tr-search-fixture',
});

describe('collectModelTraceSearchMatches', () => {
  afterEach(() => {
    jest.restoreAllMocks();
    jest.useRealTimers();
  });

  it('retains exact matches and zero-match spans across roots and descendants', async () => {
    const child = span('child');
    const root = { ...span('root', 'absent'), children: [child] };
    const other = span('other');
    const matches = await new Promise<Map<ModelTraceSpanNode, unknown>>((resolve) => {
      collectModelTraceSearchMatches([root, other], 'needle', resolve);
    });
    expect(matches.size).toBe(3);
    for (const node of [root, child, other]) expect(matches.get(node)).toEqual(getMatchesFromSpan(node, 'needle'));
  });

  it('yields between batches and suppresses completion after cancellation', () => {
    jest.useFakeTimers();
    let clock = 0;
    jest.spyOn(performance, 'now').mockImplementation(() => (clock += 100));
    const complete = jest.fn();
    const cancel = collectModelTraceSearchMatches(
      Array.from({ length: 50 }, (_, index) => span(String(index))),
      'needle',
      complete,
    );
    expect(complete).not.toHaveBeenCalled();
    expect(jest.getTimerCount()).toBe(1);
    cancel();
    jest.runAllTimers();
    expect(complete).not.toHaveBeenCalled();
    expect(jest.getTimerCount()).toBe(0);
  });

  it('visits repeated and cyclic span references once', async () => {
    const root = span('root');
    root.children = [root];
    const matches = await new Promise<Map<ModelTraceSpanNode, unknown>>((resolve) => {
      collectModelTraceSearchMatches([root, root], 'needle', resolve);
    });
    expect(matches.size).toBe(1);
    expect(matches.get(root)).toEqual(getMatchesFromSpan(root, 'needle'));
  });

  it('reuses unchanged payload counts without retaining the old span wrapper', () => {
    const root = { ...span('root'), inputs: { value: 'needle needle' } };
    const expected = getMatchesFromSpan(root, 'needle');
    const rebuilt = { ...root, title: 'new title' };
    const stringify = jest.spyOn(JSON, 'stringify');
    const matches = getMatchesFromSpan(rebuilt, 'needle');
    expect(stringify).not.toHaveBeenCalled();
    expect(matches).toEqual(expected.map((match) => ({ ...match, span: rebuilt })));
    expect(matches.every((match) => match.span === rebuilt)).toBe(true);
  });

  it('invalidates counts for a new query or replacement payload', () => {
    const root = span('root', 'needle needle other');
    expect(getMatchesFromSpan(root, 'needle')).toHaveLength(2);
    expect(getMatchesFromSpan(root, 'other')).toHaveLength(1);
    expect(getMatchesFromSpan({ ...root, attributes: { value: 'other other' } }, 'other')).toHaveLength(2);
    expect(getMatchesFromSpan(root, '')).toEqual([]);
  });
});
