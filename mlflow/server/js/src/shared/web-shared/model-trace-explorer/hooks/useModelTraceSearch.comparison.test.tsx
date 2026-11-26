import { describe, expect, jest, it } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';

import { ModelSpanType, type ModelTraceSpanNode } from '../ModelTrace.types';
import { useModelTraceSearch } from './useModelTraceSearch';

const createNode = (key: string, links?: ModelTraceSpanNode['links']): ModelTraceSpanNode => ({
  key,
  title: 'retrieve',
  children: [],
  attributes: { query: 'shared value' },
  links,
  start: 0,
  end: 1,
  type: ModelSpanType.CHAIN,
  assessments: [],
  traceId: 'test',
});

describe('useModelTraceSearch in a trace comparison', () => {
  it('skips matches in candidate-only links, which the Links tab cannot show', () => {
    const baselineNode = createNode('baseline');
    const candidateNode = createNode('candidate', [
      { trace_id: 'tr-linked', span_id: 'linked', attributes: { reason: 'candidate link value' } },
    ]);
    const renderSearch = () =>
      renderHook(() =>
        useModelTraceSearch({
          treeNodes: [baselineNode],
          selectedNode: baselineNode,
          setSelectedNode: jest.fn(),
          setSelectedNodeAndTab: jest.fn(),
          setExpandedKeys: jest.fn(),
          modelTraceInfo: null,
          comparisonNodeMap: { baseline: candidateNode },
        }),
      );

    const { result } = renderSearch();
    act(() => result.current.setSearchFilter('candidate link'));
    expect(result.current.matchData.totalMatches).toBe(0);

    act(() => result.current.setSearchFilter('shared value'));
    expect(result.current.matchData.totalMatches).toBe(1);
  });
});
