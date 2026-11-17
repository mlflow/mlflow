import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { useState } from 'react';

import { ModelSpanType, type ModelTrace, type ModelTraceSpanNode } from '../ModelTrace.types';
import { useModelTraceSearch } from './useModelTraceSearch';
import { useModelTraceExpansion } from './useModelTraceExpansion';

const createNode = (value: string): ModelTraceSpanNode => ({
  key: 'root',
  title: 'root',
  children: [],
  attributes: { value },
  start: 0,
  end: 1,
  type: ModelSpanType.CHAIN,
  assessments: [],
  traceId: 'tr-search-fixture',
});

const renderSearch = (root: ModelTraceSpanNode, selectedNode = root) => {
  const setSelectedNodeAndTab = jest.fn();
  const callbacks = { setSelectedNode: jest.fn(), setSelectedNodeAndTab, setExpandedKeys: jest.fn() };
  const hook = renderHook(
    ({ treeNodes, modelTraceInfo }: { treeNodes: ModelTraceSpanNode[]; modelTraceInfo: ModelTrace['info'] | null }) =>
      useModelTraceSearch({ ...callbacks, treeNodes, selectedNode, modelTraceInfo }),
    { initialProps: { treeNodes: [root], modelTraceInfo: null as ModelTrace['info'] | null } },
  );
  return { ...hook, ...callbacks };
};

describe('useModelTraceSearch content updates', () => {
  afterEach(() => {
    jest.restoreAllMocks();
    jest.useRealTimers();
  });

  it('searches updated payloads when span IDs stay the same', () => {
    const { result, rerender } = renderSearch(createNode('needle'));
    act(() => result.current.setSearchFilter('needle'));
    expect(result.current.matchData.totalMatches).toBe(1);

    rerender({ treeNodes: [createNode('needle needle')], modelTraceInfo: null });
    expect(result.current.matchData.totalMatches).toBe(2);
  });

  it('keeps the active match when trace metadata changes', () => {
    const root = createNode('needle needle');
    const { result, rerender, setSelectedNodeAndTab } = renderSearch(root);
    act(() => result.current.setSearchFilter('needle'));
    act(() => result.current.handleNextSearchMatch());
    expect(result.current.matchData.currentMatchIndex).toBe(1);
    setSelectedNodeAndTab.mockClear();

    rerender({ treeNodes: [root], modelTraceInfo: { tags: [{ key: 'note', value: 'updated' }] } });
    expect(result.current.matchData.currentMatchIndex).toBe(1);
    expect(setSelectedNodeAndTab).not.toHaveBeenCalled();
  });

  it('keeps the same field occurrence when an earlier field gains a match', () => {
    const { result, rerender } = renderSearch(createNode('needle needle'));
    act(() => result.current.setSearchFilter('needle'));
    act(() => result.current.handleNextSearchMatch());

    const updated = { ...createNode('needle needle'), attributes: { before: 'needle', value: 'needle needle' } };
    rerender({ treeNodes: [updated], modelTraceInfo: null });
    expect(result.current.matchData.totalMatches).toBe(3);
    expect(result.current.matchData.currentMatchIndex).toBe(2);
    expect(result.current.matchData.match).toMatchObject({ key: 'value', matchIndex: 1 });
  });

  it('reselects the refreshed child while retaining its active occurrence', () => {
    const child = { ...createNode('needle needle'), key: 'child' };
    const root = { ...createNode('absent'), children: [child] };
    const { result, rerender, setSelectedNode, setSelectedNodeAndTab } = renderSearch(root, child);
    act(() => result.current.setSearchFilter('needle'));
    act(() => result.current.handleNextSearchMatch());
    setSelectedNode.mockClear();
    setSelectedNodeAndTab.mockClear();

    const updatedChild = { ...child, title: 'updated child' };
    rerender({ treeNodes: [{ ...root, children: [updatedChild] }], modelTraceInfo: null });
    expect(result.current.matchData.currentMatchIndex).toBe(1);
    expect(result.current.matchData.match).toMatchObject({ span: updatedChild, key: 'value', matchIndex: 1 });
    expect(setSelectedNode).toHaveBeenLastCalledWith(updatedChild);
    expect(setSelectedNodeAndTab).not.toHaveBeenCalled();
  });

  it('starts a different query at its first match', () => {
    const { result } = renderSearch(createNode('needle needle other'));
    act(() => result.current.setSearchFilter('needle'));
    act(() => result.current.handleNextSearchMatch());

    act(() => result.current.setSearchFilter('other'));
    expect(result.current.matchData.totalMatches).toBe(1);
    expect(result.current.matchData.currentMatchIndex).toBe(0);
  });

  it('resets the active occurrence when span filter state changes', () => {
    const { result, setSelectedNodeAndTab } = renderSearch(createNode('needle needle'));
    act(() => result.current.setSearchFilter('needle'));
    act(() => result.current.handleNextSearchMatch());
    expect(result.current.matchData.currentMatchIndex).toBe(1);
    setSelectedNodeAndTab.mockClear();
    act(() => result.current.setSpanFilterState({ ...result.current.spanFilterState, showExceptions: false }));
    expect(result.current.matchData.totalMatches).toBe(2);
    expect(result.current.matchData.currentMatchIndex).toBe(0);
    expect(setSelectedNodeAndTab).toHaveBeenCalledTimes(1);
  });

  it('refreshes a deliberately selected span without returning to the active match span', () => {
    const child = { ...createNode('needle'), key: 'child' };
    const root: ModelTraceSpanNode = { ...createNode('needle'), children: [child] };
    const setSelectedNode = jest.fn();
    const callbacks = { setSelectedNode, setSelectedNodeAndTab: jest.fn(), setExpandedKeys: jest.fn() };
    const { result, rerender } = renderHook(
      ({ treeNodes, selectedNode }) =>
        useModelTraceSearch({ ...callbacks, treeNodes, selectedNode, modelTraceInfo: null }),
      { initialProps: { treeNodes: [root], selectedNode: root } },
    );
    act(() => result.current.setSearchFilter('needle'));
    expect(result.current.matchData.match?.span).toBe(root);
    rerender({ treeNodes: [root], selectedNode: child });
    setSelectedNode.mockClear();
    const updatedChild = { ...child, title: 'refreshed child' };
    const updatedRoot = { ...root, children: [updatedChild] };
    rerender({ treeNodes: [updatedRoot], selectedNode: child });
    expect(result.current.matchData.match?.span).toBe(updatedRoot);
    expect(setSelectedNode).toHaveBeenLastCalledWith(updatedChild);
    expect(callbacks.setSelectedNodeAndTab).toHaveBeenCalledTimes(1);
  });

  it('selects a remaining match when the active field disappears', () => {
    const root = { ...createNode('needle'), attributes: { first: 'needle', last: 'needle' } };
    const { result, rerender, setSelectedNodeAndTab } = renderSearch(root);
    act(() => result.current.setSearchFilter('needle'));
    act(() => result.current.handleNextSearchMatch());
    expect(result.current.matchData.match?.key).toBe('last');
    setSelectedNodeAndTab.mockClear();

    const updated = { ...root, attributes: { first: 'needle' } };
    rerender({ treeNodes: [updated], modelTraceInfo: null });
    expect(result.current.matchData.currentMatchIndex).toBe(0);
    expect(result.current.matchData.match?.key).toBe('first');
    expect(setSelectedNodeAndTab).toHaveBeenCalledWith(updated, 'attributes');
  });

  it('reveals the remaining match under a collapsed branch when the active match disappears', () => {
    const leafA = { ...createNode('needle'), key: 'leaf-a', parentId: 'branch-a' };
    const leafB = { ...createNode('needle'), key: 'leaf-b', parentId: 'branch-b' };
    const branchA = { ...createNode('absent'), key: 'branch-a', parentId: 'root', children: [leafA] };
    const branchB = { ...createNode('absent'), key: 'branch-b', parentId: 'root', children: [leafB] };
    const root = { ...createNode('absent'), children: [branchA, branchB] };
    const callbacks = { setSelectedNode: jest.fn(), setSelectedNodeAndTab: jest.fn() };
    const { result, rerender } = renderHook(
      ({ treeNodes }) => {
        const [expanded, setExpandedKeys] = useState<Set<string | number>>(new Set());
        const search = useModelTraceSearch({
          ...callbacks,
          treeNodes,
          selectedNode: root,
          modelTraceInfo: null,
          setExpandedKeys,
        });
        useModelTraceExpansion(
          search.filteredTreeNodes,
          root.traceId,
          search.searchFilter,
          search.spanFilterState,
          setExpandedKeys,
        );
        return { ...search, expanded, setExpandedKeys };
      },
      { initialProps: { treeNodes: [root] } },
    );
    act(() => result.current.setSearchFilter('needle'));
    expect(result.current.matchData.match?.span.key).toBe('leaf-a');
    act(() => result.current.setExpandedKeys(new Set(['root', 'branch-a'])));
    expect(result.current.expanded.has('branch-b')).toBe(false);
    const updated = {
      ...root,
      children: [{ ...branchA, children: [{ ...leafA, attributes: { value: 'gone' } }] }, branchB],
    };
    rerender({ treeNodes: [updated] });
    expect(result.current.matchData.match?.span.key).toBe('leaf-b');
    expect(callbacks.setSelectedNodeAndTab).toHaveBeenLastCalledWith(leafB, 'attributes');
    expect(result.current.expanded.has('root')).toBe(true);
    expect(result.current.expanded.has('branch-b')).toBe(true);
  });

  it('keeps displayed results and highlighting paired until the latest collection completes', () => {
    const child = { ...createNode('needle replacement replacement replacement replacement'), key: 'child' };
    const root = { ...createNode('needle needle other'), children: [child] };
    const { result } = renderSearch(root);
    act(() => result.current.setSearchFilter('needle'));
    expect(result.current.matchData.totalMatches).toBe(3);
    jest.useFakeTimers();
    let clock = 0;
    jest.spyOn(performance, 'now').mockImplementation(() => (clock += 100));

    act(() => result.current.setSearchFilter('other'));
    expect(result.current.searchFilter).toBe('needle');
    expect(result.current.matchData.totalMatches).toBe(3);
    expect(result.current.matchData.match?.span).toBe(root);
    act(() => result.current.setSearchFilter('replacement'));
    expect(result.current.searchFilter).toBe('needle');
    expect(result.current.matchData.totalMatches).toBe(3);
    act(() => jest.runAllTimers());
    expect(result.current.searchFilter).toBe('replacement');
    expect(result.current.matchData.totalMatches).toBe(4);
    expect(result.current.matchData.match?.span).toBe(child);
  });

  it('cancels pending collection on clear and unmount', () => {
    const root = { ...createNode('needle'), children: [createNode('needle')] };
    const { result, unmount } = renderSearch(root);
    jest.useFakeTimers();
    let clock = 0;
    jest.spyOn(performance, 'now').mockImplementation(() => (clock += 100));

    act(() => result.current.setSearchFilter('needle'));
    expect(jest.getTimerCount()).toBe(1);
    act(() => result.current.setSearchFilter(''));
    expect(jest.getTimerCount()).toBe(0);
    expect(result.current.searchFilter).toBe('');
    expect(result.current.matchData.totalMatches).toBe(0);
    act(() => result.current.setSearchFilter('needle'));
    expect(jest.getTimerCount()).toBe(1);
    unmount();
    expect(jest.getTimerCount()).toBe(0);
  });

  it('does not show another trace while its replacement is collecting', () => {
    const { result, rerender } = renderSearch(createNode('needle'));
    act(() => result.current.setSearchFilter('needle'));
    expect(result.current.matchData.totalMatches).toBe(1);
    jest.useFakeTimers();
    let clock = 0;
    jest.spyOn(performance, 'now').mockImplementation(() => (clock += 100));
    const other = {
      ...createNode('needle'),
      traceId: 'other-trace',
      children: [{ ...createNode('needle'), key: 'child', traceId: 'other-trace' }],
    };

    rerender({ treeNodes: [other], modelTraceInfo: null });
    expect(result.current.matchData.totalMatches).toBe(0);
    expect(result.current.filteredTreeNodes).toEqual([]);
    act(() => jest.runAllTimers());
    expect(result.current.matchData.totalMatches).toBe(2);
    expect(result.current.matchData.match?.span.traceId).toBe('other-trace');
  });
});
