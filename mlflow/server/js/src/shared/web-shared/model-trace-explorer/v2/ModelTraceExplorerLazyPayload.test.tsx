import { describe, expect, it, jest } from '@jest/globals';
import { act, renderHook, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '@databricks/web-shared/test-utils/render';

import { useModelTraceSearch } from '../hooks/useModelTraceSearch';
import { MOCK_V3_SPANS, MOCK_V3_TRACE } from '../ModelTraceExplorer.test-utils';
import {
  normalizeNewSpanData as normalizeLegacySpan,
  searchTree as searchLegacyTree,
} from '../ModelTraceExplorer.utils';
import { TEST_SPAN_FILTER_STATE as LEGACY_FILTER_STATE } from '../timeline-tree/TimelineTree.test-utils';
import { ModelSpanType } from './ModelTrace.types';
import { buildModelTraceComparisonForest, normalizeNewSpanData, searchTree } from './ModelTraceExplorer.utils';
import { TEST_SPAN_FILTER_STATE } from './timeline-tree/TimelineTree.test-utils';
import {
  ModelTraceExplorerViewStateProvider,
  useModelTraceExplorerViewState,
} from './ModelTraceExplorerViewStateContext';

const inputs = JSON.stringify({ request: 'café 🌳\nneedle', rows: Array.from({ length: 1000 }, (_, id) => ({ id })) });
const outputs = JSON.stringify({ response: 'tail' });
const metadata = JSON.stringify({ custom: 'retained' });
const attributes = {
  'mlflow.spanType': JSON.stringify(ModelSpanType.CHAIN),
  'mlflow.spanInputs': inputs,
  'mlflow.spanOutputs': outputs,
  custom: metadata,
};
const makeNode = (replacement = attributes) =>
  normalizeNewSpanData({ ...MOCK_V3_SPANS[0], attributes: replacement }, 0, 1000, [], {}, 'trace');

const watchPayloads = () => {
  const parse = jest.spyOn(JSON, 'parse');
  return { parse, calls: (data: string) => parse.mock.calls.filter(([text]) => text === data).length };
};

describe('lazy span payloads', () => {
  it('leaves an unselected child payload unread in the mounted explorer state until selection', async () => {
    const childInputs = JSON.stringify({ value: 'unselected child café 🌳' });
    const Probe = () => {
      const { rootNode, selectedNode, setSelectedNode } = useModelTraceExplorerViewState();
      return (
        <>
          <div data-testid="selected">{JSON.stringify(selectedNode?.inputs)}</div>
          <button
            onClick={() => {
              const child = rootNode?.children?.[0];
              if (child) setSelectedNode(child);
            }}
          >
            Select child
          </button>
        </>
      );
    };
    const { parse, calls } = watchPayloads();
    try {
      render(
        <ModelTraceExplorerViewStateProvider
          modelTrace={{
            ...MOCK_V3_TRACE,
            data: {
              spans: [
                { ...MOCK_V3_SPANS[0], attributes },
                {
                  ...MOCK_V3_SPANS[1],
                  parent_span_id: MOCK_V3_SPANS[0].span_id,
                  attributes: { ...attributes, 'mlflow.spanInputs': childInputs },
                },
              ],
            },
          }}
          assessmentsPaneEnabled={false}
        >
          <Probe />
        </ModelTraceExplorerViewStateProvider>,
      );
      expect(screen.getByTestId('selected')).toHaveTextContent('needle');
      expect(calls(childInputs)).toBe(0);
      await userEvent.click(screen.getByRole('button', { name: 'Select child' }));
      expect(screen.getByTestId('selected')).toHaveTextContent('unselected child café 🌳');
      expect(calls(childInputs)).toBe(1);
    } finally {
      parse.mockRestore();
    }
  });

  it('preserves deferred payloads through legacy parsing and type-only filtering', () => {
    const { parse, calls } = watchPayloads();
    try {
      const node = normalizeLegacySpan({ ...MOCK_V3_SPANS[0], attributes }, 0, 1000, [], {}, 'trace');
      const result = searchLegacyTree(node, '', {
        ...LEGACY_FILTER_STATE,
        spanTypeDisplayState: { ...LEGACY_FILTER_STATE.spanTypeDisplayState, LLM: false },
      });
      expect(result.filteredTreeNodes).toHaveLength(1);
      expect([calls(inputs), calls(outputs), calls(metadata)]).toEqual([0, 0, 0]);
      expect(result.filteredTreeNodes[0].inputs).toBe(node.inputs);
      expect(calls(inputs)).toBe(1);
    } finally {
      parse.mockRestore();
    }
  });

  it('keeps unselected payloads deferred when the comparison search hook filters span types', () => {
    const childInputs = JSON.stringify({ value: 'comparison child' });
    const { parse, calls } = watchPayloads();
    try {
      const child = normalizeLegacySpan(
        { ...MOCK_V3_SPANS[1], attributes: { ...attributes, 'mlflow.spanInputs': childInputs } },
        0,
        1000,
        [],
        {},
        'trace',
      );
      const root = normalizeLegacySpan({ ...MOCK_V3_SPANS[0], attributes }, 0, 1000, [child], {}, 'trace');
      const comparison = normalizeLegacySpan({ ...MOCK_V3_SPANS[0], attributes }, 0, 1000, [], {}, 'candidate');
      const comparisonNodeMap = { [String(root.key)]: comparison };
      const { result } = renderHook(() =>
        useModelTraceSearch({
          treeNodes: [root],
          selectedNode: root,
          setSelectedNode: jest.fn(),
          setSelectedNodeAndTab: jest.fn(),
          setExpandedKeys: jest.fn(),
          modelTraceInfo: null,
          comparisonNodeMap,
        }),
      );
      act(() =>
        result.current.setSpanFilterState({
          ...result.current.spanFilterState,
          spanTypeDisplayState: { ...result.current.spanFilterState.spanTypeDisplayState, LLM: false },
        }),
      );
      const filtered = result.current.filteredTreeNodes[0];
      expect(filtered).not.toBe(root);
      expect(filtered.children?.[0]).not.toBe(child);
      expect([calls(childInputs), calls(outputs), calls(metadata)]).toEqual([0, 0, 0]);
      expect(filtered.children?.[0].inputs).toEqual({ value: 'comparison child' });
      expect(calls(childInputs)).toBe(1);
    } finally {
      parse.mockRestore();
    }
  });

  it('does not read output or attribute payloads while aligning comparison trees', () => {
    const { parse, calls } = watchPayloads();
    try {
      const baseline = makeNode();
      const candidate = makeNode();
      const result = buildModelTraceComparisonForest([baseline], [candidate]);
      expect(calls(inputs)).toBe(2);
      expect(calls(outputs)).toBe(0);
      expect(calls(metadata)).toBe(0);
      expect(result.topLevelNodes[0].outputs).toBe(baseline.outputs);
      expect(calls(outputs)).toBe(1);
      expect(result.comparisonNodeMap[String(baseline.key)]).toBe(candidate);
    } finally {
      parse.mockRestore();
    }
  });

  it('keeps payloads unparsed until read and shares cached reads with chat normalization', () => {
    const { parse, calls } = watchPayloads();
    try {
      const node = makeNode();
      expect([calls(inputs), calls(outputs), calls(metadata)]).toEqual([0, 0, 0]);
      expect(node.inputs).toEqual(JSON.parse(inputs));
      expect(node.inputs).toBe(node.inputs);
      expect(calls(inputs)).toBe(2);
      expect(calls(outputs)).toBe(0);
      void node.chatMessages;
      void node.chatMessages;
      void node.chatTools;
      expect(calls(inputs)).toBe(2);
      expect(calls(outputs)).toBe(1);
      expect(calls(metadata)).toBe(0);
      expect(node.attributes).toEqual({ custom: { custom: 'retained' } });
      expect(node.attributes).toBe(node.attributes);
      expect(calls(metadata)).toBe(1);
    } finally {
      parse.mockRestore();
    }
  });

  it('keeps cached undefined values and independent same-id replacement data', () => {
    const { parse, calls } = watchPayloads();
    try {
      const node = makeNode();
      const next = makeNode({ ...attributes, 'mlflow.spanOutputs': 'null' });
      expect(node.key).toBe(next.key);
      expect(node.outputs).toEqual({ response: 'tail' });
      expect(next.outputs).toBeNull();
      expect(next.chatMessages).toBeUndefined();
      expect(next.chatMessages).toBeUndefined();
      expect(calls(outputs)).toBe(1);
      expect(calls(metadata)).toBe(0);
    } finally {
      parse.mockRestore();
    }
  });

  it('does not materialize payloads when only span types are filtered', () => {
    const { parse, calls } = watchPayloads();
    try {
      const node = makeNode();
      const result = searchTree(node, '', {
        ...TEST_SPAN_FILTER_STATE,
        spanTypeDisplayState: { ...TEST_SPAN_FILTER_STATE.spanTypeDisplayState, [ModelSpanType.LLM]: false },
      });
      expect(result.filteredTreeNodes).toHaveLength(1);
      expect([calls(inputs), calls(outputs), calls(metadata)]).toEqual([0, 0, 0]);
      const visible = result.filteredTreeNodes[0];
      expect(visible.inputs).toBe(node.inputs);
      expect(calls(inputs)).toBe(1);
    } finally {
      parse.mockRestore();
    }
  });

  it('preserves all enumerable values when callers explicitly spread a node', () => {
    const node = makeNode();
    const spread = { ...node };
    expect(spread.inputs).toEqual(JSON.parse(inputs));
    expect(spread.outputs).toEqual({ response: 'tail' });
    expect(spread.attributes).toEqual({ custom: { custom: 'retained' } });
    expect(Object.getOwnPropertyDescriptor(spread, 'inputs')?.value).toBe(node.inputs);
    expect(Object.keys(node)).toEqual(Object.keys(spread));
  });

  it('still searches full payload values and returns the selected node data', () => {
    const node = makeNode();
    const result = searchTree(node, 'needle', TEST_SPAN_FILTER_STATE);
    expect(result.matches).toHaveLength(1);
    expect(result.matches[0]).toMatchObject({ section: 'inputs', key: 'request', isKeyMatch: false });
    expect(result.filteredTreeNodes[0].inputs).toBe(node.inputs);
  });
});
