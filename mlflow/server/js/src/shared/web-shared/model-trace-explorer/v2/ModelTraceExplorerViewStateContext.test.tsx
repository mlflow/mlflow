import { describe, expect, it } from '@jest/globals';
import { fireEvent, render, screen } from '@testing-library/react';

import type { ModelTrace, ModelTraceComparisonRole, ModelTraceSpanV2 } from './ModelTrace.types';
import {
  ModelTraceExplorerViewStateProvider,
  useModelTraceExplorerViewState,
} from './ModelTraceExplorerViewStateContext';
import { MOCK_TRACE } from '../ModelTraceExplorer.test-utils';

const span = (id: string, parent: string | null = null): ModelTraceSpanV2 => ({
  context: { span_id: id, trace_id: 'selection-trace' },
  name: id,
  parent_id: parent,
  start_time: 1000000,
  end_time: 2000000,
  attributes: { 'mlflow.spanInputs': JSON.stringify({ value: id }) },
});
const root = span('root');
const parent = span('parent', 'root');
const leaf = span('leaf', 'parent');
const trace = (spans: ModelTraceSpanV2[]): ModelTrace => ({ ...MOCK_TRACE, data: { spans } });

function Selection() {
  const { selectedNode, setSelectedNode, nodeMap } = useModelTraceExplorerViewState();
  return (
    <>
      <output aria-label="selected span">{selectedNode?.key ?? 'none'}</output>
      <output aria-label="selected title">{selectedNode?.title}</output>
      {Object.entries(nodeMap).map(([id, node]) => (
        <button key={id} onClick={() => setSelectedNode(node)}>
          select {id}
        </button>
      ))}
    </>
  );
}

const explorer = (spans: ModelTraceSpanV2[], linkedSpan = 'leaf') => (
  <ModelTraceExplorerViewStateProvider
    modelTrace={trace(spans)}
    selectedSpanIdOnRender={linkedSpan}
    assessmentsPaneEnabled={false}
  >
    <Selection />
  </ModelTraceExplorerViewStateProvider>
);

describe('trace span selection across content updates', () => {
  it('opens an available deep-linked span', () => {
    render(explorer([root, parent, leaf]));
    expect(screen.getByLabelText('selected span')).toHaveTextContent('leaf');
  });

  it.each(['toString', 'constructor', '__proto__'])('ignores a missing inherited key: %s', (id) => {
    render(explorer([root, parent], id));
    expect(screen.getByLabelText('selected span')).toHaveTextContent('root');
  });

  it('adopts a deep-linked span when it arrives in a later trace update', () => {
    const { rerender } = render(explorer([root, parent]));
    expect(screen.getByLabelText('selected span')).toHaveTextContent('root');
    rerender(explorer([root, parent, leaf]));
    expect(screen.getByLabelText('selected span')).toHaveTextContent('leaf');
  });

  it('preserves a user selection after the deep link has been adopted', () => {
    const { rerender } = render(explorer([root, parent]));
    rerender(explorer([root, parent, leaf]));
    expect(screen.getByLabelText('selected span')).toHaveTextContent('leaf');
    fireEvent.click(screen.getByRole('button', { name: 'select parent' }));
    rerender(explorer([root, { ...parent, name: 'updated parent' }, leaf]));
    expect(screen.getByLabelText('selected span')).toHaveTextContent('parent');
    expect(screen.getByLabelText('selected title')).toHaveTextContent('updated parent');
  });

  it('does not re-adopt a linked span after it disappears and returns', () => {
    const { rerender } = render(explorer([root, parent, leaf]));
    rerender(explorer([root, parent]));
    expect(screen.getByLabelText('selected span')).toHaveTextContent('root');
    rerender(explorer([root, parent, leaf]));
    expect(screen.getByLabelText('selected span')).toHaveTextContent('root');
  });

  it('resolves a deep link among multiple top-level spans', () => {
    render(explorer([root, parent, leaf, span('second-root')]));
    expect(screen.getByLabelText('selected span')).toHaveTextContent('leaf');
  });

  it('adopts a new deep link once without overriding subsequent user selection', () => {
    const { rerender } = render(explorer([root, parent, leaf], 'parent'));
    rerender(explorer([root, parent, leaf], 'leaf'));
    expect(screen.getByLabelText('selected span')).toHaveTextContent('leaf');
    fireEvent.click(screen.getByRole('button', { name: 'select root' }));
    rerender(explorer([root, parent, { ...leaf, name: 'updated leaf' }], 'leaf'));
    expect(screen.getByLabelText('selected span')).toHaveTextContent('root');
  });

  it('retains a manual selection while the requested span remains absent', () => {
    const { rerender } = render(explorer([root, parent]));
    fireEvent.click(screen.getByRole('button', { name: 'select parent' }));
    rerender(explorer([root, { ...parent, name: 'updated parent' }]));
    expect(screen.getByLabelText('selected span')).toHaveTextContent('parent');
    expect(screen.getByLabelText('selected title')).toHaveTextContent('updated parent');
  });

  it('adopts a pending link once even after an interim manual selection', () => {
    const { rerender } = render(explorer([root, parent]));
    fireEvent.click(screen.getByRole('button', { name: 'select parent' }));
    rerender(explorer([root, parent, leaf]));
    expect(screen.getByLabelText('selected span')).toHaveTextContent('leaf');
    fireEvent.click(screen.getByRole('button', { name: 'select parent' }));
    rerender(explorer([root, { ...parent, name: 'updated parent' }, leaf]));
    expect(screen.getByLabelText('selected span')).toHaveTextContent('parent');
    expect(screen.getByLabelText('selected title')).toHaveTextContent('updated parent');
  });

  it.each<ModelTraceComparisonRole>(['baseline', 'candidate'])(
    'adopts a late comparison source link as its display node (primary: %s)',
    (comparisonRole) => {
      const comparison = (spans: ModelTraceSpanV2[]) => {
        const baseline = trace(spans);
        const candidate = trace(
          spans.map((node) => ({
            ...node,
            context: { ...node.context, span_id: `candidate-${node.context.span_id}` },
            parent_id: node.parent_id ? `candidate-${node.parent_id}` : null,
          })),
        );
        return (
          <ModelTraceExplorerViewStateProvider
            modelTrace={comparisonRole === 'baseline' ? baseline : candidate}
            comparisonModelTrace={comparisonRole === 'baseline' ? candidate : baseline}
            comparisonRole={comparisonRole}
            selectedSpanIdOnRender="candidate-leaf"
            assessmentsPaneEnabled={false}
          >
            <Selection />
          </ModelTraceExplorerViewStateProvider>
        );
      };
      const { rerender } = render(comparison([root, parent]));
      expect(screen.getByLabelText('selected span')).toHaveTextContent(/^root$/);
      rerender(comparison([root, parent, leaf]));
      expect(screen.getByLabelText('selected span')).toHaveTextContent(/^leaf$/);
      fireEvent.click(screen.getByRole('button', { name: 'select parent' }));
      rerender(comparison([root, parent, leaf]));
      expect(screen.getByLabelText('selected span')).toHaveTextContent(/^parent$/);
    },
  );
});
