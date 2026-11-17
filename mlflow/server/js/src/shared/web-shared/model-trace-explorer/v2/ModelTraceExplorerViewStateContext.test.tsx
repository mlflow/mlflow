import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { memo } from 'react';
import { fireEvent, screen } from '@testing-library/react';
import { render } from '@databricks/web-shared/test-utils/render';

import { MOCK_TRACE } from '../ModelTraceExplorer.test-utils';
import {
  ModelTraceExplorerViewStateProvider,
  useModelTraceExplorerViewState,
  useModelTraceExplorerViewControls,
  useModelTraceExplorerSpanRowState,
} from './ModelTraceExplorerViewStateContext';
import { ModelTraceExplorerPreferencesProvider } from './ModelTraceExplorerPreferencesContext';
import type { ModelTrace } from './ModelTrace.types';

const controlsRender = jest.fn();
const rowRender = jest.fn();
const Controls = memo(function Controls() {
  controlsRender();
  const { activeTab, assessmentsPaneExpanded, setAssessmentsPaneExpanded } = useModelTraceExplorerViewControls();
  return (
    <>
      <output aria-label="Current tab">{activeTab}</output>
      <button onClick={() => setAssessmentsPaneExpanded(!assessmentsPaneExpanded)}>Toggle assessments</button>
    </>
  );
});
const RowActions = memo(function RowActions() {
  rowRender();
  const { setActiveTab } = useModelTraceExplorerSpanRowState();
  return <button onClick={() => setActiveTab('attributes')}>Row attributes</button>;
});
const Selection = () => {
  const { selectedNode, nodeMap, setSelectedNode } = useModelTraceExplorerViewState();
  return (
    <>
      <output aria-label="Selected span">{selectedNode?.title}</output>
      <button onClick={() => setSelectedNode(Object.values(nodeMap).find((node) => node.key !== selectedNode?.key))}>
        Select another span
      </button>
    </>
  );
};
const Wrapper = ({ modelTrace }: { modelTrace: ModelTrace }) => (
  <ModelTraceExplorerPreferencesProvider>
    <ModelTraceExplorerViewStateProvider modelTrace={modelTrace} assessmentsPaneEnabled>
      <Controls />
      <RowActions />
      <Selection />
    </ModelTraceExplorerViewStateProvider>
  </ModelTraceExplorerPreferencesProvider>
);

describe('trace view subscriptions', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });
  it('keeps controls and row actions stable while trace content changes', () => {
    const { rerender } = render(<Wrapper modelTrace={MOCK_TRACE} />);
    const counts = [controlsRender.mock.calls.length, rowRender.mock.calls.length];
    const updated = {
      ...MOCK_TRACE,
      data: {
        ...MOCK_TRACE.data,
        spans: MOCK_TRACE.data.spans.map((span, index) => (index === 0 ? { ...span, name: 'Updated root' } : span)),
      },
    };
    rerender(<Wrapper modelTrace={updated} />);
    expect(screen.getByLabelText('Selected span')).toHaveTextContent('Updated root');
    expect([controlsRender.mock.calls.length, rowRender.mock.calls.length]).toEqual(counts);
  });

  it('keeps row actions stable through selection, tab and assessment changes', () => {
    render(<Wrapper modelTrace={MOCK_TRACE} />);
    const counts = [controlsRender.mock.calls.length, rowRender.mock.calls.length];
    const initialSelection = screen.getByLabelText('Selected span').textContent;
    fireEvent.click(screen.getByRole('button', { name: 'Select another span' }));
    expect(screen.getByLabelText('Selected span').textContent).not.toBe(initialSelection);
    expect([controlsRender.mock.calls.length, rowRender.mock.calls.length]).toEqual(counts);
    fireEvent.click(screen.getByRole('button', { name: 'Row attributes' }));
    expect(screen.getByLabelText('Current tab')).toHaveTextContent('attributes');
    expect(controlsRender.mock.calls.length).toBe(counts[0] + 1);
    fireEvent.click(screen.getByRole('button', { name: 'Toggle assessments' }));
    expect(rowRender.mock.calls.length).toBe(counts[1]);
  });
});
