import { createRef, useEffect, useState } from 'react';
import {
  ChartsTraceHighlightSource,
  RunsChartsSetHighlightContextProvider,
  useRunsChartTraceHighlight,
} from './useRunsChartTraceHighlight';
import { useRunsHighlightTableRow } from './useRunsHighlightTableRow';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const TEST_TRACES = [{ uuid: 'uuid1' }, { uuid: 'uuid2' }, { uuid: 'uuid3' }];

// Force enable tested feature
jest.mock('../../../../common/utils/FeatureUtils', () => ({
  shouldEnableUnifiedChartDataTraceHighlight: jest.fn(() => true),
}));

describe('useRunsChartTraceHighlight', () => {
  const renderTestComponent = () => {
    // A simulated chart with some traces and event handlers
    const TestChart = ({ chartName, onSelect }: { chartName: string; onSelect: (uuid: string) => void }) => {
      const { highlightDataTrace, onHighlightChange } = useRunsChartTraceHighlight();
      const [highlightedTraceUuid, setHighlightedTraceUuid] = useState<string | null>(null);

      useEffect(() => onHighlightChange(setHighlightedTraceUuid), [onHighlightChange]);

      return (
        <div>
          {TEST_TRACES.map((trace) => {
            return (
              <div
                key={trace.uuid}
                className={trace.uuid === highlightedTraceUuid ? 'is-trace-highlighted' : ''}
                onMouseEnter={() =>
                  highlightDataTrace(trace.uuid, {
                    source: ChartsTraceHighlightSource.CHART,
                  })
                }
                onMouseLeave={() => highlightDataTrace(null)}
                onClick={() => onSelect(trace.uuid)}
              >
                {chartName}, trace #{trace.uuid}
              </div>
            );
          })}
        </div>
      );
    };

    // A simulated table with some rows corresponding to traces
    const TestTable = () => {
      const divRef = createRef<HTMLDivElement>();
      const { cellMouseOutHandler, cellMouseOverHandler } = useRunsHighlightTableRow(divRef, 'is-row-highlighted');
      return (
        <div ref={divRef}>
          {TEST_TRACES.map(({ uuid }) => {
            return (
              <div
                onMouseEnter={() => cellMouseOverHandler({ data: { runUuid: uuid } } as any)}
                onMouseLeave={() => cellMouseOutHandler()}
                key={uuid}
                className="ag-row"
                // eslint-disable-next-line react/no-unknown-property
                row-id={uuid}
              >
                Row #{uuid}
              </div>
            );
          })}
          <div>Some other table area</div>
        </div>
      );
    };

    // A test component containing two charts and a table
    const TestComponent = () => {
      const [selectedRunUuid, setSelectedRunUuid] = useState<string | null>(null);
      const { highlightDataTrace } = useRunsChartTraceHighlight();
      useEffect(() => {
        highlightDataTrace(selectedRunUuid, { shouldBlock: Boolean(selectedRunUuid) });
      }, [highlightDataTrace, selectedRunUuid]);
      return (
        <>
          <TestChart chartName="Chart alpha" onSelect={setSelectedRunUuid} />
          <TestChart chartName="Chart beta" onSelect={setSelectedRunUuid} />
          <TestTable />
          <button onClick={() => setSelectedRunUuid(null)}>Clear selection</button>
        </>
      );
    };

    render(
      <RunsChartsSetHighlightContextProvider>
        <TestComponent />
      </RunsChartsSetHighlightContextProvider>,
    );
  };

  test('highlight traces in charts by hovering over table', async () => {
    renderTestComponent();

    // Confirm there's no traces highlighted at all
    expect(document.querySelector('.is-trace-highlighted')).not.toBeInTheDocument();
    expect(document.querySelector('.is-row-highlighted')).not.toBeInTheDocument();

    // Hover over uuid1 table row
    await userEvent.hover(screen.getByText('Row #uuid1'));

    // Check that the corresponding traces are highlighted
    expect(screen.getByText('Chart alpha, trace #uuid1')).toHaveClass('is-trace-highlighted');
    expect(screen.getByText('Chart alpha, trace #uuid2')).not.toHaveClass('is-trace-highlighted');

    expect(screen.getByText('Chart beta, trace #uuid1')).toHaveClass('is-trace-highlighted');
    expect(screen.getByText('Chart beta, trace #uuid2')).not.toHaveClass('is-trace-highlighted');

    // Table should not highlight itself
    expect(document.querySelector('.is-row-highlighted')).not.toBeInTheDocument();

    // We have to explicitly unhover due to JSDOM limitations
    await userEvent.unhover(screen.getByText('Row #uuid1'));

    // Hover over uuid2 table row
    await userEvent.hover(screen.getByText('Row #uuid2'));

    // Check that the corresponding traces are highlighted
    expect(screen.getByText('Chart alpha, trace #uuid1')).not.toHaveClass('is-trace-highlighted');
    expect(screen.getByText('Chart alpha, trace #uuid2')).toHaveClass('is-trace-highlighted');

    // Move the hover to some other table area
    await userEvent.unhover(screen.getByText('Row #uuid2'));
    await userEvent.hover(screen.getByText('Some other table area'));

    // Confirm there's no traces highlighted at all
    expect(document.querySelector('.is-trace-highlighted')).not.toBeInTheDocument();
    expect(document.querySelector('.is-row-highlighted')).not.toBeInTheDocument();
  });

  test('highlight traces in table and other charts by hovering over data traces', async () => {
    renderTestComponent();

    // Confirm there's no traces highlighted at all
    expect(document.querySelector('.is-trace-highlighted')).not.toBeInTheDocument();
    expect(document.querySelector('.is-row-highlighted')).not.toBeInTheDocument();

    // Hover over trace #uuid1 in Chart alpha
    await userEvent.hover(screen.getByText('Chart alpha, trace #uuid1'));

    // Check that the corresponding traces are highlighted
    expect(document.querySelector(`[row-id="uuid1"].is-row-highlighted`)).toBeInTheDocument();
    expect(document.querySelector(`[row-id="uuid2"].is-row-highlighted`)).not.toBeInTheDocument();
    expect(screen.getByText('Chart alpha, trace #uuid1')).toHaveClass('is-trace-highlighted');
    expect(screen.getByText('Chart alpha, trace #uuid2')).not.toHaveClass('is-trace-highlighted');
    expect(screen.getByText('Chart beta, trace #uuid1')).toHaveClass('is-trace-highlighted');
    expect(screen.getByText('Chart beta, trace #uuid2')).not.toHaveClass('is-trace-highlighted');

    // Hover over trace #uuid2 in Chart beta
    await userEvent.unhover(screen.getByText('Chart alpha, trace #uuid1'));
    await userEvent.hover(screen.getByText('Chart beta, trace #uuid2'));

    // Check that the corresponding traces are highlighted
    expect(document.querySelector(`[row-id="uuid1"].is-row-highlighted`)).not.toBeInTheDocument();
    expect(document.querySelector(`[row-id="uuid2"].is-row-highlighted`)).toBeInTheDocument();
    expect(screen.getByText('Chart alpha, trace #uuid1')).not.toHaveClass('is-trace-highlighted');
    expect(screen.getByText('Chart alpha, trace #uuid2')).toHaveClass('is-trace-highlighted');
    expect(screen.getByText('Chart beta, trace #uuid1')).not.toHaveClass('is-trace-highlighted');
    expect(screen.getByText('Chart beta, trace #uuid2')).toHaveClass('is-trace-highlighted');
  });

  test('should block highlight when necessary', async () => {
    renderTestComponent();

    // Confirm there's no traces highlighted at all
    expect(document.querySelector('.is-trace-highlighted')).not.toBeInTheDocument();
    expect(document.querySelector('.is-row-highlighted')).not.toBeInTheDocument();

    // Hover over trace #uuid1 in Chart alpha
    await userEvent.hover(screen.getByText('Chart alpha, trace #uuid1'));

    // Check that the corresponding traces are highlighted
    expect(document.querySelector(`[row-id="uuid1"].is-row-highlighted`)).toBeInTheDocument();
    expect(document.querySelector(`[row-id="uuid2"].is-row-highlighted`)).not.toBeInTheDocument();
    expect(screen.getByText('Chart alpha, trace #uuid1')).toHaveClass('is-trace-highlighted');
    expect(screen.getByText('Chart beta, trace #uuid1')).toHaveClass('is-trace-highlighted');

    // "Select" trace uuid1 in the chart alpha, which should block the highlight
    await userEvent.click(screen.getByText('Chart alpha, trace #uuid1'));

    // Attempt to hover over trace #uuid2 in Chart beta
    await userEvent.hover(screen.getByText('Chart alpha, trace #uuid2'));

    // Confirm that trace uuid2 is NOT highlighted
    expect(document.querySelector(`[row-id="uuid1"].is-row-highlighted`)).toBeInTheDocument();
    expect(document.querySelector(`[row-id="uuid2"].is-row-highlighted`)).not.toBeInTheDocument();
    expect(screen.getByText('Chart alpha, trace #uuid1')).toHaveClass('is-trace-highlighted');
    expect(screen.getByText('Chart beta, trace #uuid1')).toHaveClass('is-trace-highlighted');

    // "Unselect" trace uuid1 in the chart alpha
    await userEvent.click(screen.getByText('Clear selection'));

    // Again, attempt to hover over trace #uuid2 in Chart beta
    await userEvent.hover(screen.getByText('Chart alpha, trace #uuid2'));

    // Confirm that trace uuid2 is highlighted now
    expect(document.querySelector(`[row-id="uuid1"].is-row-highlighted`)).not.toBeInTheDocument();
    expect(document.querySelector(`[row-id="uuid2"].is-row-highlighted`)).toBeInTheDocument();
    expect(screen.getByText('Chart alpha, trace #uuid1')).not.toHaveClass('is-trace-highlighted');
    expect(screen.getByText('Chart beta, trace #uuid1')).not.toHaveClass('is-trace-highlighted');
    expect(screen.getByText('Chart alpha, trace #uuid2')).toHaveClass('is-trace-highlighted');
    expect(screen.getByText('Chart beta, trace #uuid2')).toHaveClass('is-trace-highlighted');
  });
});
