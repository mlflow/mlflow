import { describe, expect, jest, test } from '@jest/globals';
import { render, screen } from '@databricks/web-shared/test-utils/render';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { EvalRunCompareView } from './EvalRunCompareView';
import type { EvalRunComparisonTraceFilter } from './EvalRunComparisonDelta';

jest.mock('./useEvalRunComparisonBaseline', () => ({ useEvalRunComparisonBaseline: () => jest.fn() }));
jest.mock('./EvalRunCompareSelectionBar', () => ({
  EvalRunCompareSelectionBar: () => <div data-testid="selection-bar" />,
}));
const OUTCOME_FILTER: EvalRunComparisonTraceFilter = {
  columnId: 'assessment:quality',
  runUuid: 'run-2',
  outcome: 'regression',
};
jest.mock('./EvalRunComparisonOverviewTab', () => ({
  EvalRunComparisonOverviewTab: ({
    experimentId,
    baselineRunUuid,
    compareRunUuids,
    onNavigateToTraceView,
  }: {
    experimentId: string;
    baselineRunUuid: string;
    compareRunUuids: string[];
    onNavigateToTraceView?: (filter?: EvalRunComparisonTraceFilter) => void;
  }) => (
    <div data-testid="overview">
      {experimentId} base:{baselineRunUuid} compare:{compareRunUuids.join(',')} links:
      {onNavigateToTraceView ? 'on' : 'off'}
      {onNavigateToTraceView && (
        <>
          <button onClick={() => onNavigateToTraceView()}>Compare traces row by row</button>
          <button onClick={() => onNavigateToTraceView(OUTCOME_FILTER)}>1 regression</button>
        </>
      )}
    </div>
  ),
}));
jest.mock('./EvalRunComparisonTracesV4Table', () => ({
  EvalRunComparisonTracesV4Table: ({
    runUuids,
    initialOutcomeFilter,
  }: {
    runUuids: string[];
    initialOutcomeFilter?: EvalRunComparisonTraceFilter;
  }) => (
    <div data-testid="details">
      runs:{runUuids.join(',')} filter:{initialOutcomeFilter ? JSON.stringify(initialOutcomeFilter) : 'none'}
    </div>
  ),
}));

const renderView = (compareRunUuids = ['run-2', 'run-3']) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <EvalRunCompareView experimentId="exp-1" baselineRunUuid="run-1" compareRunUuids={compareRunUuids} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('EvalRunCompareView', () => {
  test('shows the comparison overview of the selected runs under the selection bar', () => {
    renderView();
    expect(screen.getByTestId('selection-bar')).toBeInTheDocument();
    expect(screen.getByTestId('overview')).toHaveTextContent('exp-1 base:run-1 compare:run-2,run-3 links:on');
    expect(
      screen.getByTestId('selection-bar').compareDocumentPosition(screen.getByTestId('overview')) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  test('opens on the Overview tab and switches to the Details table of all compared runs', async () => {
    renderView();
    expect(screen.getByRole('tab', { name: 'Overview' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: 'Details' })).toHaveAttribute('aria-selected', 'false');

    await userEvent.click(screen.getByRole('tab', { name: 'Details' }));

    expect(screen.getByRole('tab', { name: 'Details' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByTestId('details')).toHaveTextContent('runs:run-1,run-2,run-3 filter:none');
  });

  test('opens the Details tab from the Overview, with the outcome filter of a clicked count', async () => {
    renderView();
    await userEvent.click(screen.getByRole('button', { name: '1 regression' }));
    expect(screen.getByRole('tab', { name: 'Details' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByTestId('details')).toHaveTextContent(`filter:${JSON.stringify(OUTCOME_FILTER)}`);

    // Going back to the Overview drops the outcome filter.
    await userEvent.click(screen.getByRole('tab', { name: 'Overview' }));
    await userEvent.click(screen.getByRole('tab', { name: 'Details' }));
    expect(screen.getByTestId('details')).toHaveTextContent('filter:none');

    await userEvent.click(screen.getByRole('tab', { name: 'Overview' }));
    await userEvent.click(screen.getByRole('button', { name: 'Compare traces row by row' }));
    expect(screen.getByTestId('details')).toHaveTextContent('filter:none');
  });

  test('offers no Details links or table until a run is compared with the baseline', async () => {
    renderView([]);
    expect(screen.getByTestId('overview')).toHaveTextContent('links:off');
    await userEvent.click(screen.getByRole('tab', { name: 'Details' }));
    expect(screen.queryByTestId('details')).not.toBeInTheDocument();
    expect(screen.getByText('Add a run to compare against the baseline.')).toBeVisible();
  });
});
