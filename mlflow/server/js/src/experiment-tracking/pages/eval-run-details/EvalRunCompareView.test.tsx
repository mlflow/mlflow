import { describe, expect, jest, test } from '@jest/globals';
import { act, render, screen, waitFor } from '@databricks/web-shared/test-utils/render';
import userEvent from '@testing-library/user-event';
import { useEffect } from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { TestRouter, testRoute } from '../../../common/utils/RoutingTestUtils';
import { useLocation } from '../../../common/utils/RoutingUtils';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';
import { EvalRunCompareView } from './EvalRunCompareView';
import type { EvalRunComparisonTraceFilter } from './EvalRunComparisonDelta';

// The compare view only mounts with the new eval runs UI (EvalRunDetailsPage), so tests default to the flag on.
jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => true),
}));
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
// The Details table publishes its "start pairwise review" action; this records each call to it.
const mockTableReviewAction = jest.fn();
jest.mock('./EvalRunComparisonTracesV4Table', () => {
  const { useEffect: useMockEffect } = jest.requireActual<typeof import('react')>('react');
  return {
    EvalRunComparisonTracesV4Table: ({
      runUuids,
      initialOutcomeFilter,
      onPairwiseReviewActionChange,
    }: {
      runUuids: string[];
      initialOutcomeFilter?: EvalRunComparisonTraceFilter;
      onPairwiseReviewActionChange?: (action: (() => void) | undefined) => void;
    }) => {
      useMockEffect(() => {
        onPairwiseReviewActionChange?.(mockTableReviewAction);
        return () => onPairwiseReviewActionChange?.(undefined);
      }, [onPairwiseReviewActionChange]);
      return (
        <div data-testid="details">
          runs:{runUuids.join(',')} filter:{initialOutcomeFilter ? JSON.stringify(initialOutcomeFilter) : 'none'}
        </div>
      );
    },
  };
});

let lastSearch = '';
const LocationSpy = () => {
  const search = useLocation().search;
  useEffect(() => {
    lastSearch = search;
  }, [search]);
  return null;
};

const renderView = (
  compareRunUuids = ['run-2', 'run-3'],
  {
    search = '',
    onPairwiseReviewActionChange,
  }: { search?: string; onPairwiseReviewActionChange?: (action: (() => void) | undefined) => void } = {},
) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <TestRouter
          initialEntries={[`/compare${search}`]}
          routes={[
            testRoute(
              <>
                <LocationSpy />
                <EvalRunCompareView
                  experimentId="exp-1"
                  baselineRunUuid="run-1"
                  compareRunUuids={compareRunUuids}
                  onPairwiseReviewActionChange={onPairwiseReviewActionChange}
                />
              </>,
              '/compare',
            ),
          ]}
        />
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

  test('opens on the Details tab for a link to a trace comparison', () => {
    renderView(undefined, { search: '?baselineTraceId=tr-1&candidateTraceId=tr-2' });
    expect(screen.getByRole('tab', { name: 'Details' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByTestId('details')).toBeInTheDocument();
  });

  test('starts pairwise review with R from the Overview, on the Details tab', async () => {
    mockTableReviewAction.mockClear();
    renderView();
    expect(screen.getByRole('tab', { name: 'Overview' })).toHaveAttribute('aria-selected', 'true');

    await userEvent.keyboard('r');

    expect(screen.getByRole('tab', { name: 'Details' })).toHaveAttribute('aria-selected', 'true');
    await waitFor(() => expect(mockTableReviewAction).toHaveBeenCalledTimes(1));
    await userEvent.keyboard('R');
    expect(mockTableReviewAction).toHaveBeenCalledTimes(2);
  });

  test('leaves pairwise review with R while reviewing', async () => {
    mockTableReviewAction.mockClear();
    renderView(undefined, { search: '?baselineTraceId=tr-1&candidateTraceId=tr-2&pairwiseReview=true' });
    await userEvent.keyboard('r');
    await waitFor(() => expect(new URLSearchParams(lastSearch).get('pairwiseReview')).toBeNull());
    expect(new URLSearchParams(lastSearch).get('baselineTraceId')).toBe('tr-1');
    expect(mockTableReviewAction).not.toHaveBeenCalled();
  });

  test('ignores R with a modifier or while typing in a field', async () => {
    mockTableReviewAction.mockClear();
    renderView();
    await userEvent.keyboard('{Meta>}r{/Meta}');
    await userEvent.keyboard('{Control>}r{/Control}');
    await userEvent.keyboard('{Shift>}r{/Shift}');
    const input = document.createElement('input');
    document.body.appendChild(input);
    input.focus();
    await userEvent.keyboard('r');
    input.remove();
    expect(screen.getByRole('tab', { name: 'Overview' })).toHaveAttribute('aria-selected', 'true');
    expect(mockTableReviewAction).not.toHaveBeenCalled();
  });

  test('publishes a start-review action only while runs are compared', () => {
    const onPairwiseReviewActionChange = jest.fn();
    const { unmount } = renderView(undefined, { onPairwiseReviewActionChange });
    expect(onPairwiseReviewActionChange).toHaveBeenLastCalledWith(expect.any(Function));
    unmount();
    expect(onPairwiseReviewActionChange).toHaveBeenLastCalledWith(undefined);

    onPairwiseReviewActionChange.mockClear();
    renderView([], { onPairwiseReviewActionChange });
    expect(onPairwiseReviewActionChange).toHaveBeenLastCalledWith(undefined);
  });

  test('starts review from the published action', async () => {
    mockTableReviewAction.mockClear();
    const onPairwiseReviewActionChange = jest.fn<(action: (() => void) | undefined) => void>();
    renderView(undefined, { onPairwiseReviewActionChange });
    const action = onPairwiseReviewActionChange.mock.calls.at(-1)?.[0];
    act(() => action?.());
    expect(screen.getByRole('tab', { name: 'Details' })).toHaveAttribute('aria-selected', 'true');
    await waitFor(() => expect(mockTableReviewAction).toHaveBeenCalledTimes(1));
  });

  test('binds no R shortcut and publishes no review action with the flag off', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    try {
      mockTableReviewAction.mockClear();
      const onPairwiseReviewActionChange = jest.fn();
      renderView(undefined, { onPairwiseReviewActionChange });
      await userEvent.keyboard('r');
      expect(screen.getByRole('tab', { name: 'Overview' })).toHaveAttribute('aria-selected', 'true');
      expect(onPairwiseReviewActionChange).not.toHaveBeenCalledWith(expect.any(Function));
    } finally {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    }
  });
});
