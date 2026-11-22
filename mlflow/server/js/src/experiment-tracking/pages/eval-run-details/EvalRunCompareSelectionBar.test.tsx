import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { render, screen, waitFor, within } from '@databricks/web-shared/test-utils/render';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { useEffect } from 'react';
import { useLocation } from '../../../common/utils/RoutingUtils';
import {
  setupTestRouter,
  testRoute,
  TestRouter,
  waitForRoutesToBeRendered,
} from '../../../common/utils/RoutingTestUtils';
import { EvalRunCompareSelectionBar } from './EvalRunCompareSelectionBar';
import { useEvalRunComparisonBaseline } from './useEvalRunComparisonBaseline';

interface RunInfo {
  runUuid: string | undefined;
  runName: string | undefined;
  startTime?: number;
}

let mockRunInfos: RunInfo[] = [];

jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  useGenAiExperimentRunsForComparison: () => ({ runInfos: mockRunInfos }),
}));

jest.mock('../../components/experiment-page/hooks/useExperimentRunColor', () => ({
  useGetExperimentRunColor: () => () => '#123456',
}));

const { history } = setupTestRouter();

let lastLocation = { pathname: '', search: '' };
const LocationSpy = () => {
  const location = useLocation();
  useEffect(() => {
    lastLocation = { pathname: location.pathname, search: location.search };
  }, [location.pathname, location.search]);
  return null;
};

const SelectionBarWithNavigation = ({
  baselineRunUuid,
  compareRunUuids,
}: {
  baselineRunUuid: string;
  compareRunUuids: string[];
}) => {
  const onSelectBaseline = useEvalRunComparisonBaseline('experiment-1', baselineRunUuid, compareRunUuids);
  return (
    <EvalRunCompareSelectionBar
      experimentId="experiment-1"
      baselineRunUuid={baselineRunUuid}
      compareRunUuids={compareRunUuids}
      onSelectBaseline={onSelectBaseline}
    />
  );
};

const renderSelectionBar = async (baselineRunUuid: string, compareRunUuids: string[]) => {
  render(<div />, {
    enableEmotionCSS: true,
    wrapper: () => (
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <TestRouter
            history={history}
            routes={[
              testRoute(
                <>
                  <LocationSpy />
                  <SelectionBarWithNavigation baselineRunUuid={baselineRunUuid} compareRunUuids={compareRunUuids} />
                </>,
                '*',
              ),
            ]}
            initialEntries={[
              `/experiments/experiment-1/evaluation-runs/baseline/details?compare=true&compareRuns=${compareRunUuids.join(
                ',',
              )}&o=workspace-id`,
            ]}
          />
        </DesignSystemProvider>
      </IntlProvider>
    ),
  });
  await waitForRoutesToBeRendered();
};

describe('EvalRunCompareSelectionBar', () => {
  beforeEach(() => {
    mockRunInfos = [];
  });

  test('keeps a missing baseline run selected in its dropdown', async () => {
    mockRunInfos = [
      { runUuid: 'comparison-run', runName: 'Comparison run' },
      { runUuid: 'available-run', runName: 'Available run' },
    ];
    await renderSelectionBar('missing-baseline', ['comparison-run']);

    await userEvent.setup().click(await screen.findByRole('button', { name: /missing-baseline/ }));

    expect(await screen.findByRole('option', { name: /missing-baseline/ })).toHaveAttribute('aria-selected', 'true');
  });

  test('keeps a missing comparison run selected and offers fetched replacement runs', async () => {
    mockRunInfos = [
      { runUuid: 'baseline-run', runName: 'Baseline run' },
      { runUuid: 'available-run', runName: 'Available run' },
    ];
    await renderSelectionBar('baseline-run', ['missing-comparison']);

    await userEvent.setup().click(
      await screen.findByRole('button', {
        name: /missing-comparison/,
        expanded: false,
      }),
    );

    expect(await screen.findByRole('option', { name: /missing-comparison/ })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('option', { name: /Available run/ })).toHaveAttribute('aria-selected', 'false');
    expect(
      within(screen.getByRole('option', { name: /Available run/ })).getByText('Created Date unavailable'),
    ).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /Baseline run/ })).not.toBeInTheDocument();
  });

  test('uses one multiselect with run names and creation dates', async () => {
    mockRunInfos = [
      {
        runUuid: 'baseline-run',
        runName: 'Baseline run',
        startTime: Date.UTC(2024, 0, 1, 12),
      },
      {
        runUuid: 'comparison-one',
        runName: 'First comparison',
        startTime: Date.UTC(2024, 1, 2, 13),
      },
      {
        runUuid: 'comparison-two',
        runName: 'Second comparison',
        startTime: Date.UTC(2024, 2, 3, 14),
      },
      {
        runUuid: 'available-run',
        runName: 'Available run',
        startTime: Date.UTC(2024, 3, 4, 15),
      },
    ];
    await renderSelectionBar('baseline-run', ['comparison-one', 'comparison-two']);
    const user = userEvent.setup();

    const baselineButton = screen.getByRole('button', {
      name: /Base.*Baseline run/,
    });
    expect(within(baselineButton).getByText('Base')).toBeInTheDocument();
    expect(screen.getAllByRole('button')).toHaveLength(2);
    const comparisonButton = screen.getByRole('button', {
      name: /First comparison, Second comparison/,
    });
    expect(comparisonButton).toBeInTheDocument();
    expect(within(comparisonButton).getByTitle('Second comparison')).toHaveStyle({ flex: '1 1 auto' });

    await user.click(
      screen.getByRole('button', {
        name: /First comparison, Second comparison/,
      }),
    );

    expect(screen.getByRole('option', { name: /First comparison.*Created/ })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('option', { name: /Second comparison.*Created/ })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('option', { name: /Available run.*Created/ })).toHaveAttribute('aria-selected', 'false');
  });

  test('keeps a long final selected run name in an ellipsis-capable slot', async () => {
    const longRunName = 'second-comparison-run-with-a-name-that-needs-truncation';
    mockRunInfos = [
      { runUuid: 'baseline-run', runName: 'Baseline run' },
      { runUuid: 'comparison-one', runName: 'First comparison' },
      { runUuid: 'comparison-two', runName: longRunName },
    ];
    await renderSelectionBar('baseline-run', ['comparison-one', 'comparison-two']);

    const comparisonButton = screen.getByRole('button', { name: /First comparison/ });
    expect(within(comparisonButton).getByTitle(longRunName)).toHaveStyle({ flex: '1 1 auto' });
  });

  test('summarizes runs beyond the visible selector limit', async () => {
    mockRunInfos = [
      { runUuid: 'baseline-run', runName: 'Baseline run' },
      { runUuid: 'comparison-one', runName: 'First comparison' },
      { runUuid: 'comparison-two', runName: 'Second comparison' },
      { runUuid: 'comparison-three', runName: 'Third comparison' },
      { runUuid: 'comparison-four', runName: 'Fourth comparison' },
      { runUuid: 'comparison-five', runName: 'Fifth comparison' },
      { runUuid: 'comparison-six', runName: 'Sixth comparison' },
    ];
    await renderSelectionBar('baseline-run', [
      'comparison-one',
      'comparison-two',
      'comparison-three',
      'comparison-four',
      'comparison-five',
      'comparison-six',
    ]);

    const selector = screen.getByRole('button', { name: /First comparison, Second comparison, Third comparison/ });
    expect(selector).toHaveTextContent('Fifth comparison');
    expect(selector).toHaveTextContent('+1 more');
    expect(selector).not.toHaveTextContent('Sixth comparison');
  });

  test('updates comparison selections while preserving unrelated query parameters', async () => {
    mockRunInfos = [
      { runUuid: 'baseline-run', runName: 'Baseline run' },
      { runUuid: 'comparison-run', runName: 'Comparison run' },
      { runUuid: 'available-run', runName: 'Available run' },
    ];
    await renderSelectionBar('baseline-run', ['comparison-run']);
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: /Comparison run/ }));
    await user.click(screen.getByRole('option', { name: /Available run/ }));

    await waitFor(() => {
      expect(new URLSearchParams(lastLocation.search).get('compareRuns')).toBe('comparison-run,available-run');
      expect(new URLSearchParams(lastLocation.search).get('o')).toBe('workspace-id');
    });
  });

  test('limits baseline choices to selected runs and preserves selection count when re-anchoring', async () => {
    mockRunInfos = [
      { runUuid: 'baseline-run', runName: 'Named baseline' },
      { runUuid: 'comparison-run', runName: 'Named comparison' },
      { runUuid: 'available-run', runName: 'Available run' },
    ];
    await renderSelectionBar('baseline-run', ['comparison-run']);
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: /Named baseline/ }));

    expect(await screen.findAllByRole('option')).toHaveLength(2);
    expect(screen.getByRole('option', { name: /Named baseline/ })).toHaveAttribute('aria-selected', 'true');
    expect(screen.queryByRole('option', { name: /Available run/ })).not.toBeInTheDocument();

    await user.click(screen.getByRole('option', { name: /Named comparison/ }));

    await waitFor(() => {
      expect(lastLocation.pathname).toContain('/evaluation-runs/comparison-run/details');
      expect(new URLSearchParams(lastLocation.search).get('compareRuns')).toBe('baseline-run');
      expect(new URLSearchParams(lastLocation.search).get('o')).toBe('workspace-id');
    });
  });

  test('removes a comparison run by unchecking it', async () => {
    mockRunInfos = [
      { runUuid: 'baseline-run', runName: 'Baseline run' },
      { runUuid: 'comparison-run', runName: 'Comparison run' },
    ];
    await renderSelectionBar('baseline-run', ['comparison-run']);
    const user = userEvent.setup();

    await user.click(await screen.findByRole('button', { name: /Comparison run/ }));
    await user.click(screen.getByRole('option', { name: /Comparison run/ }));

    await waitFor(() => {
      expect(new URLSearchParams(lastLocation.search).has('compareRuns')).toBe(false);
    });
  });
});
