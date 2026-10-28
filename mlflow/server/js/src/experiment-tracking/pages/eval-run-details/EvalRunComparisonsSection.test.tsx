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
import { EvalRunComparisonsSection } from './EvalRunComparisonsSection';

interface RunInfo {
  runUuid: string | undefined;
  runName: string | undefined;
  startTime?: number;
}

let mockRunInfos: RunInfo[] = [];
let mockRequestError: Error | null = null;

jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  useGenAiExperimentRunsForComparison: () => ({
    runInfos: mockRunInfos,
    isLoading: false,
    requestError: mockRequestError,
  }),
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

const renderComparisonsSection = async () => {
  render(<div />, {
    wrapper: () => (
      <IntlProvider locale="en" timeZone="UTC">
        <DesignSystemProvider>
          <TestRouter
            history={history}
            routes={[
              testRoute(
                <>
                  <LocationSpy />
                  <EvalRunComparisonsSection
                    experimentId="experiment-1"
                    runUuid="current-run"
                    runName="Baseline model"
                  />
                </>,
                '*',
              ),
            ]}
            initialEntries={['/experiments/experiment-1/evaluation-runs/current-run/details?keep=value']}
          />
        </DesignSystemProvider>
      </IntlProvider>
    ),
  });
  await waitForRoutesToBeRendered();
};

describe('EvalRunComparisonsSection', () => {
  beforeEach(() => {
    mockRequestError = null;
    mockRunInfos = [
      {
        runUuid: 'current-run',
        runName: 'Fetched current run name',
        startTime: new Date('2025-01-15T12:00:00.000Z').getTime(),
      },
      {
        runUuid: 'candidate-1',
        runName: 'Candidate one',
        startTime: new Date('2025-01-16T12:00:00.000Z').getTime(),
      },
      {
        runUuid: 'candidate-2',
        runName: 'Candidate two',
        startTime: new Date('2025-02-17T12:00:00.000Z').getTime(),
      },
    ];
  });

  test('shows run names and creation dates, then enters comparison only after explicit action', async () => {
    const user = userEvent.setup();
    await renderComparisonsSection();

    const comparisonsSection = screen.getByRole('region', { name: 'Comparisons' });
    expect(within(comparisonsSection).queryByText('Comparisons')).not.toBeInTheDocument();
    expect(within(comparisonsSection).getByText('Baseline model')).toBeInTheDocument();
    expect(screen.getByText(/Created Jan 15, 2025/)).toBeInTheDocument();

    const compareButton = screen.getByRole('button', { name: 'Compare' });
    expect(compareButton).toBeDisabled();

    await user.click(screen.getByRole('button', { name: 'Select runs to compare' }));

    expect(screen.queryByRole('option', { name: /Fetched current run name/ })).not.toBeInTheDocument();
    const candidateOne = await screen.findByRole('option', { name: /Candidate one/ });
    const candidateTwo = await screen.findByRole('option', { name: /Candidate two/ });
    expect(candidateOne).toHaveTextContent(/Created Jan 16, 2025/);
    expect(candidateTwo).toHaveTextContent(/Created Feb 17, 2025/);

    await user.click(candidateOne);
    await user.click(candidateTwo);
    expect(screen.getByRole('button', { name: '2 runs selected' })).toBeInTheDocument();
    expect(compareButton).toBeEnabled();

    await user.click(compareButton);

    await waitFor(() => {
      const params = new URLSearchParams(lastLocation.search);
      expect(params.get('compare')).toBe('true');
      expect(params.get('compareRuns')).toBe('candidate-1,candidate-2');
      expect(params.get('keep')).toBe('value');
    });
  });

  test('explains when comparison runs could not be loaded', async () => {
    mockRunInfos = [];
    mockRequestError = new Error('Run search failed');
    await renderComparisonsSection();

    expect(screen.getByRole('status')).toHaveTextContent('Could not load runs for comparison.');
    expect(screen.getByRole('button', { name: 'Compare' })).toBeDisabled();
  });
});
