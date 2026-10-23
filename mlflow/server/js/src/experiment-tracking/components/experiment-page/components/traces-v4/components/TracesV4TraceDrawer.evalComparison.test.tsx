import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { rest } from 'msw';
import { useEffect } from 'react';
import { setupServer } from '@mlflow/mlflow/src/common/utils/setup-msw';
import { setupTestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import { useLocation } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { TracesV4TraceDrawer, type TracesV4TraceDrawerProps } from './TracesV4TraceDrawer';
import { renderTracesV4Page } from '../test-utils/renderTracesV4Page';
import { makeTaggedTrace } from '../test-utils/mockTraces';

jest.mock('@mlflow/mlflow/src/common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FeatureUtils')>(
    '@mlflow/mlflow/src/common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: jest.fn(() => true),
}));

const EXPERIMENT_ID = 'exp-1';
const PATH = '/ml/experiments/:experimentId/evaluation-runs/:runUuid/details';
const BASE_URL = '/ml/experiments/exp-1/evaluation-runs/run-a/details?compare=true&compareRuns=run-b,run-c';
// The comparison is open on the candidate; both ids are the bare (non-V4) trace ids OSS uses.
const COMPARISON_URL = `${BASE_URL}&baselineTraceId=tr-a&candidateTraceId=tr-b&traceId=tr-b`;

const TRACES = {
  'tr-a': makeTaggedTrace('tr-a', {}),
  'tr-b': makeTaggedTrace('tr-b', {}),
  'tr-c': makeTaggedTrace('tr-c', {}),
};
const RUN_OPTIONS = [
  { runUuid: 'run-a', runName: 'Run A', traceId: 'tr-a' },
  { runUuid: 'run-b', runName: 'Run B', traceId: 'tr-b' },
  { runUuid: 'run-c', runName: 'Run C', traceId: 'tr-c' },
];

const server = setupServer(
  rest.get('/ajax-api/3.0/mlflow/traces/:traceId', (req, res, ctx) => {
    const trace = TRACES[req.params['traceId'] as keyof typeof TRACES];
    return trace ? res(ctx.json({ trace: { trace_info: trace, spans: [] } })) : res(ctx.status(404));
  }),
);

const { history } = setupTestRouter();
let lastSearch = '';
const LocationSpy = () => {
  const search = useLocation().search;
  useEffect(() => {
    lastSearch = search;
  }, [search]);
  return null;
};

beforeAll(() => {
  process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
});
const JSDOM_WIDTH = window.innerWidth;
beforeEach(() => {
  lastSearch = '';
  // The comparison header (and its run pickers) shows only on wide drawers.
  window.innerWidth = 1600;
});
afterEach(() => {
  window.innerWidth = JSDOM_WIDTH;
});
afterAll(() => {
  delete process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'];
});

const renderDrawer = (url: string, props: Partial<TracesV4TraceDrawerProps> = {}) =>
  renderTracesV4Page({
    initialUrl: url,
    routes: [
      {
        path: PATH,
        element: (
          <>
            <LocationSpy />
            <TracesV4TraceDrawer
              traceId="tr-b"
              onClose={() => {}}
              experimentId={EXPERIMENT_ID}
              traces={Object.values(TRACES)}
              onSelectTrace={() => {}}
              {...props}
            />
          </>
        ),
      },
    ],
    history,
    experimentId: EXPERIMENT_ID,
  });

const findDrawer = () => screen.findByRole('dialog', {}, { timeout: 10000 });
const navButton = (name: 'Previous trace' | 'Next trace') =>
  within(screen.getByRole('dialog')).getByRole('button', { name });
const candidateRunTrigger = () =>
  document.querySelector<HTMLElement>('[data-component-id="mlflow.trace-explorer.comparison.candidate-run-trigger"]');

describe('TracesV4TraceDrawer comparison of an eval comparison row', () => {
  test('keeps previous/next off and shows no run pickers without the row’s runs and navigation', async () => {
    renderDrawer(COMPARISON_URL);
    await findDrawer();
    await waitFor(() => expect(navButton('Previous trace')).toBeDisabled());
    expect(navButton('Next trace')).toBeDisabled();
    expect(candidateRunTrigger()).toBeNull();
    expect(screen.queryByText(/Trace \d+ of \d+/)).not.toBeInTheDocument();
  });

  test('picks another run’s trace for one side and keeps the other', async () => {
    renderDrawer(COMPARISON_URL, { comparisonRunOptions: RUN_OPTIONS });
    await findDrawer();
    const trigger = await waitFor(() => {
      const element = candidateRunTrigger();
      if (!element) {
        throw new Error('Expected the candidate run picker');
      }
      return element;
    });
    expect(trigger).toHaveTextContent('Run B');
    fireEvent.click(trigger);
    fireEvent.click(await screen.findByRole('option', { name: 'Run C' }));

    await waitFor(() => expect(new URLSearchParams(lastSearch).get('candidateTraceId')).toBe('tr-c'));
    expect(new URLSearchParams(lastSearch).get('baselineTraceId')).toBe('tr-a');
    expect(new URLSearchParams(lastSearch).get('traceId')).toBe('tr-c');
  });

  test('moves between matched rows with previous/next while comparing', async () => {
    const onNext = jest.fn();
    renderDrawer(COMPARISON_URL, {
      comparisonRunOptions: RUN_OPTIONS,
      comparisonNavigation: { onNext, currentIndex: 0, totalCount: 3 },
    });
    await findDrawer();
    await waitFor(() => expect(navButton('Next trace')).toBeEnabled());
    expect(navButton('Previous trace')).toBeDisabled();
    fireEvent.click(navButton('Next trace'));
    expect(onNext).toHaveBeenCalledTimes(1);
    // The position only shows in pairwise review.
    expect(screen.queryByText('Trace 1 of 3')).not.toBeInTheDocument();
  });

  test('shows the position among matched rows in pairwise review', async () => {
    renderDrawer(`${COMPARISON_URL}&pairwiseReview=true`, {
      comparisonRunOptions: RUN_OPTIONS,
      comparisonNavigation: { onPrevious: jest.fn(), onNext: jest.fn(), currentIndex: 1, totalCount: 3 },
    });
    await findDrawer();
    expect(await screen.findByText('Trace 2 of 3')).toBeInTheDocument();
  });
});
