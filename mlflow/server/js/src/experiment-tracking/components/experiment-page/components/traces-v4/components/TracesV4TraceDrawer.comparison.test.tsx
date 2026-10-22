import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import { setupServer } from '@mlflow/mlflow/src/common/utils/setup-msw';
import { setupTestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import { useLocation } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useEffect } from 'react';
import { shouldEnableNewEvalRunsUi } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { TracesV4TraceDrawer } from './TracesV4TraceDrawer';
import { BASELINE_TRACE_ID_PARAM, CANDIDATE_TRACE_ID_PARAM, useTracesV4UrlState } from '../hooks/useTracesV4UrlState';
import { renderTracesV4Page } from '../test-utils/renderTracesV4Page';
import { makeTaggedTrace } from '../test-utils/mockTraces';

jest.mock('@mlflow/mlflow/src/common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FeatureUtils')>(
    '@mlflow/mlflow/src/common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

const EXPERIMENT_ID = 'exp-1';
const PATH = '/ml/experiments/:experimentId/traces';
const URL = '/ml/experiments/exp-1/traces';
const TRACE_INFO_V3_ENDPOINT = '/ajax-api/3.0/mlflow/traces/:traceId';

const BASELINE_TRACE = makeTaggedTrace('tr-000', { env: 'prod' });
const CANDIDATE_TRACE = makeTaggedTrace('tr-001', { env: 'prod' });
const BASELINE_LONG_ID = 'trace:/cat.sch/tr-000';
const CANDIDATE_LONG_ID = 'trace:/cat.sch/tr-001';

setupServer(
  rest.get(TRACE_INFO_V3_ENDPOINT, (req, res, ctx) => {
    const trace = req.params['traceId'] === 'tr-001' ? CANDIDATE_TRACE : BASELINE_TRACE;
    return res(ctx.json({ trace: { trace_info: trace, spans: [] } }));
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
  jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
  // The new eval runs UI shows the header metadata (and so the comparison header) only on wide drawers.
  window.innerWidth = 1600;
});

afterEach(() => {
  window.innerWidth = JSDOM_WIDTH;
});

afterAll(() => {
  delete process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'];
});

const comparisonUrl = () =>
  `${URL}?${new URLSearchParams({
    [BASELINE_TRACE_ID_PARAM]: BASELINE_LONG_ID,
    [CANDIDATE_TRACE_ID_PARAM]: CANDIDATE_LONG_ID,
  }).toString()}`;

const renderComparisonDrawer = (onClose = jest.fn()) =>
  renderTracesV4Page({
    initialUrl: comparisonUrl(),
    routes: [
      {
        path: PATH,
        element: (
          <>
            <LocationSpy />
            <TracesV4TraceDrawer
              traceId={CANDIDATE_LONG_ID}
              onClose={onClose}
              experimentId={EXPERIMENT_ID}
              traces={[BASELINE_TRACE, CANDIDATE_TRACE]}
              onSelectTrace={() => {}}
            />
          </>
        ),
      },
    ],
    history,
    experimentId: EXPERIMENT_ID,
  });

const findDrawer = () => screen.findByRole('dialog', {}, { timeout: 10000 });

describe('TracesV4TraceDrawer trace comparison', () => {
  test('opens a comparison of the baseline and candidate traces from the URL with the new eval runs UI', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderComparisonDrawer();

    const drawer = await findDrawer();
    expect(await within(drawer).findByText('Comparing traces')).toBeInTheDocument();
    expect(within(drawer).getByText('000')).toBeInTheDocument();
    expect(within(drawer).getByText('001')).toBeInTheDocument();
    expect(within(drawer).queryByText('Trace', { exact: true })).not.toBeInTheDocument();
  }, 20000);

  test('swaps the baseline and candidate in the URL', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const user = userEvent.setup();
    renderComparisonDrawer();

    const drawer = await findDrawer();
    await user.click(await within(drawer).findByRole('button', { name: 'Swap baseline and candidate' }));

    await waitFor(() => {
      const params = new URLSearchParams(lastSearch);
      expect(params.get(BASELINE_TRACE_ID_PARAM)).toBe(CANDIDATE_LONG_ID);
      expect(params.get(CANDIDATE_TRACE_ID_PARAM)).toBe(BASELINE_LONG_ID);
      expect(params.get('traceId')).toBe(BASELINE_LONG_ID);
    });
  }, 20000);

  test('clears the trace and the comparison from the URL when the drawer closes', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const user = userEvent.setup();
    // Close the drawer the way the Traces page does (`url.setTraceId(undefined)`), so the test covers
    // the real URL update instead of a mock that never touches the URL.
    const DrawerWithPageClose = () => {
      const url = useTracesV4UrlState();
      return (
        <TracesV4TraceDrawer
          traceId={CANDIDATE_LONG_ID}
          onClose={() => url.setTraceId(undefined)}
          experimentId={EXPERIMENT_ID}
          traces={[BASELINE_TRACE, CANDIDATE_TRACE]}
          onSelectTrace={() => {}}
        />
      );
    };
    renderTracesV4Page({
      initialUrl: `${comparisonUrl()}&traceId=${encodeURIComponent(CANDIDATE_LONG_ID)}`,
      routes: [
        {
          path: PATH,
          element: (
            <>
              <LocationSpy />
              <DrawerWithPageClose />
            </>
          ),
        },
      ],
      history,
      experimentId: EXPERIMENT_ID,
    });

    const drawer = await findDrawer();
    await within(drawer).findByText('Comparing traces');
    await user.click(within(drawer).getByRole('button', { name: 'Close trace panel' }));

    await waitFor(() => {
      const params = new URLSearchParams(lastSearch);
      expect(params.get('traceId')).toBeNull();
      expect(params.get(BASELINE_TRACE_ID_PARAM)).toBeNull();
      expect(params.get(CANDIDATE_TRACE_ID_PARAM)).toBeNull();
    });
  }, 20000);

  test('ignores comparison URL params without the new eval runs UI', async () => {
    renderComparisonDrawer();

    const drawer = await findDrawer();
    expect(await within(drawer).findByText('Trace', { exact: true })).toBeInTheDocument();
    expect(within(drawer).queryByText('Comparing traces')).not.toBeInTheDocument();
    expect(within(drawer).queryByRole('button', { name: 'Swap baseline and candidate' })).not.toBeInTheDocument();
  }, 20000);
});
