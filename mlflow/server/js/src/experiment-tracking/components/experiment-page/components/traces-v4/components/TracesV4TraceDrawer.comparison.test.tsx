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
import {
  BASELINE_TRACE_ID_PARAM,
  CANDIDATE_TRACE_ID_PARAM,
  PAIRWISE_REVIEW_MODE_PARAM,
  useTracesV4UrlState,
} from '../hooks/useTracesV4UrlState';
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
const CREATE_ASSESSMENT_V3_ENDPOINT = '/ajax-api/3.0/mlflow/traces/:traceId/assessments';
const TRACE_DATA_V3_ENDPOINT = '/ajax-api/3.0/mlflow/traces/get';
const CURRENT_USER_ENDPOINT = '/ajax-api/2.0/mlflow/users/current';

const BASELINE_TRACE = makeTaggedTrace('tr-000', { env: 'prod' });
const CANDIDATE_TRACE = makeTaggedTrace('tr-001', { env: 'prod' });
const BASELINE_LONG_ID = 'trace:/cat.sch/tr-000';
const CANDIDATE_LONG_ID = 'trace:/cat.sch/tr-001';

let createdAssessments: { traceId: string; assessment: Record<string, any> }[] = [];
// The signed-in user on an auth-enabled server; undefined is a server without auth, where `/users/current` 404s.
let signedInUsername: string | undefined;

setupServer(
  rest.get(CURRENT_USER_ENDPOINT, (_req, res, ctx) =>
    signedInUsername ? res(ctx.json({ user: { username: signedInUsername } })) : res(ctx.status(404)),
  ),
  // Trace data is read from `traces/get?trace_id=…`; answer it per trace so both compared sides differ.
  rest.get(TRACE_DATA_V3_ENDPOINT, (req, res, ctx) => {
    const trace = req.url.searchParams.get('trace_id') === 'tr-001' ? CANDIDATE_TRACE : BASELINE_TRACE;
    return res(ctx.json({ trace: { trace_info: trace, spans: [] } }));
  }),
  rest.get(TRACE_INFO_V3_ENDPOINT, (req, res, ctx) => {
    const trace = req.params['traceId'] === 'tr-001' ? CANDIDATE_TRACE : BASELINE_TRACE;
    return res(ctx.json({ trace: { trace_info: trace, spans: [] } }));
  }),
  rest.post(CREATE_ASSESSMENT_V3_ENDPOINT, async (req, res, ctx) => {
    const { assessment } = await req.json();
    const traceId = String(req.params['traceId']);
    createdAssessments.push({ traceId, assessment });
    return res(
      ctx.json({
        assessment: {
          ...assessment,
          assessment_id: `a-${traceId}-${createdAssessments.length}`,
          create_time: '2026-01-01T00:00:00.000Z',
          last_update_time: '2026-01-01T00:00:00.000Z',
        },
      }),
    );
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
  createdAssessments = [];
  signedInUsername = undefined;
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

const comparisonUrl = (extraParams: Record<string, string> = {}) =>
  `${URL}?${new URLSearchParams({
    [BASELINE_TRACE_ID_PARAM]: BASELINE_LONG_ID,
    [CANDIDATE_TRACE_ID_PARAM]: CANDIDATE_LONG_ID,
    ...extraParams,
  }).toString()}`;

const renderComparisonDrawer = (onClose = jest.fn(), extraParams: Record<string, string> = {}) =>
  renderTracesV4Page({
    initialUrl: comparisonUrl(extraParams),
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

const renderReviewWithPageClose = () => {
  // Close the drawer the way the Traces page does, so the real URL update is covered.
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
  return renderTracesV4Page({
    initialUrl: `${comparisonUrl({ [PAIRWISE_REVIEW_MODE_PARAM]: 'true' })}&traceId=${encodeURIComponent(
      CANDIDATE_LONG_ID,
    )}`,
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
};

describe('TracesV4TraceDrawer pairwise review', () => {
  test('opens pairwise review in full screen from the comparison header', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const user = userEvent.setup();
    renderComparisonDrawer();

    const drawer = await findDrawer();
    await within(drawer).findByText('Comparing traces');
    expect(within(drawer).queryByText('Which run is better?')).not.toBeInTheDocument();

    await user.click(within(drawer).getByRole('button', { name: 'Review (R)' }));

    expect(await within(drawer).findByText('Which run is better?')).toBeInTheDocument();
    expect(within(drawer).getByRole('button', { name: 'Exit full screen' })).toBeInTheDocument();
    expect(new URLSearchParams(lastSearch).get(PAIRWISE_REVIEW_MODE_PARAM)).toBe('true');
  }, 20000);

  test('r toggles pairwise review in a comparison, but not while typing a comment', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const user = userEvent.setup();
    renderComparisonDrawer();

    const drawer = await findDrawer();
    await within(drawer).findByText('Comparing traces');
    await user.keyboard('r');

    expect(await within(drawer).findByText('Which run is better?')).toBeInTheDocument();
    expect(within(drawer).getByRole('button', { name: 'Exit full screen' })).toBeInTheDocument();
    expect(new URLSearchParams(lastSearch).get(PAIRWISE_REVIEW_MODE_PARAM)).toBe('true');

    await user.type(within(drawer).getByRole('textbox', { name: 'Comment' }), 'r');
    expect(within(drawer).getByText('Which run is better?')).toBeInTheDocument();

    within(drawer).getByRole('button', { name: 'Review (R)' }).focus();
    await user.keyboard('r');
    await waitFor(() => expect(within(drawer).queryByText('Which run is better?')).not.toBeInTheDocument());
    expect(new URLSearchParams(lastSearch).get(PAIRWISE_REVIEW_MODE_PARAM)).toBeNull();
  }, 20000);

  test('saves a win for the chosen trace and a loss for the other one', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const user = userEvent.setup();
    renderComparisonDrawer(jest.fn(), { [PAIRWISE_REVIEW_MODE_PARAM]: 'true' });

    const drawer = await findDrawer();
    await within(drawer).findByText('Which run is better?');
    await waitFor(() => expect(within(drawer).getByRole('radio', { name: /000/ })).toBeEnabled());
    await user.keyboard('1');

    await waitFor(() => expect(createdAssessments).toHaveLength(2));
    const byTrace = Object.fromEntries(createdAssessments.map(({ traceId, assessment }) => [traceId, assessment]));
    expect(byTrace['tr-000']).toEqual(
      expect.objectContaining({
        assessment_name: 'Pairwise preference',
        feedback: { value: 1 },
        metadata: { 'mlflow.assessment.pairwiseOpponentTraceId': 'tr-001' },
        source: { source_type: 'HUMAN', source_id: 'User' },
      }),
    );
    expect(byTrace['tr-001']).toEqual(
      expect.objectContaining({
        assessment_name: 'Pairwise preference',
        feedback: { value: 0 },
        metadata: { 'mlflow.assessment.pairwiseOpponentTraceId': 'tr-000' },
      }),
    );
  }, 20000);

  test('attributes the review to the signed-in user on a server with auth', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    signedInUsername = 'alice';
    const user = userEvent.setup();
    renderComparisonDrawer(jest.fn(), { [PAIRWISE_REVIEW_MODE_PARAM]: 'true' });

    const drawer = await findDrawer();
    await within(drawer).findByText('Which run is better?');
    await waitFor(() => expect(within(drawer).getByRole('radio', { name: /000/ })).toBeEnabled());
    await user.keyboard('2');

    await waitFor(() => expect(createdAssessments).toHaveLength(2));
    for (const { assessment } of createdAssessments) {
      expect(assessment['source']).toEqual({ source_type: 'HUMAN', source_id: 'alice' });
    }
  }, 20000);

  test('saves an edited comment when Escape closes the drawer', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const user = userEvent.setup();
    renderReviewWithPageClose();

    const drawer = await findDrawer();
    await within(drawer).findByText('Which run is better?');
    await waitFor(() => expect(within(drawer).getByRole('radio', { name: /000/ })).toBeEnabled());
    await user.keyboard('1');
    await waitFor(() => expect(createdAssessments).toHaveLength(2));
    await waitFor(() => expect(within(drawer).getByRole('textbox', { name: 'Comment' })).toBeEnabled());

    await user.type(within(drawer).getByRole('textbox', { name: 'Comment' }), 'late comment');
    await user.keyboard('{Escape}');

    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    await waitFor(() => expect(createdAssessments).toHaveLength(4));
    expect(createdAssessments.slice(2).map(({ assessment }) => assessment['rationale'])).toEqual([
      'late comment',
      'late comment',
    ]);
  }, 20000);

  test('clears pairwise review from the URL when the drawer closes', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const user = userEvent.setup();
    renderReviewWithPageClose();

    const drawer = await findDrawer();
    await within(drawer).findByText('Which run is better?');
    await user.click(within(drawer).getByRole('button', { name: 'Close trace panel' }));

    await waitFor(() => {
      const params = new URLSearchParams(lastSearch);
      expect(params.get('traceId')).toBeNull();
      expect(params.get(BASELINE_TRACE_ID_PARAM)).toBeNull();
      expect(params.get(PAIRWISE_REVIEW_MODE_PARAM)).toBeNull();
    });
  }, 20000);

  test('ignores the pairwise review URL param without the new eval runs UI', async () => {
    renderComparisonDrawer(jest.fn(), { [PAIRWISE_REVIEW_MODE_PARAM]: 'true' });

    const drawer = await findDrawer();
    expect(await within(drawer).findByText('Trace', { exact: true })).toBeInTheDocument();
    expect(within(drawer).queryByText('Which run is better?')).not.toBeInTheDocument();
    expect(within(drawer).queryByRole('button', { name: 'Review (R)' })).not.toBeInTheDocument();
  }, 20000);
});
