import { beforeEach, describe, expect, test } from '@jest/globals';
import { rest } from 'msw';

import { renderWithDesignSystem, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { setupServer } from '../../../common/utils/setup-msw';
import { QueryClient, QueryClientProvider } from '../../../common/utils/reactQueryHooks';
import { MLFLOW_ISSUE_DETECTION_JOB_ID_TAG } from '../../constants';
import { useLatestIssueDetection } from './useIssueDetection';

/** The one RUNNING detection run the search endpoint reports, with or without a job tag. */
let runTags: { key: string; value: string }[];
let runStartedAtMs: number;

const server = setupServer(
  rest.post('*/ajax-api/2.0/mlflow/runs/search', (_req, res, ctx) =>
    res(
      ctx.json({
        runs: [
          {
            info: { run_id: 'run-1', status: 'RUNNING', start_time: runStartedAtMs },
            data: { tags: runTags },
          },
        ],
      }),
    ),
  ),
  rest.get('*/ajax-api/3.0/mlflow/jobs/:jobId', (_req, res, ctx) => res(ctx.json({ status: 1 }))),
);

const Probe = () => {
  const detection = useLatestIssueDetection('1');
  return (
    <div>
      <span data-testid="in-flight">{String(Boolean(detection.run))}</span>
      <span data-testid="stalled">{String(detection.submissionStalled)}</span>
    </div>
  );
};

const render = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, cacheTime: 0 } } });
  return renderWithDesignSystem(
    <QueryClientProvider client={client}>
      <Probe />
    </QueryClientProvider>,
  );
};

beforeEach(() => {
  runTags = [];
  runStartedAtMs = Date.now();
});

describe('useLatestIssueDetection, when job submission failed', () => {
  test('a RUNNING run that never got a job ID stops counting as a detection in flight', async () => {
    // Reported in review: the invoke endpoint creates the run and then tags it with the job
    // ID. If submission fails in between, the run stays RUNNING with no job, and every
    // surface derives "in flight" from the run alone -- so Detect was disabled forever with
    // nothing to cancel, recoverable only by editing the database.
    runTags = [];
    runStartedAtMs = Date.now() - 10 * 60 * 1000;

    render();

    await waitFor(() => expect(screen.getByTestId('stalled')).toHaveTextContent('true'));
    expect(screen.getByTestId('in-flight')).toHaveTextContent('false');
  });

  test('a run that is merely starting is still in flight, so a slow submission is not mistaken for a failure', async () => {
    runTags = [];
    runStartedAtMs = Date.now();

    render();

    await waitFor(() => expect(screen.getByTestId('in-flight')).toHaveTextContent('true'));
    expect(screen.getByTestId('stalled')).toHaveTextContent('false');
  });

  test('an old run that does have a job ID is left alone', async () => {
    // The grace window must key off the missing job, not the run's age, or a legitimately
    // long-running detection would be discarded mid-flight.
    runTags = [{ key: MLFLOW_ISSUE_DETECTION_JOB_ID_TAG, value: 'job-1' }];
    runStartedAtMs = Date.now() - 10 * 60 * 1000;

    render();

    await waitFor(() => expect(screen.getByTestId('in-flight')).toHaveTextContent('true'));
    expect(screen.getByTestId('stalled')).toHaveTextContent('false');
  });
});
