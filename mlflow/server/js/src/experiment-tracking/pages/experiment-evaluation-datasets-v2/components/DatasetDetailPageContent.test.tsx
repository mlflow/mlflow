import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';

import { shouldEnableNewEvalRunsUi } from '../../../../common/utils/FeatureUtils';
import { useNavigate } from '../../../../common/utils/RoutingUtils';
import { RoutePaths } from '../../../routes';
import { setupServer } from '../../../../common/utils/setup-msw';
import type { Dataset } from '../hooks/useDatasetsQueries';
import { renderDatasetsPage } from '../test-utils/renderDatasetsPage';
import { DatasetDetailPageContent } from './DatasetDetailPageContent';

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/FeatureUtils')>(
    '../../../../common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: jest.fn(() => true),
}));

// The record editor loads monaco, which jest can't resolve; these tests only need the panel around it.
jest.mock('./LazyJsonRecordEditor', () => ({ LazyJsonRecordEditor: () => null }));

const dataset = { dataset_id: 'ds-1', name: 'Support eval', create_time: '2026-01-01T00:00:00Z' } as Dataset;

const server = setupServer(
  rest.get(/\/ajax-api\/3\.0\/mlflow\/datasets\/ds-1\/records/, (_req, res, ctx) =>
    res(ctx.json({ records: JSON.stringify([{ dataset_record_id: 'r-1', inputs: { question: 'hi' } }]) })),
  ),
  rest.post(/\/ajax-api\/2\.0\/mlflow\/runs\/search$/, (_req, res, ctx) => res(ctx.json({ runs: [] }))),
);

// Stands in for the browser Back button.
const BackButton = () => {
  const navigate = useNavigate();
  return <button onClick={() => navigate(-1)}>Browser back</button>;
};

const renderContent = (initialUrl = '/experiments/exp-1/datasets/ds-1') =>
  renderDatasetsPage({
    initialUrl,
    routes: [
      {
        path: '/experiments/:experimentId/datasets/:datasetId',
        element: (
          <>
            <BackButton />
            <DatasetDetailPageContent experimentId="exp-1" datasetId="ds-1" dataset={dataset} />
          </>
        ),
      },
      {
        path: RoutePaths.experimentPageTabEvaluationRunDetails,
        element: (
          <>
            <BackButton />
            <div>Run details page</div>
          </>
        ),
      },
    ],
    history: {},
  });

describe('DatasetDetailPageContent evaluation runs panel', () => {
  beforeEach(() => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
  });

  test('opens the panel of runs that use the dataset from the toolbar', async () => {
    const user = userEvent.setup();
    renderContent();

    await user.click(await screen.findByRole('button', { name: 'Evaluation Runs' }));

    expect(await screen.findByRole('complementary', { name: 'Evaluation Runs' })).toBeInTheDocument();
    expect(screen.getByText('Runs that use Support eval')).toBeInTheDocument();
    expect(await screen.findByText('No evaluation runs use this dataset yet')).toBeInTheDocument();
  });

  test('restores the panel from the URL', async () => {
    renderContent('/experiments/exp-1/datasets/ds-1?runsPanel=true');

    expect(await screen.findByRole('complementary', { name: 'Evaluation Runs' })).toBeInTheDocument();
  });

  test('reopens the panel on Back after following a run link from it', async () => {
    server.use(
      rest.post(/\/ajax-api\/2\.0\/mlflow\/runs\/search$/, (_req, res, ctx) =>
        res(
          ctx.json({
            runs: [
              {
                info: {
                  experiment_id: 'exp-1',
                  run_uuid: 'run-1',
                  run_name: 'Support quality run',
                  status: 'FINISHED',
                  start_time: 1720000000000,
                },
                data: { metrics: [], params: [], tags: [] },
              },
            ],
          }),
        ),
      ),
    );
    const user = userEvent.setup();
    renderContent();

    await user.click(await screen.findByRole('button', { name: 'Evaluation Runs' }));
    await user.click(await screen.findByRole('link', { name: /Support quality run/ }));
    expect(await screen.findByText('Run details page')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Browser back' }));

    expect(await screen.findByRole('complementary', { name: 'Evaluation Runs' })).toBeInTheDocument();
  });

  test('keeps a new-record draft open when Back restores the panel URL state', async () => {
    const user = userEvent.setup();
    renderContent('/experiments/exp-1/datasets/ds-1?runsPanel=true');
    expect(await screen.findByRole('complementary', { name: 'Evaluation Runs' })).toBeInTheDocument();

    await user.click(await screen.findByRole('button', { name: /Add record/i }));
    expect(await screen.findByRole('complementary', { name: 'New dataset record' })).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Browser back' }));

    expect(screen.getByRole('complementary', { name: 'New dataset record' })).toBeInTheDocument();
    expect(screen.queryByRole('complementary', { name: 'Evaluation Runs' })).not.toBeInTheDocument();
  });

  test('flag off: shows no panel button and ignores the panel URL state', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    renderContent('/experiments/exp-1/datasets/ds-1?runsPanel=true');

    expect(await screen.findByRole('button', { name: /Add record/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Evaluation Runs' })).not.toBeInTheDocument();
    expect(screen.queryByRole('complementary', { name: 'Evaluation Runs' })).not.toBeInTheDocument();
  });
});
