import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';

import { shouldEnableNewEvalRunsUi } from '../../../../common/utils/FeatureUtils';
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

const dataset = { dataset_id: 'ds-1', name: 'Support eval', create_time: '2026-01-01T00:00:00Z' } as Dataset;

setupServer(
  rest.get(/\/ajax-api\/3\.0\/mlflow\/datasets\/ds-1\/records/, (_req, res, ctx) =>
    res(ctx.json({ records: JSON.stringify([{ dataset_record_id: 'r-1', inputs: { question: 'hi' } }]) })),
  ),
  rest.post(/\/ajax-api\/2\.0\/mlflow\/runs\/search$/, (_req, res, ctx) => res(ctx.json({ runs: [] }))),
);

const renderContent = (initialUrl = '/experiments/exp-1/datasets/ds-1') =>
  renderDatasetsPage({
    initialUrl,
    routes: [
      {
        path: '/experiments/:experimentId/datasets/:datasetId',
        element: <DatasetDetailPageContent experimentId="exp-1" datasetId="ds-1" dataset={dataset} />,
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

  test('flag off: shows no panel button and ignores the panel URL state', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    renderContent('/experiments/exp-1/datasets/ds-1?runsPanel=true');

    expect(await screen.findByRole('button', { name: /Add record/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Evaluation Runs' })).not.toBeInTheDocument();
    expect(screen.queryByRole('complementary', { name: 'Evaluation Runs' })).not.toBeInTheDocument();
  });
});
