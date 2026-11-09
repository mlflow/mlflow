import { describe, expect, test } from '@jest/globals';
import { useState } from 'react';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { DefaultBodyType, PathParams, ResponseResolver, RestContext, RestRequest } from 'msw';
import { rest } from 'msw';

import { setupServer } from '../../../../common/utils/setup-msw';
import { useLocation } from '../../../../common/utils/RoutingUtils';
import { renderDatasetsPage } from '../test-utils/renderDatasetsPage';
import { DatasetUsedByRunsButton, DatasetUsedByRunsPanel } from './DatasetUsedByRunsDrawer';

interface SearchRunsRequestBody {
  experiment_ids?: string[];
  filter?: string;
  order_by?: string[];
}

const searchRequests: SearchRunsRequestBody[] = [];
const searchRunsHandler: ResponseResolver<
  RestRequest<SearchRunsRequestBody, PathParams>,
  RestContext,
  DefaultBodyType
> = (req, res, ctx) => {
  searchRequests.push(req.body);
  return res(
    ctx.json({
      runs: [
        {
          data: {
            metrics: [
              { key: 'Correctness', value: 0.92345, timestamp: 0, step: 0 },
              { key: 'Relevance', value: 0.84, timestamp: 0, step: 0 },
            ],
            params: [],
            tags: [],
          },
          info: {
            artifact_uri: '',
            end_time: 1720000300000,
            experiment_id: 'exp-1',
            lifecycle_stage: 'active',
            run_uuid: 'run-1',
            run_name: 'Support quality evaluation',
            start_time: 1720000000000,
            status: 'FINISHED',
          },
          inputs: { dataset_inputs: [], model_inputs: [] },
          outputs: { model_outputs: [] },
        },
      ],
      next_page_token: null,
    }),
  );
};

const server = setupServer(rest.post(/\/ajax-api\/2\.0\/mlflow\/(?:databricks\/)?runs\/search$/, searchRunsHandler));

const DatasetUsedByRunsPanelHarness = () => {
  const [open, setOpen] = useState(false);
  return (
    <>
      <DatasetUsedByRunsButton onClick={() => setOpen(true)} />
      <DatasetUsedByRunsPanel
        datasetName="Customer Support Eval"
        experimentId="exp-1"
        open={open}
        onClose={() => setOpen(false)}
      />
    </>
  );
};

// Shows where the "View in evaluation runs tab" button navigated to.
const LocationProbe = () => {
  const location = useLocation();
  return <div data-testid="location">{`${location.pathname}${location.search}`}</div>;
};

describe('DatasetUsedByRunsPanel', () => {
  test('lists runs lazily and navigates to the dataset-filtered evaluation runs page', async () => {
    const user = userEvent.setup();
    searchRequests.length = 0;
    renderDatasetsPage({
      initialUrl: '/experiments/exp-1/datasets/ds-1',
      routes: [
        {
          path: '/experiments/:experimentId/datasets/:datasetId',
          element: <DatasetUsedByRunsPanelHarness />,
        },
        {
          path: '/experiments/:experimentId/evaluation-runs',
          element: <LocationProbe />,
        },
      ],
      history: {},
    });

    expect(searchRequests).toHaveLength(0);
    await user.click(screen.getByRole('button', { name: 'Evaluation Runs' }));

    expect(await screen.findByRole('complementary', { name: 'Evaluation Runs' })).toBeInTheDocument();
    expect(await screen.findByRole('link', { name: /Support quality evaluation/i })).toHaveAttribute(
      'href',
      expect.stringContaining('/experiments/exp-1/evaluation-runs/run-1/details'),
    );
    expect(screen.getByTestId('eval-runs-chart-area')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'All assessments' })).toBeInTheDocument();
    expect(screen.queryByText('Finished')).not.toBeInTheDocument();
    expect(screen.getByText(/^Created /)).toBeInTheDocument();
    expect(screen.getByText('Duration 5.0min')).toBeInTheDocument();
    expect(screen.getByText('Run ID run-1')).toBeInTheDocument();
    await user.click(screen.getByText('Run ID run-1'));
    expect(screen.getByRole('complementary', { name: 'Evaluation Runs' })).toBeInTheDocument();
    const results = screen.getByText('Results (2)');
    await user.hover(results);
    expect(await screen.findByRole('tooltip')).toHaveTextContent('Correctness0.9235Relevance0.84');
    await waitFor(() => expect(searchRequests[0]?.filter).toBe("dataset.name = 'Customer Support Eval'"));
    expect(searchRequests[0]?.order_by).toEqual(['attributes.start_time DESC']);

    await user.click(screen.getByRole('textbox', { name: 'Filter by evaluation name' }));
    await user.paste('quality');
    await waitFor(() =>
      expect(searchRequests.some((request) => request.filter?.includes("attributes.run_name ILIKE '%quality%'"))).toBe(
        true,
      ),
    );

    await user.click(screen.getByRole('button', { name: 'Sort by created, newest first' }));
    await waitFor(() =>
      expect(searchRequests.some((request) => request.order_by?.[0] === 'attributes.start_time ASC')).toBe(true),
    );

    await user.click(screen.getByRole('button', { name: 'View in evaluation runs tab' }));
    const [pathname, search] = (await screen.findByTestId('location')).textContent!.split('?');
    expect(pathname).toBe('/experiments/exp-1/evaluation-runs');
    const params = new URLSearchParams(search);
    // All time, so a saved date preset on the evaluation runs tab can't hide the panel's older runs.
    expect(params.get('startTimeLabel')).toBe('ALL');
    expect(JSON.parse(params.get('evalRunsFilterClauses') ?? '[]')).toEqual([
      { field: 'dataset.name', operator: '=', value: 'Customer Support Eval' },
    ]);
  });
});
