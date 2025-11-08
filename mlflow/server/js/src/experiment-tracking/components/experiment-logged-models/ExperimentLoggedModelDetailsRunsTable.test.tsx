import { IntlProvider } from 'react-intl';
import { render, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import type { LoggedModelProto, RunEntity } from '../../types';
import { ExperimentLoggedModelDetailsPageRunsTable } from './ExperimentLoggedModelDetailsRunsTable';
import { testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';

describe('ExperimentLoggedModelDetailsRunsTable', () => {
  const relatedRuns: RunEntity[] = [1, 2, 3].map(
    (i) =>
      ({
        data: {},
        info: {
          runName: `test-run-name-${i}`,
          runUuid: `test-run-id-${i}`,
        },
      } as any),
  );
  const testLoggedModelMetrics: LoggedModelProto = {
    info: {
      source_run_id: 'test-run-id-2',
      model_id: 'test-model-id',
    },
    data: {
      metrics: [
        {
          run_id: 'test-run-id-1',
          key: 'test-key',
          value: 1,
          dataset_name: 'dataset-run-1',
          dataset_digest: '1',
          model_id: 'test-model-id',
        },
        {
          run_id: 'test-run-id-2',
          key: 'test-key',
          value: 1,
          dataset_name: 'dataset-run-2',
          dataset_digest: '2',
          model_id: 'test-model-id',
        },
      ],
    },
  };

  const testLoggedModelNoMetrics: LoggedModelProto = {
    info: {
      source_run_id: 'test-run-id-2',
      model_id: 'test-model-id',
    },
  };

  const renderComponent = (loggedModel: LoggedModelProto) =>
    render(
      <IntlProvider locale="en">
        <TestRouter
          routes={[
            testRoute(
              <ExperimentLoggedModelDetailsPageRunsTable relatedRunsData={relatedRuns} loggedModel={loggedModel} />,
            ),
          ]}
        />
      </IntlProvider>,
    );

  test('should render runs extracted from metrics', async () => {
    renderComponent(testLoggedModelMetrics);

    await waitFor(() => {
      expect(screen.getByRole('cell', { name: 'test-run-name-1' })).toBeVisible();
      expect(screen.getByRole('cell', { name: 'test-run-name-2' })).toBeVisible();
    });

    // We have 2 runs + 1 header row
    expect(screen.getAllByRole('row')).toHaveLength(2 + 1);
  });

  test('should render data based only on source run', async () => {
    renderComponent(testLoggedModelNoMetrics);

    await waitFor(() => {
      expect(screen.getByRole('cell', { name: 'test-run-name-2' })).toBeVisible();
    });

    // We have 1 run + 1 header row
    expect(screen.getAllByRole('row')).toHaveLength(1 + 1);
  });
});
