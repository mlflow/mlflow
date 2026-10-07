import { jest, describe, test, expect } from '@jest/globals';
import type { DeepPartial } from 'redux';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import type { RunDatasetWithTags, RunInfoEntity } from '../../../types';
import { RunViewDatasetBoxV2 } from './RunViewDatasetBoxV2';
import userEvent from '@testing-library/user-event';
import { ExperimentViewDatasetDrawer } from '../../experiment-page/components/runs/ExperimentViewDatasetDrawer';
import { DesignSystemProvider } from '@databricks/design-system';
import { MemoryRouter, Route, Routes } from '@mlflow/mlflow/src/common/utils/RoutingUtils';

jest.mock('../../experiment-page/components/runs/ExperimentViewDatasetDrawer', () => ({
  ExperimentViewDatasetDrawer: jest.fn(() => <div />),
}));

const testRunInfo = {
  experimentId: '12345',
  lifecycleStage: 'active',
  runName: 'Test run name',
  runUuid: 'test-run-uuid',
} as RunInfoEntity;

const testTags = { testTag: { key: 'testTag', value: 'xyz' } } as any;

const renderWithRoute = (datasets: DeepPartial<RunDatasetWithTags>[]) =>
  renderWithIntl(
    <DesignSystemProvider>
      <MemoryRouter initialEntries={['/experiments/12345/runs/test-run-uuid']}>
        <Routes>
          <Route
            path="/experiments/:experimentId/runs/:runUuid"
            element={<RunViewDatasetBoxV2 runInfo={testRunInfo} datasets={datasets as any} tags={testTags} />}
          />
        </Routes>
      </MemoryRouter>
    </DesignSystemProvider>,
  );

describe('RunViewDatasetBoxV2', () => {
  test('Classic dataset opens the drawer on click', async () => {
    jest.mocked(ExperimentViewDatasetDrawer).mockClear();
    const dataset = { tags: [], dataset: { digest: '12345', name: 'dataset_train' } };
    renderWithRoute([dataset]);

    const button = screen.getByRole('button', { name: /dataset_train \(12345\)/ });
    await userEvent.click(button);

    expect(ExperimentViewDatasetDrawer).toHaveBeenLastCalledWith(
      expect.objectContaining({
        isOpen: true,
        selectedDatasetWithRun: expect.objectContaining({ datasetWithTags: dataset }),
      }),
      // @ts-expect-error Expected 1 arguments, but got 2
      {},
    );
  });

  test('Evaluation dataset links directly to the dataset detail page instead of opening the drawer', async () => {
    jest.mocked(ExperimentViewDatasetDrawer).mockClear();
    const evalDataset = {
      tags: [],
      dataset: {
        digest: 'ev123',
        name: 'eval_dataset',
        sourceType: 'mlflow_evaluation_dataset',
        source: JSON.stringify({ dataset_id: 'd-eval-123' }),
      },
    };
    renderWithRoute([evalDataset]);

    const link = screen.getByRole('link', { name: /eval_dataset/ });
    expect(link.getAttribute('href')).toContain('d-eval-123');

    await userEvent.click(link);
    expect(ExperimentViewDatasetDrawer).not.toHaveBeenCalled();
  });
});
