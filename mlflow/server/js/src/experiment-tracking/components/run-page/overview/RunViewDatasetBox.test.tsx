import type { DeepPartial } from 'redux';
import { renderWithIntl, act, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import type { RunDatasetWithTags, RunInfoEntity } from '../../../types';
import { RunViewDatasetBox } from './RunViewDatasetBox';
import userEvent from '@testing-library/user-event';
import { openDropdownMenu } from '@databricks/design-system/test-utils/rtl';
import { ExperimentViewDatasetDrawer } from '../../experiment-page/components/runs/ExperimentViewDatasetDrawer';
import { DesignSystemProvider } from '@databricks/design-system';
jest.mock('../../experiment-page/components/runs/ExperimentViewDatasetDrawer', () => ({
  ExperimentViewDatasetDrawer: jest.fn(() => <div />),
}));

const testRunUuid = 'test-run-uuid';
const testRunName = 'Test run name';
const testExperimentId = '12345';

const testRunInfo = {
  experimentId: testExperimentId,
  lifecycleStage: 'active',
  runName: testRunName,
  runUuid: testRunUuid,
} as RunInfoEntity;

const testTags = { testTag: { key: 'testTag', value: 'xyz' } } as any;

describe('RunViewDatasetBox', () => {
  const renderComponent = (datasets: DeepPartial<RunDatasetWithTags>[] = []) => {
    return renderWithIntl(
      <DesignSystemProvider>
        <RunViewDatasetBox runInfo={testRunInfo} datasets={datasets as any} tags={testTags} />
      </DesignSystemProvider>,
    );
  };

  test('Render single dataset', async () => {
    const testDatasetWithTags = {
      tags: [{ key: 'mlflow.data.context', value: 'train' }],
      dataset: {
        digest: '12345',
        name: 'dataset_train',
      },
    };
    renderComponent([testDatasetWithTags]);

    const linkElement = screen.getByRole('link', { name: /dataset_train \(12345\)/ });

    expect(linkElement).toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();

    await userEvent.click(linkElement);

    expect(ExperimentViewDatasetDrawer).toHaveBeenLastCalledWith(
      expect.objectContaining({
        isOpen: true,
        selectedDatasetWithRun: expect.objectContaining({
          datasetWithTags: testDatasetWithTags,
        }),
      }),
      {},
    );
  });

  test('Render multiple datasets', async () => {
    const datasets = [
      {
        tags: [{ key: 'mlflow.data.context', value: 'train' }],
        dataset: {
          digest: '12345',
          name: 'dataset_train',
        },
      },
      {
        tags: [{ key: 'mlflow.data.context', value: 'eval' }],
        dataset: {
          digest: '54321',
          name: 'dataset_eval',
        },
      },
    ];
    const [, evalDataset] = datasets;
    renderComponent(datasets);

    const expandButton = screen.getByRole('button', { name: '+1' });
    expect(expandButton).toBeInTheDocument();

    await act(async () => {
      await openDropdownMenu(expandButton);
    });

    const linkElement = screen.getByRole('link', { name: /dataset_eval \(54321\)/ });

    await userEvent.click(linkElement);

    expect(ExperimentViewDatasetDrawer).toHaveBeenLastCalledWith(
      expect.objectContaining({
        isOpen: true,
        selectedDatasetWithRun: expect.objectContaining({
          datasetWithTags: evalDataset,
        }),
      }),
      {},
    );
  });
});
