import { describe, test, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { IntlProvider } from 'react-intl';
import {
  ExperimentLoggedModelListPageKnownColumns,
  useExperimentLoggedModelListPageTableColumns,
} from './hooks/useExperimentLoggedModelListPageTableColumns';
import { ExperimentLoggedModelListPageColumnSelector } from './ExperimentLoggedModelListPageColumnSelector';

const getMetric = (key: string, datasetName: string | undefined) => ({
  key,
  value: 1000,
  step: 1,
  timestamp: 0,
  dataset_digest: `1234-${datasetName}`,
  dataset_name: datasetName,
  modelId: '',
  run_id: '',
});

const getDemoData = () => [
  {
    info: undefined,
    data: {
      metrics: [
        getMetric('loss', 'train'),
        getMetric('loss', 'eval'),
        getMetric('alpha', 'train'),
        getMetric('alpha', 'eval'),
        getMetric('ungrouped', undefined),
      ],
      params: [
        { key: 'param1', value: '0.9' },
        { key: 'param2', value: '0.9' },
      ],
    },
  },
];

describe('ExperimentLoggedModelListPageColumnSelector', () => {
  let currentColumnVisibility: any = {};
  const renderTestComponent = () => {
    const TestComponent = () => {
      const { columnDefs } = useExperimentLoggedModelListPageTableColumns({ loggedModels: getDemoData() });
      const [columnVisibility, setColumnVisibility] = useState<Record<string, boolean>>({});
      currentColumnVisibility = columnVisibility;
      return (
        <ExperimentLoggedModelListPageColumnSelector
          onUpdateColumns={setColumnVisibility}
          columnVisibility={columnVisibility}
          columnDefs={columnDefs}
        />
      );
    };
    render(<TestComponent />, {
      wrapper: ({ children }) => <IntlProvider locale="en">{children}</IntlProvider>,
    });
  };
  test('should handle enabling and disabling arbitrary columns', async () => {
    const user = userEvent.setup();
    renderTestComponent();

    // We start with no columns hidden
    expect(currentColumnVisibility).toEqual({});

    // Click on the columns selector
    await user.click(screen.getByRole('button', { name: 'Columns' }));

    // Toggle "eval" dataset metrics
    await user.click(screen.getByTitle('Dataset: eval (#1234-eval)'));

    // We should have only eval metrics hidden
    expect(currentColumnVisibility).toEqual({
      'metrics.{"metricKey":"alpha","datasetName":"eval","datasetDigest":"1234-eval"}': false,
      'metrics.{"metricKey":"loss","datasetName":"eval","datasetDigest":"1234-eval"}': false,
    });

    // Now toggle ungrouped metrics
    await user.click(screen.getByTitle('No dataset'));

    expect(currentColumnVisibility).toEqual({
      'metrics.{"metricKey":"alpha","datasetName":"eval","datasetDigest":"1234-eval"}': false,
      'metrics.{"metricKey":"loss","datasetName":"eval","datasetDigest":"1234-eval"}': false,
      'metrics.{"metricKey":"ungrouped"}': false,
    });

    // Disable attribute columns one by one
    await user.click(screen.getByTitle('Status'));
    await user.click(screen.getByTitle('Source run'));
    await user.click(screen.getByTitle('Dataset'));

    expect(currentColumnVisibility).toEqual({
      'metrics.{"metricKey":"alpha","datasetName":"eval","datasetDigest":"1234-eval"}': false,
      'metrics.{"metricKey":"loss","datasetName":"eval","datasetDigest":"1234-eval"}': false,
      'metrics.{"metricKey":"ungrouped"}': false,
      [ExperimentLoggedModelListPageKnownColumns.Dataset]: false,
      [ExperimentLoggedModelListPageKnownColumns.SourceRun]: false,
      [ExperimentLoggedModelListPageKnownColumns.Status]: false,
    });

    // Toggle datasets and attributes again
    await user.click(screen.getByTitle('Dataset: eval (#1234-eval)'));
    await user.click(screen.getByTitle('No dataset'));
    await user.click(screen.getByTitle('Status'));
    await user.click(screen.getByTitle('Source run'));
    await user.click(screen.getByTitle('Dataset'));

    // However, now click on parameters group
    await user.click(screen.getByTitle('Parameters'));

    // We should have only parameters hidden
    expect(currentColumnVisibility).toEqual({
      'params.param1': false,
      'params.param2': false,
    });

    // Retoggle parameters in the end
    await user.click(screen.getByTitle('Parameters'));

    // We should have all columns visible again
    expect(currentColumnVisibility).toEqual({});
  });
});
