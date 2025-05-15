import { useState } from 'react';
import { useExperimentLoggedModelListPageTableColumns } from './hooks/useExperimentLoggedModelListPageTableColumns';
import { ExperimentLoggedModelListPageOrderBySelector } from './ExperimentLoggedModelListPageOrderBySelector';
import { render, screen, waitFor, within } from '../../../common/utils/TestUtils.react18';
import { IntlProvider } from 'react-intl';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';

const metrics = [
  { dataset_name: 'train', dataset_digest: '123456', key: 'rmse', value: 0.1 },
  { dataset_name: 'train', dataset_digest: '123456', key: 'r2', value: 0.1 },
  { dataset_name: 'test', dataset_digest: '987654', key: 'rmse', value: 0.1 },
  { dataset_name: 'test', dataset_digest: '987654', key: 'r2', value: 0.1 },
  { dataset_name: undefined, dataset_digest: undefined, key: 'accuracy', value: 0.1 },
];

const testLoggedModels = [
  {
    info: { model_id: 'm-1', name: 'model_with_all_metrics' },
    data: { metrics },
  },
  {
    info: { model_id: 'm-2', name: 'model_with_test_metrics' },
    data: { metrics: metrics.filter((m) => m.dataset_name === 'test') },
  },
  {
    info: { model_id: 'm-2', name: 'model_with_train_metrics' },
    data: { metrics: metrics.filter((m) => m.dataset_name === 'train') },
  },
];
describe('ExperimentLoggedModelListPageOrderBySelector', () => {
  const renderTestComponent = () => {
    const TestComponent = () => {
      const [loggedModels, setLoggedModels] = useState(testLoggedModels);
      const [orderByAsc, setOrderByAsc] = useState(true);
      const [orderByColumn, setOrderByColumn] = useState('creation_time');
      const { columnDefs } = useExperimentLoggedModelListPageTableColumns({
        loggedModels,
      });

      return (
        <div>
          <ExperimentLoggedModelListPageOrderBySelector
            orderByColumn={orderByColumn}
            orderByAsc={orderByAsc}
            onChangeOrderBy={(column, asc) => {
              setOrderByAsc(asc);
              setOrderByColumn(column);
            }}
            columnDefs={columnDefs}
          />
          <div>
            Currently sorting by: {orderByColumn} {orderByAsc ? 'ascending' : 'descending'}
          </div>
          <div>
            <button
              onClick={() => {
                setLoggedModels([testLoggedModels[1]]);
              }}
            >
              Load models with test dataset only
            </button>
            <button
              onClick={() => {
                setLoggedModels([]);
              }}
            >
              Clear model list
            </button>
          </div>
        </div>
      );
    };

    return render(<TestComponent />, {
      wrapper: ({ children }) => (
        <DesignSystemProvider>
          <IntlProvider locale="en">{children}</IntlProvider>
        </DesignSystemProvider>
      ),
    });
  };

  it('should render without crashing', async () => {
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Sort/ })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('button', { name: /Sort/ }));

    expect(screen.getByRole('group', { name: 'Metrics (test (#987654))' })).toBeInTheDocument();
    expect(screen.getByRole('group', { name: 'Metrics (train (#123456))' })).toBeInTheDocument();
    expect(screen.getByRole('group', { name: 'Metrics' })).toBeInTheDocument();

    await userEvent.click(
      within(screen.getByRole('group', { name: 'Metrics (test (#987654))' })).getByRole('menuitemcheckbox', {
        name: 'rmse',
      }),
    );

    await waitFor(() => {
      screen.getByText('Currently sorting by: metrics.["test","987654"].rmse ascending');
    });

    expect(screen.getByRole('button', { name: 'Sort: rmse' })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Sort: rmse' }));

    await userEvent.click(screen.getByRole('button', { name: 'Sort descending' }));

    await waitFor(() => {
      screen.getByText('Currently sorting by: metrics.["test","987654"].rmse descending');
    });

    // Close the dropdown
    await userEvent.click(screen.getByRole('menu', { name: 'Sort: rmse' }));

    // Click "Clear model list" button
    await userEvent.click(screen.getByText('Clear model list'));

    // Open the dropdown again
    await userEvent.click(screen.getByRole('button', { name: 'Sort: rmse' }));

    // Check if the currently sorted column is still visible
    await waitFor(() => {
      expect(
        within(screen.getByRole('group', { name: 'Currently sorted by' })).getByRole('menuitemcheckbox', {
          name: 'rmse',
        }),
      ).toBeInTheDocument();
    });

    // Close the dropdown
    await userEvent.click(screen.getByRole('menu', { name: 'Sort: rmse' }));

    // Load models with test dataset only
    await userEvent.click(screen.getByText('Load models with test dataset only'));

    // Open the dropdown again
    await userEvent.click(screen.getByRole('button', { name: 'Sort: rmse' }));

    expect(screen.getByRole('group', { name: 'Metrics (test (#987654))' })).toBeInTheDocument();
    expect(screen.queryByRole('group', { name: 'Metrics (train (#123456))' })).not.toBeInTheDocument();
    expect(screen.queryByRole('group', { name: 'Metrics' })).not.toBeInTheDocument();
  });
});
