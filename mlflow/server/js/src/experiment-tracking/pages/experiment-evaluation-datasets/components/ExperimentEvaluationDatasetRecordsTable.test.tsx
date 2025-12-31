import { jest, describe, beforeEach, afterEach, test, expect } from '@jest/globals';
import { render, screen, fireEvent } from '../../../../common/utils/TestUtils.react18';
import { ExperimentEvaluationDatasetRecordsTable } from './ExperimentEvaluationDatasetRecordsTable';
import { useGetDatasetRecords } from '../hooks/useGetDatasetRecords';
import { useDeleteDatasetRecordsMutation } from '../hooks/useDeleteDatasetRecordsMutation';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

jest.mock('../hooks/useGetDatasetRecords');
jest.mock('../hooks/useDeleteDatasetRecordsMutation');
jest.mock('../hooks/useInfiniteScrollFetch', () => ({
  useInfiniteScrollFetch: jest.fn(() => jest.fn()),
}));

const mockGetDatasetRecords = useGetDatasetRecords as jest.Mock;
const mockDeleteDatasetRecordsMutation = useDeleteDatasetRecordsMutation as jest.Mock;

describe('ExperimentEvaluationDatasetRecordsTable', () => {
  const mockDataset = {
    dataset_id: 'dataset-123',
    name: 'test-dataset',
    digest: 'digest-123',
    dataset_experiments: [],
    created_time: Date.now(),
    last_update_time: Date.now(),
    created_by: 'test-user',
    last_updated_by: 'test-user',
    experiment_ids: ['0'],
  };

  const mockRecords = [
    {
      dataset_record_id: 'record-1',
      dataset_id: 'dataset-123',
      inputs: { input_key: 'input_value_1' },
      outputs: { output_key: 'output_value_1' },
      expectations: null,
    },
    {
      dataset_record_id: 'record-2',
      dataset_id: 'dataset-123',
      inputs: { input_key: 'input_value_2' },
      outputs: { output_key: 'output_value_2' },
      expectations: null,
    },
  ];

  const mockDeleteMutation = jest.fn();

  beforeEach(() => {
    mockGetDatasetRecords.mockReturnValue({
      data: mockRecords,
      isLoading: false,
      isFetching: false,
      fetchNextPage: jest.fn(),
      hasNextPage: false,
    });

    mockDeleteDatasetRecordsMutation.mockReturnValue({
      deleteDatasetRecordsMutation: mockDeleteMutation,
      isLoading: false,
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  const renderComponent = () => {
    const queryClient = new QueryClient();
    return render(
      <QueryClientProvider client={queryClient}>
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <ExperimentEvaluationDatasetRecordsTable dataset={mockDataset} />
          </DesignSystemProvider>
        </IntlProvider>
      </QueryClientProvider>,
    );
  };

  test('renders records correctly', () => {
    renderComponent();
    expect(screen.getByText(/input_value_1/)).toBeInTheDocument();
    expect(screen.getByText(/input_value_2/)).toBeInTheDocument();
  });

  test('shows delete button when rows are selected', async () => {
    renderComponent();

    expect(screen.queryByRole('button', { name: /Delete/ })).not.toBeInTheDocument();

    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[1]);

    expect(screen.getByRole('button', { name: /Delete 1 record/ })).toBeInTheDocument();

    fireEvent.click(checkboxes[2]);

    expect(screen.getByRole('button', { name: /Delete 2 records/ })).toBeInTheDocument();
  });

  test('calls delete mutation when delete button is clicked', async () => {
    renderComponent();

    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[1]);

    const deleteButton = screen.getByRole('button', { name: /Delete 1 record/ });
    fireEvent.click(deleteButton);

    expect(mockDeleteMutation).toHaveBeenCalledWith({
      datasetId: 'dataset-123',
      datasetRecordIds: ['record-1'],
    });
  });

  test('handles select all', async () => {
    renderComponent();

    const selectAllCheckbox = screen.getAllByRole('checkbox')[0];
    fireEvent.click(selectAllCheckbox);

    expect(screen.getByRole('button', { name: /Delete 2 records/ })).toBeInTheDocument();

    const deleteButton = screen.getByRole('button', { name: /Delete 2 records/ });
    fireEvent.click(deleteButton);

    expect(mockDeleteMutation).toHaveBeenCalledWith({
      datasetId: 'dataset-123',
      datasetRecordIds: ['record-1', 'record-2'],
    });
  });

  test('shows loading state during deletion', () => {
    mockDeleteDatasetRecordsMutation.mockReturnValue({
      deleteDatasetRecordsMutation: mockDeleteMutation,
      isLoading: true,
    });

    renderComponent();

    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[1]);

    const deleteButton = screen.getByRole('button', { name: /Delete 1 record/ });
    expect(deleteButton).toBeDisabled();
  });
});
