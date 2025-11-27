import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { renderWithIntl, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { ExperimentEvaluationDatasetRecordsTable } from './ExperimentEvaluationDatasetRecordsTable';
import type { EvaluationDataset } from '../types';

// Mock the hooks
jest.mock('../hooks/useGetDatasetRecords', () => ({
  useGetDatasetRecords: jest.fn(() => ({
    data: [],
    isLoading: false,
    isFetching: false,
    error: null,
    fetchNextPage: jest.fn(),
    hasNextPage: false,
  })),
}));

jest.mock('../hooks/useInfiniteScrollFetch', () => ({
  useInfiniteScrollFetch: jest.fn(() => jest.fn()),
}));

// Import the mocked hook to control its return value
import { useGetDatasetRecords } from '../hooks/useGetDatasetRecords';

describe('ExperimentEvaluationDatasetRecordsTable - Source Cell Rendering', () => {
  const mockDataset: EvaluationDataset = {
    dataset_id: 'test-dataset-id',
    name: 'Test Dataset',
    experiment_ids: ['0'],
    created_time: Date.now(),
    last_update_time: Date.now(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders SourceCell with trace link in the table', async () => {
    const traceId = 'tr-test-trace';
    const mockRecords = [
      {
        dataset_record_id: 'record-1',
        dataset_id: 'test-dataset-id',
        inputs: JSON.stringify({ query: 'test' }),
        expectations: JSON.stringify({}),
        outputs: JSON.stringify({}),
        source: JSON.stringify({
          source_type: 'TRACE',
          source_data: { trace_id: traceId },
        }),
        tags: null,
      },
    ];

    (useGetDatasetRecords as jest.Mock).mockReturnValue({
      data: mockRecords,
      isLoading: false,
      isFetching: false,
      error: null,
      fetchNextPage: jest.fn(),
      hasNextPage: false,
    });

    renderWithIntl(
      <ExperimentEvaluationDatasetRecordsTable
        dataset={mockDataset}
        onOpenTraceModal={jest.fn()}
      />,
    );

    // Verify the Source column header is rendered
    await waitFor(() => {
      expect(screen.getByText('Source')).toBeInTheDocument();
    });

    // Verify the trace link is rendered in the Source cell
    const traceLink = screen.getByRole('button', { name: `Trace: ${traceId}` });
    expect(traceLink).toBeInTheDocument();
    expect(traceLink).toHaveTextContent(`Trace: ${traceId}`);
  });

  test('renders SourceCell with hyphen for non-trace sources', async () => {
    const mockRecords = [
      {
        dataset_record_id: 'record-1',
        dataset_id: 'test-dataset-id',
        inputs: JSON.stringify({ query: 'test' }),
        expectations: JSON.stringify({}),
        outputs: JSON.stringify({}),
        source: JSON.stringify({
          source_type: 'HUMAN',
          source_data: {},
        }),
        tags: null,
      },
    ];

    (useGetDatasetRecords as jest.Mock).mockReturnValue({
      data: mockRecords,
      isLoading: false,
      isFetching: false,
      error: null,
      fetchNextPage: jest.fn(),
      hasNextPage: false,
    });

    renderWithIntl(
      <ExperimentEvaluationDatasetRecordsTable
        dataset={mockDataset}
        onOpenTraceModal={jest.fn()}
      />,
    );

    // Verify Source column is rendered
    await waitFor(() => {
      expect(screen.getByText('Source')).toBeInTheDocument();
    });

    // Verify no trace link is rendered for HUMAN source
    expect(screen.queryByRole('button', { name: /Trace:/ })).not.toBeInTheDocument();
  });
});
