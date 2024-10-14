import { type ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';
import { screen } from '@testing-library/react';
import { TracesViewTable, TracesViewTableProps } from './TracesViewTable';
import { renderWithIntl } from '../../../common/utils/TestUtils.react18';
import { KeyValueEntity } from '../../types';

jest.setTimeout(30000); // Larger timeout for integration testing (table rendering)

const generateMockTrace = (
  uniqueId: string,
  timestampMs = 100,
  requestMetadata: KeyValueEntity[] = [],
  tags: KeyValueEntity[] = [],
): ModelTraceInfo => ({
  request_id: `tr-${uniqueId}`,
  experiment_id: 'test-experiment-id',
  timestamp_ms: 1712134300000 + timestampMs,
  execution_time_ms: timestampMs,
  status: 'OK',
  attributes: {},
  request_metadata: [...requestMetadata],
  tags: [
    {
      key: 'mlflow.traceName',
      value: `Trace name: ${uniqueId}`,
    },
    ...tags,
  ],
});

describe('ExperimentViewTracesTable', () => {
  const renderTestComponent = ({
    error,
    traces,
    disableTokenColumn = false,
    loading = false,
  }: Partial<TracesViewTableProps> = {}) => {
    return renderWithIntl(
      <TracesViewTable
        error={error}
        traces={traces || []}
        hasNextPage={false}
        hasPreviousPage={false}
        loading={loading}
        onNextPage={() => {}}
        onPreviousPage={() => {}}
        onResetFilters={() => {}}
        sorting={[]}
        setSorting={() => {}}
        rowSelection={{}}
        setRowSelection={() => {}}
        disableTokenColumn={disableTokenColumn}
      />,
    );
  };

  test('renders the table with traces', () => {
    const mockTraces = new Array(12).fill(0).map((_, i) => generateMockTrace(`trace-${i + 1}`, i));
    renderTestComponent({ traces: mockTraces });
    expect(screen.getByRole('cell', { name: 'Trace name: trace-1' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: 'Trace name: trace-12' })).toBeInTheDocument();
  });

  test('renders the table with trace full of data', () => {
    const mockTrace = generateMockTrace(
      'trace-test',
      123,
      [
        {
          key: 'mlflow.traceInputs',
          value: 'test-inputs',
        },
        {
          key: 'mlflow.traceOutputs',
          value: 'test-outputs',
        },
        {
          key: 'total_tokens',
          value: '1234',
        },
      ],
      [
        {
          key: 'some-test-tag',
          value: 'value',
        },
        {
          key: 'mlflow.source.type',
          value: 'NOTEBOOK',
        },
        {
          key: 'mlflow.source.name',
          value: '/Users/test@databricks.com/test-notebook',
        },
        {
          key: 'mlflow.databricks.notebookID',
          value: 'test-id',
        },
      ],
    );
    renderTestComponent({ traces: [mockTrace] });
    expect(screen.getByRole('cell', { name: 'Trace name: trace-test' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: 'test-inputs' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: 'test-outputs' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: '1234' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: /some-test-tag/ })).toBeInTheDocument();
    expect(screen.getByText('test-notebook')).toBeInTheDocument();
  });

  test('renders the table with trace when no tokens are present', () => {
    const mockTraces = new Array(12).fill(0).map((_, i) => generateMockTrace(`trace-${i + 1}`, i));

    renderTestComponent({ traces: mockTraces, disableTokenColumn: true });
    expect(screen.queryByRole('columnheader', { name: 'Tokens' })).not.toBeInTheDocument();
  });

  test('renders the table with error', () => {
    renderTestComponent({ traces: [], error: new Error('Test error') });
    expect(screen.getByText('Test error')).toBeInTheDocument();
  });

  test('renders the table with empty state', () => {
    renderTestComponent({ traces: [] });
    expect(screen.getByText('No traces recorded')).toBeInTheDocument();
  });
});
