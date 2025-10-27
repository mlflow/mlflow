import { type ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';
import { screen } from '@testing-library/react';
import type { TracesViewTableProps } from './TracesViewTable';
import { TracesViewTable } from './TracesViewTable';
import { renderWithIntl } from '../../../common/utils/TestUtils.react18';
import type { KeyValueEntity } from '../../../common/types';
import userEvent from '@testing-library/user-event';
import { ExperimentViewTracesTableColumns } from './TracesView.utils';
import { DesignSystemProvider } from '@databricks/design-system';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
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
  const mockToggleHiddenColumn = jest.fn();
  const renderTestComponent = ({
    error,
    traces,
    disableTokenColumn = false,
    loading = false,
    hiddenColumns = [],
    usingFilters = false,
  }: Partial<TracesViewTableProps> = {}) => {
    return renderWithIntl(
      <DesignSystemProvider>
        <TracesViewTable
          experimentIds={['123']}
          hiddenColumns={hiddenColumns}
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
          baseComponentId="test"
          toggleHiddenColumn={mockToggleHiddenColumn}
          usingFilters={usingFilters}
        />
      </DesignSystemProvider>,
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

  test('renders the quickstart when no traces are present', () => {
    renderTestComponent({ traces: [] });
    expect(document.body.textContent).not.toBe('');
  });

  test('renders the empty message when using filters with no results', () => {
    renderTestComponent({ traces: [], usingFilters: true });
    expect(screen.getByText('No traces found')).toBeInTheDocument();
    expect(screen.getByText('Reset filters')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /select columns/i })).toBeInTheDocument();
  });

  describe('Column selector', () => {
    const user = userEvent.setup();
    beforeEach(() => {
      jest.clearAllMocks();
    });

    async function openColumnSelector(hiddenColumns: string[] = []) {
      // Add at least one trace to ensure the table is rendered with the column selector
      const mockTraces = [generateMockTrace('test-trace')];
      renderTestComponent({ traces: mockTraces, hiddenColumns });
      const columnSelectorButton = screen.getByRole('button', { name: /select columns/i });
      await user.click(columnSelectorButton);
    }

    it('renders the column selector dropdown', async () => {
      await openColumnSelector();

      // Verify dropdown content
      expect(screen.getByRole('menu')).toBeInTheDocument();
    });

    it('shows correct checkbox states based on hidden columns', async () => {
      await openColumnSelector([ExperimentViewTracesTableColumns.tags]);

      // Check that the Tags column checkbox is unchecked
      const tagsCheckbox = screen.getByRole('menuitemcheckbox', { name: /tags/i });
      expect(tagsCheckbox).toHaveAttribute('aria-checked', 'false');

      // Check that other column checkboxes are checked
      expect(screen.getByRole('menuitemcheckbox', { name: /request id/i })).toHaveAttribute('aria-checked', 'true');
    });

    it('calls toggleHiddenColumn when a checkbox is clicked', async () => {
      await openColumnSelector([ExperimentViewTracesTableColumns.tags]);

      // Click the Tags column checkbox
      const tagsCheckbox = screen.getByRole('menuitemcheckbox', { name: /tags/i });
      await user.click(tagsCheckbox);

      // Verify that toggleHiddenColumn was called with the correct column ID
      expect(mockToggleHiddenColumn).toHaveBeenCalledWith(ExperimentViewTracesTableColumns.tags);
    });
  });
});
