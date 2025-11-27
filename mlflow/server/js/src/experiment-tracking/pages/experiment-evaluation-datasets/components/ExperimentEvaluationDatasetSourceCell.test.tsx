import { describe, test, expect, jest } from '@jest/globals';
import { render, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { SourceCell } from './ExperimentEvaluationDatasetSourceCell';
import userEvent from '@testing-library/user-event';
import type { Cell, Table } from '@tanstack/react-table';
import type { EvaluationDatasetRecord } from '../types';

describe('ExperimentEvaluationDatasetSourceCell', () => {
  const mockOnOpenTraceModal = jest.fn();

  const createMockCell = (sourceValue?: string): Cell<EvaluationDatasetRecord, any> => {
    return {
      getValue: () => sourceValue,
      row: {
        original: {} as EvaluationDatasetRecord,
      },
    } as any;
  };

  const createMockTable = (): Table<EvaluationDatasetRecord> => {
    return {
      options: {
        meta: {
          onOpenTraceModal: mockOnOpenTraceModal,
        },
      },
    } as any;
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders hyphen when source is undefined', () => {
    const cell = createMockCell(undefined);
    const table = createMockTable();

    render(<SourceCell cell={cell} table={table} />);

    expect(screen.getByText('-')).toBeInTheDocument();
  });

  test('renders hyphen when source is empty string', () => {
    const cell = createMockCell('');
    const table = createMockTable();

    render(<SourceCell cell={cell} table={table} />);

    expect(screen.getByText('-')).toBeInTheDocument();
  });

  test('renders hyphen when source is null', () => {
    const cell = createMockCell(JSON.stringify(null));
    const table = createMockTable();

    render(<SourceCell cell={cell} table={table} />);

    expect(screen.getByText('-')).toBeInTheDocument();
  });

  test('renders hyphen for HUMAN source type', () => {
    const source = {
      source_type: 'HUMAN',
      source_data: {},
    };
    const cell = createMockCell(JSON.stringify(source));
    const table = createMockTable();

    render(<SourceCell cell={cell} table={table} />);

    expect(screen.getByText('-')).toBeInTheDocument();
  });

  test('renders hyphen for DOCUMENT source type', () => {
    const source = {
      source_type: 'DOCUMENT',
      source_data: {},
    };
    const cell = createMockCell(JSON.stringify(source));
    const table = createMockTable();

    render(<SourceCell cell={cell} table={table} />);

    expect(screen.getByText('-')).toBeInTheDocument();
  });

  test('renders hyphen for CODE source type', () => {
    const source = {
      source_type: 'CODE',
      source_data: {},
    };
    const cell = createMockCell(JSON.stringify(source));
    const table = createMockTable();

    render(<SourceCell cell={cell} table={table} />);

    expect(screen.getByText('-')).toBeInTheDocument();
  });

  test('renders clickable trace link for TRACE source type', () => {
    const traceId = 'tr-abc123def456';
    const source = {
      source_type: 'TRACE',
      source_data: {
        trace_id: traceId,
      },
    };
    const cell = createMockCell(JSON.stringify(source));
    const table = createMockTable();

    render(<SourceCell cell={cell} table={table} />);

    // Verify trace link is rendered
    const link = screen.getByRole('button', { name: `Trace: ${traceId}` });
    expect(link).toBeInTheDocument();
    expect(link).toHaveTextContent(`Trace: ${traceId}`);
  });

  test('calls onOpenTraceModal when trace link is clicked', async () => {
    const traceId = 'tr-test-trace-id';
    const source = {
      source_type: 'TRACE',
      source_data: {
        trace_id: traceId,
      },
    };
    const cell = createMockCell(JSON.stringify(source));
    const table = createMockTable();

    render(<SourceCell cell={cell} table={table} />);

    // Click the trace link
    const link = screen.getByRole('button', { name: `Trace: ${traceId}` });
    await userEvent.click(link);

    // Verify the callback was called with the correct trace ID
    expect(mockOnOpenTraceModal).toHaveBeenCalledTimes(1);
    expect(mockOnOpenTraceModal).toHaveBeenCalledWith(traceId);
  });

  test('renders hyphen for TRACE source without trace_id', () => {
    const source = {
      source_type: 'TRACE',
      source_data: {},
    };
    const cell = createMockCell(JSON.stringify(source));
    const table = createMockTable();

    render(<SourceCell cell={cell} table={table} />);

    // Should render hyphen since trace_id is missing
    expect(screen.getByText('-')).toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  test('handles source as object instead of string', () => {
    const traceId = 'tr-object-trace-id';
    const source = {
      source_type: 'TRACE',
      source_data: {
        trace_id: traceId,
      },
    };

    // Pass source as object, not JSON string
    const cell = {
      getValue: () => source,
      row: {
        original: {} as EvaluationDatasetRecord,
      },
    } as any;
    const table = createMockTable();

    render(<SourceCell cell={cell} table={table} />);

    // Should still render trace link
    const link = screen.getByRole('button', { name: `Trace: ${traceId}` });
    expect(link).toBeInTheDocument();
  });

  test('does not call onOpenTraceModal when callback is not provided', async () => {
    const traceId = 'tr-no-callback';
    const source = {
      source_type: 'TRACE',
      source_data: {
        trace_id: traceId,
      },
    };
    const cell = createMockCell(JSON.stringify(source));

    // Table without onOpenTraceModal callback
    const table = {
      options: {
        meta: {},
      },
    } as any;

    render(<SourceCell cell={cell} table={table} />);

    // Link should still render
    const link = screen.getByRole('button', { name: `Trace: ${traceId}` });
    expect(link).toBeInTheDocument();

    // Click should not throw error
    await userEvent.click(link);

    // No callback should have been called
    expect(mockOnOpenTraceModal).not.toHaveBeenCalled();
  });
});
