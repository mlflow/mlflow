import { Button } from '@databricks/design-system';
import type { Cell, Table } from '@tanstack/react-table';
import type { EvaluationDatasetRecord } from '../types';
import { parseJSONSafe } from '@mlflow/mlflow/src/common/utils/TagUtils';

export const SourceCell = ({
  cell,
  table,
}: {
  cell: Cell<EvaluationDatasetRecord, any>;
  table: Table<EvaluationDatasetRecord>;
}) => {
  const sourceString = cell.getValue();
  const onOpenTraceModal = (table.options.meta as any)?.onOpenTraceModal;

  if (!sourceString) {
    return <span>-</span>;
  }

  let source;
  if (typeof sourceString === 'string') {
    source = parseJSONSafe(sourceString);
  } else {
    source = sourceString;
  }

  if (!source) {
    return <span>-</span>;
  }

  // Handle trace sources
  if (source.source_type === 'TRACE' && source.source_data?.trace_id) {
    const traceId = source.source_data.trace_id;

    return (
      <Button
        type="link"
        size="small"
        componentId="mlflow.eval-datasets.source-trace-link"
        onClick={() => onOpenTraceModal?.(traceId)}
        title={`Trace: ${traceId}`}
      >
        Trace: {traceId}
      </Button>
    );
  }

  return <span>-</span>;
};
