import type { CellContext, ColumnDefTemplate } from '@tanstack/react-table';
import { ModelTraceInfoWithRunName } from './hooks/useExperimentTraces';
import { keyBy } from 'lodash';
import { SourceCellRenderer } from '../runs/cells/SourceCellRenderer';

export const ExperimentViewTracesTableNotebookCell: ColumnDefTemplate<
  CellContext<ModelTraceInfoWithRunName, unknown>
> = ({ row: { original } }) => <SourceCellRenderer value={keyBy(original.tags, 'key')} />;
