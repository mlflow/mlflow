import type { CellContext, ColumnDefTemplate } from '@tanstack/react-table';
import type { ModelTraceInfoWithRunName } from './hooks/useExperimentTraces';
import { keyBy } from 'lodash';
import { SourceCellRenderer } from '../experiment-page/components/runs/cells/SourceCellRenderer';

export const TracesViewTableSourceCell: ColumnDefTemplate<CellContext<ModelTraceInfoWithRunName, unknown>> = ({
  row: { original },
}) => <SourceCellRenderer value={keyBy(original.tags, 'key')} />;
