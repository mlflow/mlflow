import { Row } from '@tanstack/react-table';
import { SourceCellRenderer } from '../../cellRenderers/Source/SourceRenderer';
import { SessionTableRow } from '../types';

export const SessionSourceCellRenderer = ({ row }: { row: Row<SessionTableRow> }) => {
  const firstTrace = row.original.firstTrace;
  return <SourceCellRenderer traceInfo={firstTrace} isComparing={false} />;
};
