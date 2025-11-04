import { ColumnDef, Row } from '@tanstack/react-table';
import { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

export type SessionTableRow = {
  sessionId: string;
  requestPreview?: string;
  firstTrace: ModelTraceInfoV3;
  experimentId: string;
  sessionStartTime: string;
  sessionDuration: string | null;
};

export type SessionTableColumn = {
  id: string;
  header: string;
  accessorKey?: string;
  cell?: React.ComponentType<{ row: Row<SessionTableRow> }>;
  defaultVisibility: boolean;
} & ColumnDef<SessionTableRow>;
