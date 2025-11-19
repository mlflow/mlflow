import type { ColumnDef } from '@tanstack/react-table';

import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

export type SessionTableRow = {
  sessionId: string;
  requestPreview?: string;
  firstTrace: ModelTraceInfoV3;
  experimentId: string;
  sessionStartTime: string;
  sessionDuration: string | null;
  tokens: number;
  turns: number;
};

export type SessionTableColumn = {
  id: string;
  header: string;
  accessorKey?: string;
  defaultVisibility: boolean;
} & ColumnDef<SessionTableRow>;
