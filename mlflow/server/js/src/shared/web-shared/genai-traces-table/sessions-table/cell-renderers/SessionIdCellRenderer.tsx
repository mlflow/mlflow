import type { Row } from '@tanstack/react-table';

import { Tag, Typography } from '@databricks/design-system';

import type { SessionTableRow } from '../types';

export const SessionIdCellRenderer = ({ row }: { row: Row<SessionTableRow> }) => {
  const sessionId = row.original.sessionId;

  return (
    <Tag
      componentId="mlflow.chat-sessions.session-id-tag"
      color="indigo"
      css={{ maxWidth: '100%', overflow: 'hidden', cursor: 'pointer' }}
    >
      <Typography.Text ellipsis>{sessionId}</Typography.Text>
    </Tag>
  );
};
