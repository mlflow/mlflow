import type { Row } from '@tanstack/react-table';

import { Tag, Typography } from '@databricks/design-system';

import MlflowUtils from '../../utils/MlflowUtils';
import { Link } from '../../utils/RoutingUtils';
import type { SessionTableRow } from '../utils';

export const SessionIdCellRenderer = ({ row }: { row: Row<SessionTableRow> }) => {
  const experimentId = row.original.experimentId;
  const sessionId = row.original.sessionId;

  return (
    <Link to={MlflowUtils.getExperimentChatSessionPageRoute(experimentId, sessionId)}>
      <Tag
        componentId="mlflow.chat-sessions.session-id-tag"
        color="indigo"
        css={{ maxWidth: '100%', overflow: 'hidden' }}
      >
        <Typography.Text ellipsis>{sessionId}</Typography.Text>
      </Tag>
    </Link>
  );
};
