import { SELECTED_TRACE_ID_QUERY_PARAM } from '@databricks/web-shared/model-trace-explorer';

import MlflowUtils from '../utils/MlflowUtils';
import { Link } from '../utils/RoutingUtils';

export const SessionIdLinkWrapper = ({
  sessionId,
  experimentId,
  traceId,
  children,
}: {
  sessionId: string;
  experimentId: string;
  traceId?: string;
  children: React.ReactElement;
}) => {
  const baseUrl = MlflowUtils.getExperimentChatSessionPageRoute(experimentId, sessionId);
  const url = traceId
    ? `${baseUrl}?${new URLSearchParams({ [SELECTED_TRACE_ID_QUERY_PARAM]: traceId }).toString()}`
    : baseUrl;

  return (
    <Link
      // prettier-ignore
      to={url}
    >
      {children}
    </Link>
  );
};
