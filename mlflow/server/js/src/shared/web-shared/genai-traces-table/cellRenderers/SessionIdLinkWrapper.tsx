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
  const url = traceId
    ? `${MlflowUtils.getExperimentChatSessionPageRoute(experimentId, sessionId)}?selectedTraceId=${encodeURIComponent(
        traceId,
      )}`
    : MlflowUtils.getExperimentChatSessionPageRoute(experimentId, sessionId);

  return (
    <Link
      // prettier-ignore
      to={url}
    >
      {children}
    </Link>
  );
};
