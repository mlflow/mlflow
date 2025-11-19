import { shouldEnableChatSessionsTab } from '../utils/FeatureUtils';
import MlflowUtils from '../utils/MlflowUtils';
import { Link } from '../utils/RoutingUtils';

export const SessionIdLinkWrapper = ({
  sessionId,
  experimentId,
  children,
}: {
  sessionId: string;
  experimentId: string;
  children: React.ReactElement;
}) => {
  if (shouldEnableChatSessionsTab()) {
    return (
      <Link
        // prettier-ignore
        to={MlflowUtils.getExperimentChatSessionPageRoute(experimentId, sessionId)}
      >
        {children}
      </Link>
    );
  }
  return children;
};
