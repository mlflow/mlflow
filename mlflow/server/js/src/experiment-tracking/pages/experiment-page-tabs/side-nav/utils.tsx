import { ExperimentPageTabName } from '../../../constants';
import { FormattedMessage } from '@databricks/i18n';

export const isTracesRelatedTab = (tabName: ExperimentPageTabName) => {
  return (
    tabName === ExperimentPageTabName.Overview ||
    tabName === ExperimentPageTabName.Traces ||
    tabName === ExperimentPageTabName.SingleChatSession ||
    tabName === ExperimentPageTabName.ChatSessions
  );
};
