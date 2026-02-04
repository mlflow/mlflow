import { ExperimentPageTabName } from '../../../constants';

export const isTracesRelatedTab = (tabName: ExperimentPageTabName) => {
  return (
    tabName === ExperimentPageTabName.Overview ||
    tabName === ExperimentPageTabName.Traces ||
    tabName === ExperimentPageTabName.SingleChatSession ||
    tabName === ExperimentPageTabName.ChatSessions
  );
};
