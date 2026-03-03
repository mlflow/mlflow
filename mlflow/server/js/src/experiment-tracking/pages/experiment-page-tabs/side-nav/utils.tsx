import { ExperimentPageTabName } from '../../../constants';

export const isTracesRelatedTab = (tabName: ExperimentPageTabName) => {
  return (
    tabName === ExperimentPageTabName.Overview ||
    tabName === ExperimentPageTabName.Traces ||
    tabName === ExperimentPageTabName.SingleChatSession ||
    tabName === ExperimentPageTabName.ChatSessions ||
    tabName === ExperimentPageTabName.Judges ||
    tabName === ExperimentPageTabName.Datasets ||
    tabName === ExperimentPageTabName.EvaluationRuns ||
    tabName === ExperimentPageTabName.Prompts ||
    tabName === ExperimentPageTabName.Models
  );
};
