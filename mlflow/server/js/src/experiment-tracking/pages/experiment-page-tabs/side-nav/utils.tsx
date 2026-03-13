import { ExperimentPageTabName } from '../../../constants';

const PRESERVED_QUERY_PARAMS = ['startTimeLabel', 'startTime', 'endTime', 'workflowType', 'workspace'];

export const isTracesRelatedTab = (tabName: ExperimentPageTabName) => {
  return (
    tabName === ExperimentPageTabName.Overview ||
    tabName === ExperimentPageTabName.Traces ||
    tabName === ExperimentPageTabName.SingleChatSession ||
    tabName === ExperimentPageTabName.ChatSessions
  );
};

/**
 * Builds a query string preserving only the params that should carry across
 * experiment tab navigations (time range filters and workflow type).
 */
export const getPreservedQueryString = (search: string): string | undefined => {
  const searchParams = new URLSearchParams(search);
  const preservedParams = new URLSearchParams();

  for (const param of PRESERVED_QUERY_PARAMS) {
    const value = searchParams.get(param);
    if (value) {
      preservedParams.set(param, value);
    }
  }

  const result = preservedParams.toString();
  return result ? `?${result}` : undefined;
};
