import { ExperimentPageTabName } from '../../../constants';

const TIME_RANGE_PARAMS = ['startTimeLabel', 'startTime', 'endTime'];

export const isTracesRelatedTab = (tabName: ExperimentPageTabName) => {
  return (
    tabName === ExperimentPageTabName.Overview ||
    tabName === ExperimentPageTabName.Traces ||
    tabName === ExperimentPageTabName.SingleChatSession ||
    tabName === ExperimentPageTabName.ChatSessions
  );
};

export const getTimeRangeQueryString = (search: string): string | undefined => {
  const searchParams = new URLSearchParams(search);
  const timeRangeParams = new URLSearchParams();

  for (const param of TIME_RANGE_PARAMS) {
    const value = searchParams.get(param);
    if (value) {
      timeRangeParams.set(param, value);
    }
  }

  const result = timeRangeParams.toString();
  return result ? `?${result}` : undefined;
};
