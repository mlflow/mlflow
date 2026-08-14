import { ErrorBoundary } from 'react-error-boundary';
import { ExperimentChatSessionsGenericErrorState } from './ExperimentChatSessionsGenericErrorState';
import { MonitoringConfigProvider } from '../../hooks/useMonitoringConfig';

export const ExperimentChatSessionsPageWrapper = ({ children }: { children: React.ReactNode }) => {
  return <ErrorBoundary fallback={<ExperimentChatSessionsGenericErrorState />}>{children}</ErrorBoundary>;
};
