import { ErrorBoundary } from 'react-error-boundary';
import { TracesV3GenericErrorState } from './TracesV3GenericErrorState';

export const TracesV3PageWrapper = ({ children }: { children: React.ReactNode }) => {
  return <ErrorBoundary fallback={<TracesV3GenericErrorState />}>{children}</ErrorBoundary>;
};
