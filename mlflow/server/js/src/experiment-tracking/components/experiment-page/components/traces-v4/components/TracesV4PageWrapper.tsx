import { ErrorBoundary } from 'react-error-boundary';
import { Notification } from '@databricks/design-system';
import { TracesV3GenericErrorState } from '../../traces-v3/TracesV3GenericErrorState';

/**
 * Error-boundary + notification host for the V4 traces tab. An error boundary falls back to a
 * generic error state if rendering fails, and `Notification.Provider`/`Viewport` back the toast
 * queue used by the delete flow. `resetKey` (the experiment id) resets the boundary so navigating
 * between experiments recovers cleanly from a prior render error.
 */
export const TracesV4PageWrapper = ({ children, resetKey }: { children: React.ReactNode; resetKey?: unknown }) => {
  return (
    <ErrorBoundary fallback={<TracesV3GenericErrorState />} resetKeys={[resetKey]}>
      <Notification.Provider>
        {children}
        <Notification.Viewport />
      </Notification.Provider>
    </ErrorBoundary>
  );
};
