import { useCallback, useMemo, useRef, useState } from 'react';
import { Notification } from '@databricks/design-system';
import { useIntl } from 'react-intl';

export interface TracesV4NotifyApi {
  success: (message: React.ReactNode) => void;
  error: (error: unknown, fallbackTitle?: React.ReactNode) => void;
}

interface UseTracesV4NotificationsResult {
  notify: TracesV4NotifyApi;
  /** Render somewhere stable in the page tree — the toast Roots mount here. */
  notificationContainer: React.ReactElement;
}

type ToastSeverity = 'success' | 'error';

interface ToastItem {
  key: number;
  severity: ToastSeverity;
  title: React.ReactNode;
  description?: React.ReactNode;
}

/**
 * Imperative toast wrapper around the declarative `Notification.Root` API, siloed for traces-v4.
 * Mirrors the datasets-v2 `useDatasetNotifications` queue pattern. Requires a `Notification.Provider`
 * + `Notification.Viewport` ancestor (mounted by `TracesV4PageWrapper`).
 */
export const useTracesV4Notifications = (): UseTracesV4NotificationsResult => {
  const intl = useIntl();
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const keyCounter = useRef(0);

  const remove = useCallback((key: number) => {
    setToasts((prev) => prev.filter((t) => t.key !== key));
  }, []);

  const notify = useMemo<TracesV4NotifyApi>(
    () => ({
      success: (message) => {
        const key = keyCounter.current;
        keyCounter.current += 1;
        setToasts((prev) => [...prev, { key, severity: 'success', title: message }]);
      },
      error: (error, fallbackTitle) => {
        const key = keyCounter.current;
        keyCounter.current += 1;
        const title =
          fallbackTitle ??
          intl.formatMessage({
            defaultMessage: 'Something went wrong',
            description: 'Generic title used when a trace action surfaces an unexpected error',
          });
        const description = error instanceof Error ? error.message : typeof error === 'string' ? error : undefined;
        setToasts((prev) => [...prev, { key, severity: 'error', title, description }]);
      },
    }),
    [intl],
  );

  const closeAriaLabel = intl.formatMessage({
    defaultMessage: 'Close notification',
    description: 'Aria label for the close button on V4 traces notifications',
  });

  const notificationContainer = useMemo(
    () => (
      <>
        {toasts.map((toast) => (
          <Notification.Root
            key={toast.key}
            componentId={
              toast.severity === 'success'
                ? 'mlflow.traces-v4.notifications.success'
                : 'mlflow.traces-v4.notifications.error'
            }
            severity={toast.severity}
            open
            onOpenChange={(next) => {
              if (!next) remove(toast.key);
            }}
          >
            <Notification.Title>{toast.title}</Notification.Title>
            {toast.description !== undefined && toast.description !== null && (
              <Notification.Description>{toast.description}</Notification.Description>
            )}
            <Notification.Close componentId="mlflow.traces-v4.notifications.close" closeLabel={closeAriaLabel} />
          </Notification.Root>
        ))}
      </>
    ),
    [toasts, closeAriaLabel, remove],
  );

  return { notify, notificationContainer };
};
