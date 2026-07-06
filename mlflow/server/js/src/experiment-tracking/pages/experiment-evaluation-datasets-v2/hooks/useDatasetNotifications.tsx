import { useCallback, useMemo, useRef, useState } from 'react';
import { Notification } from '@databricks/design-system';
import { useIntl } from 'react-intl';

export interface DatasetNotifyApi {
  /** Pop a green success toast. */
  success: (message: React.ReactNode, description?: React.ReactNode) => void;
  /** Pop a red error toast. Pass an Error to surface its message in the description. */
  error: (error: unknown, fallbackTitle?: React.ReactNode) => void;
}

interface UseDatasetNotificationsResult {
  notify: DatasetNotifyApi;
  /** Render this somewhere stable in the page tree — the toast Roots mount here. */
  notificationContainer: React.ReactElement;
}

type ToastSeverity = 'success' | 'error';

interface ToastItem {
  key: number;
  severity: ToastSeverity;
  title: React.ReactNode;
  description?: React.ReactNode;
}

interface ToastProps {
  componentId: string;
  severity: ToastSeverity;
  open: boolean;
  onClose: () => void;
  title: React.ReactNode;
  description?: React.ReactNode;
  closeLabel: string;
}

const Toast = ({ componentId, severity, open, onClose, title, description, closeLabel }: ToastProps) => (
  <Notification.Root
    componentId={componentId}
    severity={severity}
    open={open}
    onOpenChange={(next) => {
      if (!next) onClose();
    }}
  >
    <Notification.Title>{title}</Notification.Title>
    {description !== undefined && description !== null && (
      <Notification.Description>{description}</Notification.Description>
    )}
    <Notification.Close componentId={`${componentId}.close`} closeLabel={closeLabel} />
  </Notification.Root>
);

/**
 * Imperative wrapper around the declarative `Notification.Root` API. Mirrors the queue
 * pattern used by `web-shared/src/notification/Notification.tsx`'s `NotificationsRpcProvider`
 * so we can keep the call sites' imperative `notify.success(...)` / `notify.error(...)`
 * shape unchanged while moving off the deprecated `useLegacyNotification`.
 *
 * Provider/Viewport are mounted by `ExperimentEvaluationDatasetsPageWrapper`; this hook only
 * needs to render the Roots into the page tree.
 */
export const useDatasetNotifications = (): UseDatasetNotificationsResult => {
  const intl = useIntl();
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const keyCounter = useRef(0);

  const remove = useCallback((key: number) => {
    setToasts((prev) => prev.filter((t) => t.key !== key));
  }, []);

  const notify = useMemo<DatasetNotifyApi>(
    () => ({
      success: (message, description) => {
        const key = keyCounter.current;
        keyCounter.current += 1;
        setToasts((prev) => [...prev, { key, severity: 'success', title: message, description }]);
      },
      error: (error, fallbackTitle) => {
        const key = keyCounter.current;
        keyCounter.current += 1;
        const title =
          fallbackTitle ??
          intl.formatMessage({
            defaultMessage: 'Something went wrong',
            description: 'Generic title used when a dataset action surfaces an unexpected error',
          });
        const description = error instanceof Error ? error.message : typeof error === 'string' ? error : undefined;
        setToasts((prev) => [...prev, { key, severity: 'error', title, description }]);
      },
    }),
    [intl],
  );

  const closeAriaLabel = intl.formatMessage({
    defaultMessage: 'Close notification',
    description: 'Aria label for the close button on V2 evaluation dataset notifications',
  });

  const notificationContainer = useMemo(
    () => (
      <>
        {toasts.map((toast) => (
          <Toast
            key={toast.key}
            componentId={
              toast.severity === 'success'
                ? 'mlflow.eval-datasets-v2.notifications.success'
                : 'mlflow.eval-datasets-v2.notifications.error'
            }
            severity={toast.severity}
            open
            onClose={() => remove(toast.key)}
            title={toast.title}
            description={toast.description}
            closeLabel={closeAriaLabel}
          />
        ))}
      </>
    ),
    [toasts, closeAriaLabel, remove],
  );

  return { notify, notificationContainer };
};
