import type { ReactNode } from 'react';
import { Alert, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';

interface ConfirmationModalProps {
  componentId: string;
  title: string;
  visible: boolean;
  message: ReactNode;
  onConfirm: () => void;
  onCancel: () => void;
  isLoading?: boolean;
  /** When set, renders an inline error Alert above the message. */
  error?: string | null;
  onErrorDismiss?: () => void;
  okText?: string;
  /** Apply the danger style to the OK button (default: ``true`` for delete-style flows). */
  danger?: boolean;
}

/**
 * Thin wrapper around design-system ``Modal`` for
 * "Are you sure?" / delete-style confirmation flows. Shared body
 * layout (optional inline error Alert + message) plus the standard
 * danger-OK button props. Callers control wording, loading state, and
 * the actual confirm/cancel handlers.
 */
export const ConfirmationModal = ({
  componentId,
  title,
  visible,
  message,
  onConfirm,
  onCancel,
  isLoading,
  error,
  onErrorDismiss,
  okText = 'Delete',
  danger = true,
}: ConfirmationModalProps) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Modal
      componentId={componentId}
      title={title}
      visible={visible}
      onCancel={onCancel}
      onOk={onConfirm}
      okText={okText}
      okButtonProps={danger ? { danger: true } : undefined}
      confirmLoading={isLoading}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {error && (
          <Alert
            componentId={`${componentId}.error`}
            type="error"
            message={error}
            closable={Boolean(onErrorDismiss)}
            onClose={onErrorDismiss}
          />
        )}
        <Typography.Text>{message}</Typography.Text>
      </div>
    </Modal>
  );
};
