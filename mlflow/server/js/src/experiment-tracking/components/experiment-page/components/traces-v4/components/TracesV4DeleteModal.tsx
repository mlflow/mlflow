import { Alert, DangerModal, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

interface TracesV4DeleteModalProps {
  open: boolean;
  count: number;
  isLoading: boolean;
  error?: Error;
  onConfirm: () => void;
  onCancel: () => void;
}

/** Confirmation modal for bulk-deleting selected traces. Mirrors the datasets-v2 delete modal. */
export const TracesV4DeleteModal = ({
  open,
  count,
  isLoading,
  error,
  onConfirm,
  onCancel,
}: TracesV4DeleteModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <DangerModal
      componentId="mlflow.traces-v4.bulk-delete-confirm-modal"
      visible={open}
      title={
        <FormattedMessage
          defaultMessage="Delete traces"
          description="Title for the V4 traces bulk-delete confirmation modal"
        />
      }
      okText={intl.formatMessage({
        defaultMessage: 'Delete',
        description: 'Confirm-button text for the V4 traces bulk-delete modal',
      })}
      cancelText={intl.formatMessage({
        defaultMessage: 'Cancel',
        description: 'Cancel-button text for the V4 traces bulk-delete modal',
      })}
      okButtonProps={{ loading: isLoading }}
      cancelButtonProps={{ disabled: isLoading }}
      onOk={onConfirm}
      onCancel={onCancel}
    >
      {/* Wrapped so the body + optional error alert are a single child (no array-key warning). */}
      <div>
        <FormattedMessage
          defaultMessage="Are you sure you want to delete {count, plural, one {# trace} other {# traces}}? This action cannot be undone."
          description="Body for the V4 traces bulk-delete confirmation modal"
          values={{ count }}
        />
        {error && (
          <Alert
            componentId="mlflow.traces-v4.bulk-delete-error"
            type="error"
            message={error.message}
            css={{ marginTop: theme.spacing.sm }}
            closable={false}
          />
        )}
      </div>
    </DangerModal>
  );
};
