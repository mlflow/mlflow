import { Alert, DangerModal, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

interface BulkDeleteRecordsModalProps {
  open: boolean;
  count: number;
  isLoading: boolean;
  error?: Error;
  onConfirm: () => void;
  onCancel: () => void;
}

export const BulkDeleteRecordsModal = ({
  open,
  count,
  isLoading,
  error,
  onConfirm,
  onCancel,
}: BulkDeleteRecordsModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <DangerModal
      componentId="mlflow.eval-datasets-v2.records.bulk-delete-confirm-modal"
      visible={open}
      title={
        <FormattedMessage
          defaultMessage="Delete records"
          description="Title for the V2 dataset records bulk-delete confirmation modal"
        />
      }
      okText={intl.formatMessage({
        defaultMessage: 'Delete',
        description: 'Confirm-button text for the V2 dataset records bulk-delete modal',
      })}
      cancelText={intl.formatMessage({
        defaultMessage: 'Cancel',
        description: 'Cancel-button text for the V2 dataset records bulk-delete modal',
      })}
      okButtonProps={{ loading: isLoading }}
      cancelButtonProps={{ disabled: isLoading }}
      onOk={onConfirm}
      onCancel={onCancel}
    >
      <FormattedMessage
        defaultMessage="Are you sure you want to delete {count, plural, one {# record} other {# records}}? This action cannot be undone."
        description="Body for the V2 dataset records bulk-delete confirmation modal"
        values={{ count }}
      />
      {error && (
        <Alert
          componentId="mlflow.eval-datasets-v2.records.bulk-delete-error"
          type="error"
          message={error.message}
          css={{ marginTop: theme.spacing.sm }}
          closable={false}
        />
      )}
    </DangerModal>
  );
};
