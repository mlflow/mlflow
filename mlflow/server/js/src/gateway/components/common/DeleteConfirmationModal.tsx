import { useState, useEffect } from 'react';
import { Alert, Button, Input, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

export interface DeleteConfirmationModalProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void>;
  title: string;
  itemName: string;
  itemType: string;
  componentIdPrefix: string;
  warningMessage?: React.ReactNode;
  requireConfirmation?: boolean;
}

export const DeleteConfirmationModal = ({
  open,
  onClose,
  onConfirm,
  title,
  itemName,
  itemType,
  componentIdPrefix,
  warningMessage,
  requireConfirmation = false,
}: DeleteConfirmationModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [confirmationText, setConfirmationText] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const isConfirmed = !requireConfirmation || confirmationText === itemName;

  useEffect(() => {
    if (!open) {
      setConfirmationText('');
      setError(null);
    }
  }, [open]);

  const handleDelete = async () => {
    if (!isConfirmed) return;

    setError(null);
    setIsDeleting(true);

    try {
      await onConfirm();
      handleClose();
    } catch (err) {
      setError(
        intl.formatMessage(
          {
            defaultMessage: 'Failed to delete {itemType}. Please try again.',
            description: 'Error message when deletion fails',
          },
          { itemType },
        ),
      );
    } finally {
      setIsDeleting(false);
    }
  };

  const handleClose = () => {
    setConfirmationText('');
    setError(null);
    onClose();
  };

  return (
    <Modal
      componentId={`${componentIdPrefix}.modal`}
      title={title}
      visible={open}
      onCancel={handleClose}
      footer={
        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          <Button componentId={`${componentIdPrefix}.cancel`} onClick={handleClose} disabled={isDeleting}>
            <FormattedMessage defaultMessage="Cancel" description="Cancel button text" />
          </Button>
          <Button
            componentId={`${componentIdPrefix}.delete`}
            type="primary"
            danger
            onClick={handleDelete}
            disabled={!isConfirmed || isDeleting}
            loading={isDeleting}
          >
            <FormattedMessage defaultMessage="Delete" description="Delete button text" />
          </Button>
        </div>
      }
      size="normal"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {error && <Alert componentId={`${componentIdPrefix}.error`} type="error" message={error} closable={false} />}

        <Typography.Text>
          <FormattedMessage
            defaultMessage='Are you sure you want to delete the {itemType} "{itemName}"?'
            description="Delete confirmation message"
            values={{
              itemType,
              itemName: <strong>{itemName}</strong>,
            }}
          />
        </Typography.Text>

        {warningMessage && (
          <Alert
            componentId={`${componentIdPrefix}.warning`}
            type="warning"
            message={warningMessage}
            closable={false}
          />
        )}

        {requireConfirmation && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <Typography.Text>
              <FormattedMessage
                defaultMessage="Type {itemName} to confirm deletion:"
                description="Type to confirm instruction"
                values={{ itemName: <strong>{itemName}</strong> }}
              />
            </Typography.Text>
            <Input
              componentId={`${componentIdPrefix}.confirmation-input`}
              value={confirmationText}
              onChange={(e) => setConfirmationText(e.target.value)}
              placeholder={itemName}
              disabled={isDeleting}
            />
          </div>
        )}
      </div>
    </Modal>
  );
};
