import { Button, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { useDeleteLabelSchemaMutation } from '../../components/label-schemas/hooks/useDeleteLabelSchemaMutation';
import type { LabelSchema } from '../../components/label-schemas/types';

export interface DeleteLabelSchemaModalProps {
  schema: LabelSchema | null;
  onClose: () => void;
}

export const DeleteLabelSchemaModal = ({ schema, onClose }: DeleteLabelSchemaModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { deleteLabelSchemaAsync, isDeleting, error } = useDeleteLabelSchemaMutation();

  const handleConfirm = async () => {
    if (schema == null) {
      return;
    }
    try {
      await deleteLabelSchemaAsync({ schema_id: schema.schema_id });
      onClose();
    } catch {
      // The mutation hook surfaces the error via `error` below; keep the
      // modal open so the user sees what went wrong rather than the
      // schema staying in the list silently.
    }
  };

  return (
    <Modal
      componentId="mlflow.experiment-label-schemas.delete-modal"
      visible={schema != null}
      title={intl.formatMessage({
        defaultMessage: 'Delete label schema',
        description: 'Delete label schema modal title',
      })}
      onCancel={onClose}
      footer={
        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          <Button componentId="mlflow.experiment-label-schemas.delete-modal.cancel" onClick={onClose}>
            <FormattedMessage defaultMessage="Cancel" description="Cancel delete button" />
          </Button>
          <Button
            componentId="mlflow.experiment-label-schemas.delete-modal.confirm"
            danger
            loading={isDeleting}
            onClick={handleConfirm}
          >
            <FormattedMessage defaultMessage="Delete" description="Confirm delete button" />
          </Button>
        </div>
      }
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        <Typography.Text>
          <FormattedMessage
            defaultMessage="Are you sure you want to delete the label schema {name}? Assessments already collected under this schema are not removed and will render as free-form values in the review UI."
            description="Delete label schema confirmation prompt"
            values={{ name: <Typography.Text bold>{schema?.name}</Typography.Text> }}
          />
        </Typography.Text>
        {error && (
          <Typography.Text color="error">
            {error.message ??
              intl.formatMessage({
                defaultMessage: 'Failed to delete label schema.',
                description: 'Generic delete label schema error',
              })}
          </Typography.Text>
        )}
      </div>
    </Modal>
  );
};
