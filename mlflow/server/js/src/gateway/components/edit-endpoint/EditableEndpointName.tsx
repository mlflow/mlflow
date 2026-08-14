import { useState } from 'react';
import {
  Alert,
  Button,
  Input,
  Modal,
  PencilIcon,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { isValidEndpointName } from '../../utils/gatewayUtils';
import type { Endpoint } from '../../types';

interface EditableEndpointNameProps {
  endpoint: Endpoint | undefined;
  existingEndpoints: Endpoint[] | undefined;
  onNameUpdate: (newName: string) => Promise<void>;
  isSubmitting?: boolean;
}

export const EditableEndpointName = ({
  endpoint,
  existingEndpoints,
  onNameUpdate,
  isSubmitting = false,
}: EditableEndpointNameProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [newName, setNewName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isUpdating, setIsUpdating] = useState(false);

  const handleOpenModal = () => {
    setNewName(endpoint?.name ?? '');
    setError(null);
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setNewName('');
    setError(null);
  };

  const validateName = (name: string): string | null => {
    if (!name.trim()) {
      return intl.formatMessage({
        defaultMessage: 'Endpoint name is required',
        description: 'Error message when endpoint name is empty',
      });
    }

    if (!isValidEndpointName(name)) {
      return intl.formatMessage({
        defaultMessage:
          'Name can only contain letters, numbers, underscores, hyphens, and dots. Spaces and special characters are not allowed.',
        description: 'Error message for invalid endpoint name format',
      });
    }

    const otherEndpoints = existingEndpoints?.filter((e) => e.endpoint_id !== endpoint?.endpoint_id);
    if (otherEndpoints?.some((e) => e.name === name)) {
      return intl.formatMessage({
        defaultMessage: 'An endpoint with this name already exists',
        description: 'Error message when endpoint name already exists',
      });
    }

    return null;
  };

  const handleSubmit = async () => {
    const validationError = validateName(newName);
    if (validationError) {
      setError(validationError);
      return;
    }

    if (newName === endpoint?.name) {
      handleCloseModal();
      return;
    }

    setIsUpdating(true);
    try {
      await onNameUpdate(newName);
      handleCloseModal();
    } catch (e: any) {
      setError(e.message || 'Failed to update endpoint name');
    } finally {
      setIsUpdating(false);
    }
  };

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setNewName(e.target.value);
    if (error) {
      setError(null);
    }
  };

  return (
    <>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <Typography.Title level={2} withoutMargins>
          {endpoint?.name}
        </Typography.Title>
        <Tooltip
          componentId="mlflow.gateway.edit-endpoint.name-edit-tooltip"
          content={intl.formatMessage({
            defaultMessage: 'Edit endpoint name',
            description: 'Tooltip for edit endpoint name button',
          })}
        >
          <Button
            componentId="mlflow.gateway.edit-endpoint.name-edit-button"
            size="small"
            type="tertiary"
            icon={<PencilIcon />}
            onClick={handleOpenModal}
            disabled={isSubmitting}
            aria-label={intl.formatMessage({
              defaultMessage: 'Edit endpoint name',
              description: 'Aria label for edit endpoint name button',
            })}
          />
        </Tooltip>
      </div>

      <Modal
        componentId="mlflow.gateway.edit-endpoint-name-modal"
        title={intl.formatMessage({
          defaultMessage: 'Edit Endpoint Name',
          description: 'Title for edit endpoint name modal',
        })}
        visible={isModalOpen}
        onCancel={handleCloseModal}
        onOk={handleSubmit}
        okText={intl.formatMessage({
          defaultMessage: 'Save',
          description: 'Save button text for edit endpoint name modal',
        })}
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Cancel button text',
        })}
        confirmLoading={isUpdating}
        okButtonProps={{ disabled: !newName.trim() || newName === endpoint?.name }}
        size="normal"
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {error && (
            <Alert
              componentId="mlflow.gateway.edit-endpoint-name-modal.error"
              type="error"
              message={error}
              closable={false}
            />
          )}

          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <Typography.Text bold>
              <FormattedMessage defaultMessage="Endpoint Name" description="Label for endpoint name input" />
            </Typography.Text>
            <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
              <FormattedMessage
                defaultMessage="The name is used in the endpoint URL. Only letters, numbers, underscores, hyphens, and dots are allowed."
                description="Help text for endpoint name input"
              />
            </Typography.Text>
            <Input
              componentId="mlflow.gateway.edit-endpoint-name-modal.name-input"
              value={newName}
              onChange={handleNameChange}
              placeholder={intl.formatMessage({
                defaultMessage: 'my-endpoint',
                description: 'Placeholder for endpoint name input',
              })}
              validationState={error ? 'error' : undefined}
              autoFocus
            />
          </div>
        </div>
      </Modal>
    </>
  );
};
