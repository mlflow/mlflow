import { useState, useEffect } from 'react';
import { Modal, Typography, Button, Input, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

interface KeyValueTagFullViewModalProps {
  tagKey: string;
  tagValue: string;
  isKeyValueTagFullViewModalVisible: boolean;
  setIsKeyValueTagFullViewModalVisible: (visible: boolean) => void;
  onSave?: (key: string, value: string) => Promise<void>;
  onDelete?: (key: string) => Promise<void>;
  isEditMode?: boolean;
}

export const KeyValueTagFullViewModal = ({
  tagKey,
  tagValue,
  isKeyValueTagFullViewModalVisible,
  setIsKeyValueTagFullViewModalVisible,
  onSave,
  onDelete,
  isEditMode = true, // Default to edit mode for adding tags
}: KeyValueTagFullViewModalProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [key, setKey] = useState(tagKey);
  const [value, setValue] = useState(tagValue);
  const [isLoading, setIsLoading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  // Update local state when props change
  useEffect(() => {
    setKey(tagKey);
    setValue(tagValue);
  }, [tagKey, tagValue]);

  const handleClose = () => {
    setIsKeyValueTagFullViewModalVisible(false);
    // Reset to original values
    setKey(tagKey);
    setValue(tagValue);
    setIsLoading(false);
    setIsDeleting(false);
  };

  const isValid = key.trim().length > 0 && value.trim().length > 0;

  const handleSave = async () => {
    if (!isValid || !onSave) return;

    setIsLoading(true);
    try {
      await onSave(key.trim(), value.trim());
      // Only close on successful save
      setIsKeyValueTagFullViewModalVisible(false);
    } catch (error) {
      // Keep modal open on error, let error be displayed by parent
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!onDelete || !tagKey) return;

    setIsDeleting(true);
    try {
      await onDelete(tagKey);
      // Only close on successful delete
      setIsKeyValueTagFullViewModalVisible(false);
    } catch (error) {
      // Keep modal open on error, let error be displayed by parent
    } finally {
      setIsDeleting(false);
    }
  };

  const isAddingNewTag = !tagKey && !tagValue;

  return (
    <Modal
      componentId="mlflow.eval-datasets.key-value-tag-modal"
      visible={isKeyValueTagFullViewModalVisible}
      onCancel={handleClose}
      title={
        isAddingNewTag
          ? intl.formatMessage({
              defaultMessage: 'Add Tag',
              description: 'Modal title for adding a new tag',
            })
          : intl.formatMessage({
              defaultMessage: 'Edit Tag',
              description: 'Modal title for editing an existing tag',
            })
      }
      footer={
        isEditMode
          ? [
              <Button
                key="save"
                type="primary"
                onClick={handleSave}
                disabled={!isValid || isLoading}
                loading={isLoading}
                componentId="mlflow.eval-datasets.key-value-tag-modal-save"
              >
                {isAddingNewTag
                  ? intl.formatMessage({
                      defaultMessage: 'Add Tag',
                      description: 'Save button text for adding a new tag',
                    })
                  : intl.formatMessage({
                      defaultMessage: 'Save Changes',
                      description: 'Save button text for editing an existing tag',
                    })}
              </Button>,
              ...(onDelete && !isAddingNewTag
                ? [
                    <Button
                      key="delete"
                      danger
                      onClick={handleDelete}
                      disabled={isLoading || isDeleting}
                      loading={isDeleting}
                      componentId="mlflow.eval-datasets.key-value-tag-modal-delete"
                    >
                      {intl.formatMessage({
                        defaultMessage: 'Delete',
                        description: 'Delete button for tag modal',
                      })}
                    </Button>,
                  ]
                : []),
            ]
          : [
              <Button key="close" onClick={handleClose} componentId="mlflow.eval-datasets.key-value-tag-modal-close">
                {intl.formatMessage({
                  defaultMessage: 'Close',
                  description: 'Close button for tag details modal',
                })}
              </Button>,
            ]
      }
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div>
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.sm }}>
            {intl.formatMessage({
              defaultMessage: 'Key:',
              description: 'Label for tag key in modal',
            })}
          </Typography.Text>
          {isEditMode ? (
            <Input
              value={key}
              onChange={(e) => setKey(e.target.value)}
              placeholder="Enter tag key"
              componentId="mlflow.eval-datasets.tag-key-input"
              disabled={isLoading || isDeleting}
            />
          ) : (
            <Typography.Text css={{ fontFamily: 'monospace' }}>{tagKey}</Typography.Text>
          )}
        </div>
        <div>
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.sm }}>
            {intl.formatMessage({
              defaultMessage: 'Value:',
              description: 'Label for tag value in modal',
            })}
          </Typography.Text>
          {isEditMode ? (
            <Input
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder="Enter tag value"
              componentId="mlflow.eval-datasets.tag-value-input"
              disabled={isLoading || isDeleting}
            />
          ) : (
            <Typography.Text css={{ fontFamily: 'monospace', wordBreak: 'break-word' }}>{tagValue}</Typography.Text>
          )}
        </div>
      </div>
    </Modal>
  );
};
