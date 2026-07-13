import { useEffect, useState } from 'react';
import { Alert, Input, Modal, useDesignSystemTheme } from '@databricks/design-system';
import { flexColumnGapStyles } from '../styles';
import { FormattedMessage, useIntl } from 'react-intl';

export const UpdateVersionDisplayNameModal = ({
  visible,
  currentDisplayName,
  isLoading,
  error,
  onUpdate,
  onCancel,
}: {
  visible: boolean;
  currentDisplayName: string;
  isLoading?: boolean;
  error?: Error | null;
  onUpdate: (displayName: string) => void;
  onCancel: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [draft, setDraft] = useState(currentDisplayName);

  useEffect(() => {
    if (visible) {
      setDraft(currentDisplayName);
    }
  }, [visible, currentDisplayName]);

  return (
    <Modal
      componentId="mlflow.mcp_registry.detail.update_display_name_modal"
      title={
        <FormattedMessage
          defaultMessage="Edit display name"
          description="MCP server update version display name modal title"
        />
      }
      visible={visible}
      onCancel={onCancel}
      onOk={() => onUpdate(draft.trim())}
      okText={intl.formatMessage({
        defaultMessage: 'Save',
        description: 'MCP server update version display name modal confirm button',
      })}
      confirmLoading={isLoading}
    >
      <div css={flexColumnGapStyles(theme, theme.spacing.md)}>
        {error && (
          <Alert
            componentId="mlflow.mcp_registry.detail.update_display_name_error"
            type="error"
            message={error.message}
            closable={false}
          />
        )}
        <Input
          componentId="mlflow.mcp_registry.detail.display_name_input"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          aria-label={intl.formatMessage({
            defaultMessage: 'Display name',
            description: 'Aria label for display name input',
          })}
          placeholder={intl.formatMessage({
            defaultMessage: 'Enter display name',
            description: 'Placeholder for version display name input',
          })}
        />
      </div>
    </Modal>
  );
};
