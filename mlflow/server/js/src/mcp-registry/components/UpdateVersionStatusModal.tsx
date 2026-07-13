import { useEffect, useState } from 'react';
import {
  Alert,
  Modal,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { MCPStatus } from '../types';
import { STATUS_TRANSITIONS } from '../utils';
import { flexColumnGapStyles } from '../styles';

const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export const UpdateVersionStatusModal = ({
  visible,
  currentStatus,
  isLoading,
  error,
  onUpdate,
  onCancel,
}: {
  visible: boolean;
  currentStatus: MCPStatus;
  isLoading?: boolean;
  error?: Error | null;
  onUpdate: (newStatus: MCPStatus) => void;
  onCancel: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const allowedTransitions = STATUS_TRANSITIONS[currentStatus] ?? [];
  const isTerminal = allowedTransitions.length === 0;
  const [selectedStatus, setSelectedStatus] = useState<MCPStatus | undefined>(allowedTransitions[0]);

  useEffect(() => {
    if (visible) {
      const transitions = STATUS_TRANSITIONS[currentStatus] ?? [];
      setSelectedStatus(transitions[0]);
    }
  }, [visible, currentStatus]);

  return (
    <Modal
      componentId="mlflow.mcp_registry.detail.update_status_modal"
      title={
        <FormattedMessage
          defaultMessage="Update version status"
          description="MCP server update version status modal title"
        />
      }
      visible={visible}
      onCancel={onCancel}
      onOk={() => selectedStatus && onUpdate(selectedStatus)}
      okText={intl.formatMessage({
        defaultMessage: 'Update',
        description: 'MCP server update version status modal confirm button',
      })}
      confirmLoading={isLoading}
      okButtonProps={{ disabled: isTerminal || !selectedStatus }}
    >
      <div css={flexColumnGapStyles(theme, theme.spacing.md)}>
        {error && (
          <Alert
            componentId="mlflow.mcp_registry.detail.update_status_error"
            type="error"
            message={error.message}
            closable={false}
          />
        )}
        {isTerminal && (
          <Alert
            componentId="mlflow.mcp_registry.detail.update_status_terminal"
            type="warning"
            message={intl.formatMessage({
              defaultMessage: 'This version is in a terminal state and cannot be transitioned.',
              description: 'MCP server terminal status warning',
            })}
            closable={false}
          />
        )}
        {!isTerminal && (
          <>
            <div css={flexColumnGapStyles(theme, theme.spacing.xs)}>
              <Typography.Text bold>
                <FormattedMessage
                  defaultMessage="Current status:"
                  description="MCP server current status label in update modal"
                />
              </Typography.Text>
              <Typography.Text color="secondary">{capitalize(currentStatus)}</Typography.Text>
            </div>
            <div css={flexColumnGapStyles(theme, theme.spacing.xs)}>
              <Typography.Text bold>
                <FormattedMessage
                  defaultMessage="New status:"
                  description="MCP server new status label in update modal"
                />
              </Typography.Text>
              <SimpleSelect
                id="mcp-registry-update-version-status"
                componentId="mlflow.mcp_registry.detail.status_select"
                value={selectedStatus}
                onChange={({ target }) => setSelectedStatus(target.value as MCPStatus)}
              >
                {allowedTransitions.map((status) => (
                  <SimpleSelectOption key={status} value={status}>
                    {capitalize(status)}
                  </SimpleSelectOption>
                ))}
              </SimpleSelect>
            </div>
          </>
        )}
      </div>
    </Modal>
  );
};
