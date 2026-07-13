import { useState } from 'react';
import {
  Alert,
  Input,
  Modal,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { MCPServer, MCPServerVersion, MCPStatus } from '../types';
import { STATUS_TRANSITIONS, LATEST_ALIAS, RESERVED_ALIASES, validateToolsJson } from '../utils';
import { flexColumnGapStyles, blockLabelStyles, monoFontStyles } from '../styles';
import { AliasSelect } from '../../common/components/AliasSelect';
import { useUpdateMCPServerVersion } from '../hooks/useMCPServerVersionMutations';

export const EditVersionModal = ({
  visible,
  server,
  version,
  aliasesByVersion,
  onClose,
}: {
  visible: boolean;
  server: MCPServer;
  version: MCPServerVersion;
  aliasesByVersion: Record<string, string[]>;
  onClose: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const updateVersionMutation = useUpdateMCPServerVersion(server.name);

  const initialDisplayName = version.display_name || version.server_json?.title || '';
  const initialToolsText = version.tools?.length ? JSON.stringify(version.tools, null, 2) : '';
  const existingAliases = (aliasesByVersion[version.version] ?? []).filter((a) => a !== LATEST_ALIAS);

  const [displayName, setDisplayName] = useState(initialDisplayName);
  const [status, setStatus] = useState<MCPStatus>(version.status);
  const [aliases, setAliases] = useState<string[]>(existingAliases);
  const [toolsText, setToolsText] = useState(initialToolsText);
  const [toolsValidationError, setToolsValidationError] = useState<string | null>(null);

  const handleSave = () => {
    const payload: Parameters<typeof updateVersionMutation.mutate>[0] = { version: version.version };

    if (displayName !== initialDisplayName) {
      payload.displayName = displayName;
    }
    if (status !== version.status) {
      payload.status = status;
    }

    setToolsValidationError(null);
    if (toolsText !== initialToolsText) {
      if (toolsText.trim()) {
        const toolsResult = validateToolsJson(toolsText);
        if (!toolsResult.valid) {
          setToolsValidationError(toolsResult.error ?? 'Invalid tools JSON');
          return;
        }
        payload.tools = toolsResult.parsed ?? null;
      } else {
        payload.tools = [];
      }
    }

    const addedAliases = aliases.filter((a) => !existingAliases.includes(a));
    const deletedAliases = existingAliases.filter((a) => !aliases.includes(a));
    if (addedAliases.length > 0 || deletedAliases.length > 0) {
      payload.aliases = { add: addedAliases, remove: deletedAliases };
    }

    updateVersionMutation.mutate(payload, {
      onSuccess: onClose,
    });
  };

  const handleCancel = () => {
    updateVersionMutation.reset();
    setToolsValidationError(null);
    onClose();
  };

  return (
    <Modal
      componentId="mlflow.mcp_registry.detail.version.edit_version_modal"
      title={
        <FormattedMessage
          defaultMessage="Edit version details"
          description="MCP server version edit details modal title"
        />
      }
      visible={visible}
      size="wide"
      destroyOnClose
      confirmLoading={updateVersionMutation.isLoading}
      okText={<FormattedMessage defaultMessage="Save" description="Save button" />}
      onOk={handleSave}
      onCancel={handleCancel}
    >
      {updateVersionMutation.error && (
        <Alert
          componentId="mlflow.mcp_registry.detail.version.edit_version_error"
          type="error"
          closable
          onClose={() => updateVersionMutation.reset()}
          message={
            updateVersionMutation.error instanceof Error
              ? updateVersionMutation.error.message
              : String(updateVersionMutation.error)
          }
          css={{ marginBottom: theme.spacing.sm }}
        />
      )}
      <div css={flexColumnGapStyles(theme, theme.spacing.md)}>
        <div>
          <Typography.Text bold css={blockLabelStyles(theme)}>
            <FormattedMessage defaultMessage="Display name" description="Version edit display name label" />
          </Typography.Text>
          <Input
            componentId="mlflow.mcp_registry.detail.version.edit_display_name_input"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            placeholder={intl.formatMessage({
              defaultMessage: 'Enter display name',
              description: 'Placeholder for version display name input',
            })}
          />
        </div>
        <div>
          <Typography.Text bold css={blockLabelStyles(theme)}>
            <FormattedMessage defaultMessage="Status" description="Version edit status label" />
          </Typography.Text>
          <SimpleSelect
            id="mcp-registry-edit-version-status"
            componentId="mlflow.mcp_registry.detail.version.edit_status_select"
            value={status}
            onChange={({ target }) => setStatus(target.value as MCPStatus)}
          >
            {(['draft', 'active', 'deprecated'] as MCPStatus[]).map((s) => (
              <SimpleSelectOption
                key={s}
                value={s}
                disabled={s !== version.status && !STATUS_TRANSITIONS[version.status]?.includes(s)}
              >
                {s.charAt(0).toUpperCase() + s.slice(1)}
              </SimpleSelectOption>
            ))}
          </SimpleSelect>
        </div>
        <div>
          <Typography.Text bold css={blockLabelStyles(theme)}>
            <FormattedMessage defaultMessage="Aliases" description="Version edit aliases label" />
          </Typography.Text>
          <AliasSelect
            renderKey={visible}
            disabled={false}
            draftAliases={aliases}
            existingAliases={(server.aliases ?? []).map((a) => a.alias).filter((a) => !RESERVED_ALIASES.includes(a))}
            setDraftAliases={setAliases}
            version={version.version}
            aliasToVersionMap={Object.fromEntries((server.aliases ?? []).map((a) => [a.alias, a.version]))}
          />
        </div>
        <div>
          <Typography.Text bold css={blockLabelStyles(theme)}>
            <FormattedMessage defaultMessage="Tools" description="Version edit tools label" />
          </Typography.Text>
          {toolsValidationError && (
            <Alert
              componentId="mlflow.mcp_registry.detail.version.tools_validation_error"
              type="error"
              closable
              onClose={() => setToolsValidationError(null)}
              message={toolsValidationError}
              css={{ marginBottom: theme.spacing.xs }}
            />
          )}
          <Input.TextArea
            componentId="mlflow.mcp_registry.detail.version.edit_tools_input"
            value={toolsText}
            onChange={(e) => {
              setToolsText(e.target.value);
              setToolsValidationError(null);
            }}
            autoSize={{ minRows: 4, maxRows: 12 }}
            css={monoFontStyles}
            placeholder={intl.formatMessage({
              defaultMessage: 'Enter tools JSON array',
              description: 'Placeholder for version tools input',
            })}
          />
        </div>
      </div>
    </Modal>
  );
};
