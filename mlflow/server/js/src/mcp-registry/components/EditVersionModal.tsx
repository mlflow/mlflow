import { useState } from 'react';
import {
  Alert,
  Modal,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import type { MCPServer, MCPServerVersion } from '../types';
import { MCPStatus } from '../types';
import { STATUS_TRANSITIONS, LATEST_ALIAS, RESERVED_ALIASES } from '../utils';
import { flexColumnGapStyles, blockLabelStyles } from '../styles';
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
  const updateVersionMutation = useUpdateMCPServerVersion(server.name);

  const existingAliases = (aliasesByVersion[version.version] ?? []).filter((a) => a !== LATEST_ALIAS);

  const [status, setStatus] = useState<MCPStatus>(version.status);
  const [aliases, setAliases] = useState<string[]>(existingAliases);

  const handleSave = () => {
    const payload: Parameters<typeof updateVersionMutation.mutate>[0] = { version: version.version };

    if (status !== version.status) {
      payload.status = status;
    }

    const addedAliases = aliases.filter((a) => !existingAliases.includes(a));
    const deletedAliases = existingAliases.filter((a) => !aliases.includes(a));
    if (addedAliases.length > 0 || deletedAliases.length > 0) {
      payload.aliases = { add: addedAliases, remove: deletedAliases };
    }

    updateVersionMutation.mutate(payload, {
      onSuccess: () => onClose(),
    });
  };

  const handleCancel = () => {
    updateVersionMutation.reset();
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
            <FormattedMessage defaultMessage="Status" description="Version edit status label" />
          </Typography.Text>
          <SimpleSelect
            id="mcp-registry-edit-version-status"
            componentId="mlflow.mcp_registry.detail.version.edit_status_select"
            value={status}
            onChange={({ target }) => setStatus(target.value as MCPStatus)}
          >
            {[MCPStatus.DRAFT, MCPStatus.ACTIVE, MCPStatus.DEPRECATED].map((s) => (
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
      </div>
    </Modal>
  );
};
