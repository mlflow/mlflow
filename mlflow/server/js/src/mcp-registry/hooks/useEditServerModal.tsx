import { useState } from 'react';
import { isEqual } from 'lodash';
import { Alert, Input, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

import type { MCPIcon, MCPServer, UpdateMCPServerRequest } from '../types';
import { MCPRegistryApi } from '../api';
import { flexColumnGapStyles, blockLabelStyles } from '../styles';
import { IconEditor } from '../components/IconEditor';
import { SubsectionHelpHeading } from '../components/SubsectionHelpHeading';
import { useInvalidateServerQueries } from './useMCPServerVersionMutations';

interface EditServerState {
  displayName: string;
  description: string;
  icons: MCPIcon[];
}

const stripSource = (icons: MCPIcon[]): MCPIcon[] => icons.map(({ source, ...rest }) => rest);

export const useEditServerModal = ({ serverName }: { serverName: string }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const invalidate = useInvalidateServerQueries();

  const [visible, setVisible] = useState(false);
  const [initial, setInitial] = useState<EditServerState>({ displayName: '', description: '', icons: [] });
  const [state, setState] = useState<EditServerState>({ displayName: '', description: '', icons: [] });
  const [serverJsonIcons, setServerJsonIcons] = useState<MCPIcon[] | undefined>(undefined);

  const mutation = useMutation<unknown, Error, UpdateMCPServerRequest>({
    mutationFn: (request) => MCPRegistryApi.updateMCPServer(serverName, request),
    onSuccess: () => invalidate(serverName),
  });

  const openEditServer = (server: MCPServer, sjIcons?: MCPIcon[]) => {
    const init: EditServerState = {
      displayName: server.display_name || '',
      description: server.description || '',
      icons: stripSource(server.icons?.filter((i) => i.source !== 'version') ?? []),
    };
    setInitial(init);
    setState(init);
    setServerJsonIcons(sjIcons);
    mutation.reset();
    setVisible(true);
  };

  const handleSave = () => {
    const request: UpdateMCPServerRequest = {};
    let hasChanges = false;

    if (state.displayName !== initial.displayName) {
      request.display_name = state.displayName.trim() || null;
      hasChanges = true;
    }

    if (state.description !== initial.description) {
      request.description = state.description.trim() || null;
      hasChanges = true;
    }

    const validIcons = state.icons.filter((i) => i.src.trim());
    const initialValidIcons = initial.icons.filter((i) => i.src.trim());
    if (!isEqual(validIcons, initialValidIcons)) {
      request.icons = validIcons.length > 0 ? validIcons : null;
      hasChanges = true;
    }

    if (!hasChanges) {
      setVisible(false);
      return;
    }

    mutation.mutate(request, {
      onSuccess: () => setVisible(false),
    });
  };

  const handleCancel = () => {
    mutation.reset();
    setVisible(false);
  };

  const EditServerModal = visible ? (
    <Modal
      componentId="mlflow.mcp_registry.edit_server_modal"
      title={<FormattedMessage defaultMessage="Edit server details" description="MCP server edit modal title" />}
      visible={visible}
      size="wide"
      destroyOnClose
      confirmLoading={mutation.isLoading}
      okText={<FormattedMessage defaultMessage="Save" description="Edit server modal save button" />}
      onOk={handleSave}
      onCancel={handleCancel}
    >
      {mutation.error && (
        <Alert
          componentId="mlflow.mcp_registry.edit_server_modal.error"
          type="error"
          closable
          onClose={() => mutation.reset()}
          message={mutation.error instanceof Error ? mutation.error.message : String(mutation.error)}
          css={{ marginBottom: theme.spacing.sm }}
        />
      )}
      <div css={flexColumnGapStyles(theme, theme.spacing.md)}>
        <div>
          <Typography.Text bold css={blockLabelStyles(theme)}>
            <FormattedMessage defaultMessage="Display name" description="Edit server display name label" />
          </Typography.Text>
          <Input
            componentId="mlflow.mcp_registry.edit_server_modal.display_name"
            value={state.displayName}
            onChange={(e) => setState((prev) => ({ ...prev, displayName: e.target.value }))}
            placeholder={intl.formatMessage({
              defaultMessage: 'Enter display name',
              description: 'Placeholder for server display name input',
            })}
          />
        </div>
        <div>
          <Typography.Text bold css={blockLabelStyles(theme)}>
            <FormattedMessage defaultMessage="Description" description="Edit server description label" />
          </Typography.Text>
          <Input
            componentId="mlflow.mcp_registry.edit_server_modal.description"
            value={state.description}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setState((prev) => ({ ...prev, description: e.target.value }))
            }
            placeholder={intl.formatMessage({
              defaultMessage: 'Enter description',
              description: 'Placeholder for server description input',
            })}
          />
        </div>
        <div>
          <SubsectionHelpHeading
            title={<FormattedMessage defaultMessage="Icons" description="Edit server icons label" />}
            componentId="mlflow.mcp_registry.edit_server_modal.icons_help"
            helpAriaLabel={intl.formatMessage({
              defaultMessage: 'About icons',
              description: 'Aria label for icons help popover in edit server modal',
            })}
            helpText={
              <FormattedMessage
                defaultMessage="Set icons or override icons from server.json. Use 'light' or 'dark' for theme-specific icons, or 'any' for one that works in both."
                description="Help text for icons in edit server modal"
              />
            }
          />
          <IconEditor
            icons={state.icons}
            onChange={(icons) => setState((prev) => ({ ...prev, icons }))}
            serverJsonIcons={serverJsonIcons}
          />
        </div>
      </div>
    </Modal>
  ) : null;

  return { EditServerModal, openEditServer };
};
