import { CopyIcon, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { CopyButton } from '../../shared/building_blocks/CopyButton';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { overlayButtonStyles } from '../styles';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';

export const AddToolsModal = ({
  visible,
  serverName,
  version,
  onClose,
}: {
  visible: boolean;
  serverName: string;
  version: string;
  onClose: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const workspace = useActiveWorkspace();

  const hostname = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
  const trackingUri = `http://${hostname}:<port>`;
  const escapedName = serverName.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  const escapedVersion = version.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  const workspaceLine = workspace ? `\nmlflow.set_workspace("${workspace}")` : '';

  const discoverSnippet = `import mlflow

mlflow.set_tracking_uri("${trackingUri}")${workspaceLine}

server_version = mlflow.genai.refresh_mcp_server_version_tools(
    name="${escapedName}",
    version="${escapedVersion}",
)`;

  if (!visible) return null;

  return (
    <Modal
      componentId="mlflow.mcp_registry.add_tools_modal"
      title={<FormattedMessage defaultMessage="Auto-discover tools" description="Add tools modal title" />}
      visible={visible}
      onCancel={onClose}
      footer={null}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Auto-discover tools by connecting to the server's official endpoint. Discovered tools will replace any existing tool definitions on this version."
            description="Add tools modal description"
          />
        </Typography.Text>
        <div css={{ position: 'relative' }}>
          <CopyButton
            componentId="mlflow.mcp_registry.add_tools_modal.discover.copy"
            showLabel={false}
            copyText={discoverSnippet}
            icon={<CopyIcon />}
            css={overlayButtonStyles(theme)}
          />
          <CodeSnippet
            language="python"
            theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
            style={{ padding: theme.spacing.sm, paddingRight: theme.spacing.xl + theme.spacing.sm }}
          >
            {discoverSnippet}
          </CodeSnippet>
        </div>
      </div>
    </Modal>
  );
};
