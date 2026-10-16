import { McpIcon } from '@databricks/design-system';
import { useIntl } from 'react-intl';

import { RegistryIconEditor } from '../../common/components/RegistryIconEditor';
import type { MCPIcon } from '../types';

export const IconEditor = ({
  icons,
  onChange,
  serverJsonIcons,
}: {
  icons: MCPIcon[];
  onChange: (icons: MCPIcon[]) => void;
  serverJsonIcons?: MCPIcon[];
}) => {
  const intl = useIntl();
  return (
    <RegistryIconEditor
      icons={icons}
      onChange={onChange}
      defaultIcon={<McpIcon aria-hidden />}
      componentId="mlflow.mcp_registry.icon_editor"
      fallback={{
        icons: serverJsonIcons ?? [],
        describe: (url) =>
          intl.formatMessage(
            { defaultMessage: 'From server.json: {url}', description: 'Tooltip for server.json icon' },
            { url },
          ),
      }}
    />
  );
};
