import { Tag } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export const MCPRegistryBetaTag = () => (
  <Tag componentId="mlflow.mcp_registry.beta_tag" color="turquoise">
    <FormattedMessage defaultMessage="Beta" description="MCP Registry beta feature tag" />
  </Tag>
);
