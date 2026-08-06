import { Tag } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export const GatewayNewTag = () => (
  <Tag componentId="mlflow.sidebar.gateway_new_tag" color="turquoise" css={{ marginLeft: 'auto' }}>
    <FormattedMessage defaultMessage="New" description="Sidebar > AI Gateway > New feature tag" />
  </Tag>
);

export const GatewayLabel = () => (
  <FormattedMessage
    defaultMessage="AI Gateway"
    description="Shared label for AI Gateway used in sidebar and breadcrumbs"
  />
);
