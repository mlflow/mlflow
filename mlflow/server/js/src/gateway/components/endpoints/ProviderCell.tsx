import { Tag, Typography } from '@databricks/design-system';
import { formatProviderName } from '../../utils/providerUtils';
import type { Endpoint } from '../../types';

interface ProviderCellProps {
  modelMappings: Endpoint['model_mappings'];
}

export const ProviderCell = ({ modelMappings }: ProviderCellProps) => {
  if (!modelMappings || modelMappings.length === 0) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }

  const primaryProvider = modelMappings[0]?.model_definition?.provider;
  if (!primaryProvider) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }

  return <Tag componentId="mlflow.gateway.endpoints-list.provider-tag">{formatProviderName(primaryProvider)}</Tag>;
};
