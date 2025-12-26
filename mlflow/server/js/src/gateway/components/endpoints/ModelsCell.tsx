import { Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { Endpoint } from '../../types';

interface ModelsCellProps {
  modelMappings: Endpoint['model_mappings'];
}

export const ModelsCell = ({ modelMappings }: ModelsCellProps) => {
  const { theme } = useDesignSystemTheme();

  if (!modelMappings || modelMappings.length === 0) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }

  const primaryMapping = modelMappings[0];
  const primaryModelDef = primaryMapping.model_definition;
  const additionalMappings = modelMappings.slice(1);
  const additionalCount = additionalMappings.length;

  const tooltipContent =
    additionalCount > 0 ? additionalMappings.map((m) => m.model_definition?.model_name ?? '-').join(', ') : undefined;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: theme.spacing.xs / 2 }}>
      <Typography.Text css={{ fontSize: theme.typography.fontSizeSm }}>
        {primaryModelDef?.model_name ?? '-'}
      </Typography.Text>
      {additionalCount > 0 && (
        <Tooltip componentId="mlflow.gateway.endpoints-list.models-more-tooltip" content={tooltipContent}>
          <button
            type="button"
            css={{
              background: 'none',
              border: 'none',
              padding: 0,
              margin: 0,
              textAlign: 'left',
              fontSize: theme.typography.fontSizeSm,
              color: theme.colors.textSecondary,
              cursor: 'default',
              '&:hover': { textDecoration: 'underline' },
            }}
          >
            +{additionalCount} more
          </button>
        </Tooltip>
      )}
    </div>
  );
};
