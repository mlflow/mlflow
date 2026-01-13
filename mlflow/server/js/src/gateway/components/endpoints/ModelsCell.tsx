import { useState } from 'react';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { Endpoint } from '../../types';

interface ModelsCellProps {
  modelMappings: Endpoint['model_mappings'];
}

export const ModelsCell = ({ modelMappings }: ModelsCellProps) => {
  const { theme } = useDesignSystemTheme();
  const [isExpanded, setIsExpanded] = useState(false);

  if (!modelMappings || modelMappings.length === 0) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }

  const primaryMapping = modelMappings[0];
  const additionalMappings = modelMappings.slice(1);
  const additionalCount = additionalMappings.length;

  const displayedMappings = isExpanded ? modelMappings : [primaryMapping];

  return (
    <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: theme.spacing.xs / 2 }}>
      {displayedMappings.map((mapping) => (
        <Typography.Text key={mapping.mapping_id} css={{ fontSize: theme.typography.fontSizeSm }}>
          {mapping.model_definition?.model_name ?? '-'}
        </Typography.Text>
      ))}
      {additionalCount > 0 && (
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          css={{
            background: 'none',
            border: 'none',
            padding: 0,
            margin: 0,
            textAlign: 'left',
            fontSize: theme.typography.fontSizeSm,
            color: theme.colors.textSecondary,
            cursor: 'pointer',
            '&:hover': { textDecoration: 'underline' },
          }}
        >
          {isExpanded ? 'Show less' : `+${additionalCount} more`}
        </button>
      )}
    </div>
  );
};
