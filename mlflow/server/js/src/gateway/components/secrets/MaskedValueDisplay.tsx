import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useMemo } from 'react';
import { formatCredentialFieldName } from '../../utils/providerUtils';

interface MaskedValueDisplayProps {
  maskedValue: Record<string, string>;
  compact?: boolean;
}

export const MaskedValueDisplay = ({ maskedValue, compact = false }: MaskedValueDisplayProps) => {
  const { theme } = useDesignSystemTheme();

  const entries = useMemo(() => Object.entries(maskedValue), [maskedValue]);

  const isSingleValue = entries.length === 1;
  const singleValue = isSingleValue ? entries[0][1] : null;

  if (isSingleValue && singleValue) {
    return (
      <Typography.Text
        css={{
          fontFamily: 'monospace',
          fontSize: theme.typography.fontSizeSm,
          backgroundColor: theme.colors.tagDefault,
          padding: `${theme.spacing.xs / 2}px ${theme.spacing.xs}px`,
          borderRadius: theme.general.borderRadiusBase,
          width: 'fit-content',
        }}
      >
        {singleValue}
      </Typography.Text>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: compact ? 2 : theme.spacing.xs / 2 }}>
      {entries.map(([key, value]) => (
        <div key={key} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
            {formatCredentialFieldName(key)}:
          </Typography.Text>
          <Typography.Text
            css={{
              fontFamily: 'monospace',
              fontSize: theme.typography.fontSizeSm,
              backgroundColor: theme.colors.tagDefault,
              padding: `${theme.spacing.xs / 4}px ${theme.spacing.xs}px`,
              borderRadius: theme.general.borderRadiusBase,
            }}
          >
            {value}
          </Typography.Text>
        </div>
      ))}
    </div>
  );
};
