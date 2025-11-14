import { LightningIcon, useDesignSystemTheme } from '@databricks/design-system';

export interface ProviderBadgeProps {
  provider?: string;
}

export const ProviderBadge = ({ provider }: ProviderBadgeProps) => {
  const { theme } = useDesignSystemTheme();

  const getProviderColor = (provider?: string) => {
    if (!provider) {
      return { bg: theme.colors.backgroundSecondary, text: theme.colors.textSecondary };
    }
    switch (provider.toLowerCase()) {
      case 'openai':
        return { bg: '#10A37F15', text: '#10A37F' };
      case 'anthropic':
        return { bg: '#D4915215', text: '#D49152' };
      case 'bedrock':
      case 'aws':
        return { bg: '#FF990015', text: '#FF9900' };
      case 'vertex_ai':
      case 'google':
        return { bg: '#4285F415', text: '#4285F4' };
      case 'azure':
        return { bg: '#0078D415', text: '#0078D4' };
      case 'databricks':
        return { bg: '#FF362115', text: '#FF3621' };
      default:
        return { bg: theme.colors.backgroundSecondary, text: theme.colors.textSecondary };
    }
  };

  const colors = getProviderColor(provider);

  return (
    <div
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '4px 8px',
        gap: theme.spacing.xs,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: colors.bg,
        color: colors.text,
        fontSize: theme.typography.fontSizeSm,
        fontWeight: 600,
      }}
    >
      <LightningIcon css={{ fontSize: 12 }} />
      {provider || 'Unknown'}
    </div>
  );
};
