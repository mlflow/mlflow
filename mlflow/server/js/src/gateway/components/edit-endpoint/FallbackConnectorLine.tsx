import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

/**
 * Simple vertical connector line used between model sections.
 */
export const ConnectorLine = () => {
  const { theme } = useDesignSystemTheme();
  return <div css={{ width: 2, height: theme.spacing.md, backgroundColor: theme.colors.border }} />;
};

/**
 * Vertical connector with a "Fallback" label, used between
 * the primary model section and fallback models, and between fallback model items.
 */
export const FallbackConnectorLine = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <ConnectorLine />
      <Typography.Text
        color="secondary"
        css={{
          padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
          fontSize: theme.typography.fontSizeSm,
        }}
      >
        <FormattedMessage
          defaultMessage="Fallback"
          description="Gateway > Endpoint details > Label on connector between fallback model sections"
        />
      </Typography.Text>
      <ConnectorLine />
    </div>
  );
};
