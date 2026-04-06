import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

/**
 * Vertical connector line with optional "Fallback" label, used between
 * the primary model section and fallback models, and between fallback model items.
 */
export const FallbackConnectorLine = ({ showLabel = true }: { showLabel?: boolean }) => {
  const { theme } = useDesignSystemTheme();

  const line = <div css={{ width: 2, height: theme.spacing.md, backgroundColor: theme.colors.border }} />;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      {line}
      {showLabel && (
        <>
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
          {line}
        </>
      )}
    </div>
  );
};
