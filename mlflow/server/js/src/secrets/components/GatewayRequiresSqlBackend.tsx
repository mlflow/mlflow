import { DatabaseIcon, Empty, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

/**
 * Placeholder component shown when Gateway features are accessed but the backend
 * is using FileStore instead of a SQL backend.
 */
export const GatewayRequiresSqlBackend = ({ storeType }: { storeType?: string | null }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '60vh',
        padding: theme.spacing.lg,
      }}
    >
      <div css={{ maxWidth: 600, textAlign: 'center' as const }}>
        <Empty
          description={
            <div css={{ textAlign: 'left' as const }}>
              <FormattedMessage
                defaultMessage="Gateway features (secrets and endpoints management) require a SQL database backend. You are currently using {storeType}."
                description="Empty state description for Gateway when FileStore backend is detected"
                values={{
                  storeType: storeType || 'FileStore',
                }}
              />
              <div css={{ marginTop: 16, marginBottom: 8, fontWeight: 500 }}>
                <FormattedMessage
                  defaultMessage="To use Gateway features, restart your MLflow server with a SQL backend:"
                  description="Instructions header for switching to SQL backend"
                />
              </div>
              <code
                css={{
                  display: 'block',
                  marginTop: theme.spacing.sm,
                  marginBottom: theme.spacing.sm,
                  padding: theme.spacing.md,
                  backgroundColor: theme.colors.backgroundSecondary,
                  borderRadius: theme.borders.borderRadiusMd,
                  fontFamily: 'monospace',
                  fontSize: theme.typography.fontSizeSm,
                  textAlign: 'left' as const,
                  whiteSpace: 'pre-wrap' as const,
                }}
              >
                export MLFLOW_BACKEND_STORE_URI="sqlite:///mlflow.db"
                <br />
                mlflow server
              </code>
              <div
                css={{
                  marginTop: theme.spacing.sm,
                  fontSize: theme.typography.fontSizeSm,
                  color: theme.colors.textSecondary,
                }}
              >
                <FormattedMessage
                  defaultMessage="Or use any other SQL database (PostgreSQL, MySQL, etc.)"
                  description="Alternative backend options text"
                />
              </div>
            </div>
          }
          title={
            <FormattedMessage
              defaultMessage="SQL Backend Required"
              description="Empty state title for Gateway when SQL backend is required"
            />
          }
          image={
            <DatabaseIcon
              css={{
                fontSize: 64,
                color: theme.colors.textSecondary,
              }}
            />
          }
        />
      </div>
    </div>
  );
};
