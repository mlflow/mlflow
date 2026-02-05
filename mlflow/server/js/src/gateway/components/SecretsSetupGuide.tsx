import {
  Alert,
  CloudModelIcon,
  CopyIcon,
  NewWindowIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import { CodeSnippet } from '@databricks/web-shared/snippet';

const INSTALL_SNIPPET = `pip install "mlflow[genai]"`;

const START_SERVER_SNIPPET = `mlflow server \\
  --backend-store-uri sqlite:///mlflow.db \\
  --host 0.0.0.0 \\
  --port 5000`;

const PASSPHRASE_SNIPPET = `# Generate a passphrase once and store it securely
openssl rand -base64 32

# Set the environment variable before starting the server
export MLFLOW_CRYPTO_KEK_PASSPHRASE="<your-stored-passphrase>"`;

const GATEWAY_DOCS_URL = 'https://mlflow.org/docs/latest';

export const GatewaySetupGuide = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: theme.spacing.lg,
        maxWidth: 800,
        margin: '0 auto',
      }}
    >
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: theme.spacing.sm,
          marginBottom: theme.spacing.lg,
        }}
      >
        <div
          css={{
            borderRadius: theme.borders.borderRadiusMd,
            background: theme.colors.actionDefaultBackgroundHover,
            padding: theme.spacing.md,
            color: theme.colors.textPrimary,
          }}
        >
          <CloudModelIcon css={{ fontSize: 48 }} />
        </div>
        <Typography.Title level={2} css={{ margin: 0, textAlign: 'center' }}>
          <FormattedMessage
            defaultMessage="Set up MLflow AI Gateway"
            description="AI Gateway setup guide > Main title"
          />
        </Typography.Title>
        <Typography.Text color="secondary" css={{ textAlign: 'center' }}>
          <FormattedMessage
            defaultMessage="Follow these steps to enable the AI Gateway feature for managing AI provider credentials."
            description="AI Gateway setup guide > Subtitle"
          />
        </Typography.Text>
      </div>

      <div
        css={{
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.lg,
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusLg,
          padding: theme.spacing.lg,
          backgroundColor: theme.colors.backgroundPrimary,
          boxShadow: theme.shadows.xs,
        }}
      >
        <section css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Title level={4} css={{ margin: 0 }}>
            <FormattedMessage
              defaultMessage="1. Install MLflow with GenAI extras on the server"
              description="AI Gateway setup guide > Step 1 title"
            />
          </Typography.Title>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="The AI Gateway requires additional dependencies installed on the MLflow tracking server (not client machines):"
              description="AI Gateway setup guide > Step 1 description"
            />
          </Typography.Text>
          <div css={{ position: 'relative', width: '100%' }}>
            <CopyButton
              componentId="mlflow.gateway.setup.install.copy"
              css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
              showLabel={false}
              copyText={INSTALL_SNIPPET}
              icon={<CopyIcon />}
            />
            <CodeSnippet
              language="text"
              theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
              style={{
                padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
              }}
            >
              {INSTALL_SNIPPET}
            </CodeSnippet>
          </div>
        </section>

        <section css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Title level={4} css={{ margin: 0 }}>
            <FormattedMessage
              defaultMessage="2. Use a SQL-based tracking store"
              description="AI Gateway setup guide > Step 2 title"
            />
          </Typography.Title>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="The AI Gateway requires a SQL-based backend store (SQLite, PostgreSQL, MySQL, or MSSQL) to persist credentials securely. Start the MLflow server with a database URI:"
              description="AI Gateway setup guide > Step 2 description"
            />
          </Typography.Text>
          <div css={{ position: 'relative', width: '100%' }}>
            <CopyButton
              componentId="mlflow.gateway.setup.server.copy"
              css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
              showLabel={false}
              copyText={START_SERVER_SNIPPET}
              icon={<CopyIcon />}
            />
            <CodeSnippet
              showLineNumbers
              language="text"
              theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
              style={{
                padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
              }}
            >
              {START_SERVER_SNIPPET}
            </CodeSnippet>
          </div>
        </section>

        <section css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Title level={4} css={{ margin: 0 }}>
            <FormattedMessage
              defaultMessage="3. Configure encryption passphrase (production deployments)"
              description="AI Gateway setup guide > Step 3 title"
            />
          </Typography.Title>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="For local development, MLflow uses a default passphrase. For production deployments, server administrators must set a secure encryption passphrase on the tracking server before starting it:"
              description="AI Gateway setup guide > Step 3 description"
            />
          </Typography.Text>
          <div css={{ position: 'relative', width: '100%' }}>
            <CopyButton
              componentId="mlflow.gateway.setup.passphrase.copy"
              css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
              showLabel={false}
              copyText={PASSPHRASE_SNIPPET}
              icon={<CopyIcon />}
            />
            <CodeSnippet
              showLineNumbers
              language="text"
              theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
              style={{
                padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
              }}
            >
              {PASSPHRASE_SNIPPET}
            </CodeSnippet>
          </div>
          <Alert
            type="warning"
            componentId="mlflow.gateway.setup.passphrase.warning"
            closable={false}
            message={
              <FormattedMessage
                defaultMessage="This passphrase protects encryption keys and should never be shared. {securityNote}"
                description="AI Gateway setup guide > Passphrase warning"
                values={{
                  securityNote: (
                    <strong>
                      <FormattedMessage
                        defaultMessage="Store it securely and restrict access to server administrators only."
                        description="AI Gateway setup guide > Passphrase warning security note"
                      />
                    </strong>
                  ),
                }}
              />
            }
          />
        </section>

        <section>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Learn more about the AI Gateway in the {gatewayDocs}."
              description="AI Gateway setup guide > Documentation link"
              values={{
                gatewayDocs: (
                  <a
                    href={GATEWAY_DOCS_URL}
                    target="_blank"
                    rel="noopener noreferrer"
                    css={{ display: 'inline-flex', gap: theme.spacing.xs, alignItems: 'center' }}
                  >
                    <FormattedMessage
                      defaultMessage="MLflow documentation"
                      description="AI Gateway setup guide > Documentation link text"
                    />
                    <NewWindowIcon css={{ fontSize: 14 }} />
                  </a>
                ),
              }}
            />
          </Typography.Text>
        </section>
      </div>
    </div>
  );
};
