import { Alert, Typography, useDesignSystemTheme, KeyIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { SnippetCopyAction } from '@mlflow/mlflow/src/shared/web-shared/snippet';

const PASSPHRASE_ENV_VAR = 'MLFLOW_CRYPTO_KEK_PASSPHRASE';

interface CodeBlockProps {
  code: string;
}

const CodeBlock = ({ code }: CodeBlockProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        position: 'relative',
        padding: theme.spacing.sm,
        paddingRight: theme.spacing.lg * 2,
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${theme.colors.borderDecorative}`,
        backgroundColor: theme.colors.backgroundSecondary,
      }}
    >
      <div css={{ position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs, zIndex: 1 }}>
        <SnippetCopyAction copyText={code} componentId="mlflow.gateway.secrets-setup" />
      </div>
      <code
        css={{
          display: 'block',
          fontFamily: 'monospace',
          fontSize: 13,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {code}
      </code>
    </div>
  );
};

export const SecretsSetupGuide = () => {
  const { theme } = useDesignSystemTheme();

  const exportCommand = `export ${PASSPHRASE_ENV_VAR}="your-secure-passphrase"`;
  const serverCommand = 'mlflow server';

  return (
    <div
      css={{
        maxWidth: 640,
        margin: '0 auto',
        padding: theme.spacing.lg,
      }}
    >
      <div css={{ textAlign: 'center', marginBottom: theme.spacing.lg }}>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 56,
            height: 56,
            margin: '0 auto',
            marginBottom: theme.spacing.md,
            borderRadius: '50%',
            backgroundColor: theme.colors.tagDefault,
          }}
        >
          <KeyIcon css={{ fontSize: 28, color: theme.colors.textSecondary }} />
        </div>
        <Typography.Title level={2} css={{ marginBottom: theme.spacing.sm }}>
          <FormattedMessage
            defaultMessage="Encryption Required"
            description="Title for secrets setup guide in gateway"
          />
        </Typography.Title>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="AI Gateway stores API keys securely using encryption. Follow the steps below to enable this feature."
            description="Subtitle for secrets setup guide in gateway"
          />
        </Typography.Text>
      </div>

      <div css={{ marginBottom: theme.spacing.lg }}>
        <ol
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md,
            margin: 0,
            paddingLeft: theme.spacing.lg,
          }}
        >
          <li>
            <Typography.Text>
              <FormattedMessage
                defaultMessage="Stop the MLflow tracking server if it's running"
                description="Step 1 in secrets setup guide"
              />
            </Typography.Text>
          </li>
          <li>
            <Typography.Text>
              <FormattedMessage
                defaultMessage="Generate a secure passphrase (16+ characters) and store it in a secrets manager"
                description="Step 2 in secrets setup guide"
              />
            </Typography.Text>
          </li>
          <li>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              <Typography.Text>
                <FormattedMessage
                  defaultMessage="Set the environment variable:"
                  description="Step 3 in secrets setup guide"
                />
              </Typography.Text>
              <CodeBlock code={exportCommand} />
            </div>
          </li>
          <li>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              <Typography.Text>
                <FormattedMessage
                  defaultMessage="Restart the tracking server:"
                  description="Step 4 in secrets setup guide"
                />
              </Typography.Text>
              <CodeBlock code={serverCommand} />
            </div>
          </li>
        </ol>
      </div>

      <Alert
        type="warning"
        componentId="mlflow.gateway.secrets-setup.security"
        closable={false}
        message={
          <FormattedMessage
            defaultMessage="Security Requirements"
            description="Security section title in secrets setup guide"
          />
        }
        description={
          <ul css={{ margin: 0, marginTop: theme.spacing.xs, paddingLeft: theme.spacing.lg }}>
            <li>
              <FormattedMessage
                defaultMessage="Never commit the passphrase to version control"
                description="Security requirement 1 in secrets setup guide"
              />
            </li>
            <li>
              <FormattedMessage
                defaultMessage="Use a secrets manager (HashiCorp Vault, AWS Secrets Manager)"
                description="Security requirement 2 in secrets setup guide"
              />
            </li>
            <li>
              <FormattedMessage
                defaultMessage="Changing the passphrase will make existing secrets unreadable"
                description="Security requirement 3 in secrets setup guide"
              />
            </li>
          </ul>
        }
      />
    </div>
  );
};
