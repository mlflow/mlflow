import { Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { ProviderBadge } from './ProviderBadge';

interface ModelInfoProps {
  modelName: string;
  secretName?: string;
  secretMaskedValue?: string;
  provider?: string;
  showSecret?: boolean;
}

export const ModelInfo = ({ modelName, secretName, secretMaskedValue, provider, showSecret = true }: ModelInfoProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ position: 'relative' }}>
      {/* Provider badge in top-right */}
      {provider && (
        <div css={{ position: 'absolute', top: 0, right: 0 }}>
          <ProviderBadge provider={provider} />
        </div>
      )}

      {/* Model Section */}
      <div css={{ marginBottom: showSecret && secretName ? theme.spacing.md : 0 }}>
        <Typography.Text color="secondary" size="sm" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
          <FormattedMessage defaultMessage="Model" description="Model info > model label" />
        </Typography.Text>
        <Tag componentId="mlflow.model_info.model_tag">
          <Typography.Text>{modelName}</Typography.Text>
        </Tag>
      </div>

      {/* Secret Section (if provided) */}
      {showSecret && secretName && (
        <div
          css={{
            paddingLeft: theme.spacing.lg,
            display: 'grid',
            gridTemplateColumns: secretMaskedValue ? '1fr 1fr' : '1fr',
            gap: theme.spacing.md,
          }}
        >
          <div>
            <Typography.Text color="secondary" size="sm" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
              <FormattedMessage defaultMessage="Secret Name" description="Model info > secret name label" />
            </Typography.Text>
            <Typography.Paragraph css={{ marginTop: 0, marginBottom: 0, fontWeight: 500, fontSize: 14 }}>
              {secretName}
            </Typography.Paragraph>
          </div>
          {secretMaskedValue && (
            <div>
              <Typography.Text color="secondary" size="sm" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                <FormattedMessage defaultMessage="Masked Value" description="Model info > secret masked value label" />
              </Typography.Text>
              <Typography.Text
                css={{
                  display: 'block',
                  fontFamily: 'monospace',
                  fontSize: theme.typography.fontSizeSm,
                  color: theme.colors.textSecondary,
                }}
              >
                {secretMaskedValue}
              </Typography.Text>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

interface ModelListInfoProps {
  models: Array<{
    modelName: string;
    secretName?: string;
    secretMaskedValue?: string;
    provider?: string;
  }>;
  showSecrets?: boolean;
}

export const ModelListInfo = ({ models, showSecrets = true }: ModelListInfoProps) => {
  const { theme } = useDesignSystemTheme();

  if (models.length === 0) {
    return null;
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {models.map((model, index) => (
        <ModelInfo
          key={`${model.modelName}-${index}`}
          modelName={model.modelName}
          secretName={model.secretName}
          secretMaskedValue={model.secretMaskedValue}
          provider={model.provider}
          showSecret={showSecrets}
        />
      ))}
    </div>
  );
};
