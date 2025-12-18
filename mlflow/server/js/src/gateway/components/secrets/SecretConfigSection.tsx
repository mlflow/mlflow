import { Radio, useDesignSystemTheme, FormUI, Tooltip } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { SecretSelector } from './SecretSelector';
import { CreateSecretForm } from './CreateSecretForm';
import { useSecretsQuery } from '../../hooks/useSecretsQuery';

export type SecretMode = 'existing' | 'new';

interface SecretConfigSectionProps {
  provider: string;
  mode: SecretMode;
  onModeChange: (mode: SecretMode) => void;
  selectedSecretId?: string;
  onSecretSelect: (secretId: string) => void;
  newSecretFieldPrefix?: string;
  error?: string;
  showModeSelector?: boolean;
  label?: string;
  componentIdPrefix?: string;
}

export const SecretConfigSection = ({
  provider,
  mode,
  onModeChange,
  selectedSecretId,
  onSecretSelect,
  newSecretFieldPrefix = 'newSecret',
  error,
  showModeSelector = true,
  label,
  componentIdPrefix = 'mlflow.gateway.secret-config',
}: SecretConfigSectionProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { data: secrets } = useSecretsQuery({ provider });
  const filteredSecrets = provider ? secrets?.filter((s) => s.provider === provider) : secrets;
  const hasExistingSecrets = filteredSecrets && filteredSecrets.length > 0;

  if (!provider) {
    return (
      <div>
        {label !== '' && (
          <FormUI.Label>
            {label || <FormattedMessage defaultMessage="Connections" description="Label for connections section" />}
          </FormUI.Label>
        )}
        <div css={{ color: theme.colors.textSecondary, marginTop: theme.spacing.xs }}>
          <FormattedMessage
            defaultMessage="Select a provider to configure authentication"
            description="Message when no provider selected"
          />
        </div>
      </div>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {showModeSelector && (
        <div>
          {label !== '' && (
            <FormUI.Label>
              {label || <FormattedMessage defaultMessage="Connections" description="Label for connections section" />}
            </FormUI.Label>
          )}
          <Radio.Group
            componentId={`${componentIdPrefix}.mode`}
            name="secretMode"
            value={mode}
            onChange={(e) => onModeChange(e.target.value as SecretMode)}
            layout="horizontal"
            css={{ gap: theme.spacing.md }}
          >
            <Radio value="new">
              <FormattedMessage defaultMessage="Create new API key" description="Option to create new API key" />
            </Radio>
            <Tooltip
              componentId={`${componentIdPrefix}.mode.tooltip`}
              content={
                !hasExistingSecrets
                  ? intl.formatMessage({
                      defaultMessage: 'No existing API keys for this provider',
                      description: 'Tooltip when no API keys exist for provider',
                    })
                  : undefined
              }
            >
              <span>
                <Radio value="existing" disabled={!hasExistingSecrets}>
                  <FormattedMessage
                    defaultMessage="Use existing API key"
                    description="Option to use existing API key"
                  />
                </Radio>
              </span>
            </Tooltip>
          </Radio.Group>
        </div>
      )}

      {mode === 'existing' ? (
        <SecretSelector provider={provider} value={selectedSecretId ?? ''} onChange={onSecretSelect} error={error} />
      ) : (
        <CreateSecretForm
          provider={provider}
          fieldPrefix={newSecretFieldPrefix}
          componentIdPrefix={componentIdPrefix}
        />
      )}
    </div>
  );
};
