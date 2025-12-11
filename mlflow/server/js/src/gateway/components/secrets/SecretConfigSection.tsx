import { Radio, useDesignSystemTheme, FormUI, Tooltip } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { SecretSelector } from './SecretSelector';
import { CreateSecretForm } from './CreateSecretForm';
import { useSecretsQuery } from '../../hooks/useSecretsQuery';

export type SecretMode = 'existing' | 'new';

interface SecretConfigSectionProps {
  /** Provider to filter secrets and fetch auth config */
  provider: string;
  /** Current mode: 'existing' to select an existing secret, 'new' to create a new one */
  mode: SecretMode;
  /** Callback when mode changes */
  onModeChange: (mode: SecretMode) => void;
  /** Currently selected secret ID (when mode is 'existing') */
  selectedSecretId?: string;
  /** Callback when an existing secret is selected */
  onSecretSelect: (secretId: string) => void;
  /** Field prefix for react-hook-form nested fields (default: 'newSecret') */
  newSecretFieldPrefix?: string;
  /** Error message for the secret selector */
  error?: string;
  /** Whether to show the mode selector radio buttons (default: true) */
  showModeSelector?: boolean;
  /** Label for the section (default: 'Connections') */
  label?: string;
  /** Component ID prefix for telemetry */
  componentIdPrefix?: string;
}

/**
 * A section for configuring authentication via secrets.
 * Allows users to either select an existing secret or create a new one.
 *
 * When `showModeSelector` is false, only shows the form for the current mode.
 * This is useful for edit flows where you might only want to show the "new secret" form.
 */
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
