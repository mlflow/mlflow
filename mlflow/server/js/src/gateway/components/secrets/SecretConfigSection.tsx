import { Radio, useDesignSystemTheme, FormUI } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { SecretSelector } from './SecretSelector';
import { CreateSecretForm } from './CreateSecretForm';

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
  /** Label for the section (default: 'Authentication') */
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

  if (!provider) {
    return (
      <div>
        {label !== null && (
          <FormUI.Label>
            {label ?? (
              <FormattedMessage defaultMessage="Authentication" description="Label for authentication section" />
            )}
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
          {label !== null && (
            <FormUI.Label>
              {label ?? (
                <FormattedMessage defaultMessage="Authentication" description="Label for authentication section" />
              )}
            </FormUI.Label>
          )}
          <Radio.Group
            componentId={`${componentIdPrefix}.mode`}
            name="secretMode"
            value={mode}
            onChange={(e) => onModeChange(e.target.value as SecretMode)}
            layout="horizontal"
          >
            <Radio value="existing">
              <FormattedMessage defaultMessage="Use existing secret" description="Option to use existing secret" />
            </Radio>
            <Radio value="new">
              <FormattedMessage defaultMessage="Create new secret" description="Option to create new secret" />
            </Radio>
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
