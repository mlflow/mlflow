/**
 * Inline in-chat prompt collecting the resolved provider's API key, shown above
 * the composer when the next send needs one. The first send doubles as setup:
 * the queued message is delivered as soon as the key is saved.
 */
import { useCallback, useState } from 'react';
import { Button, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useAssistant } from './AssistantContext';
import { updateConfig } from './AssistantService';
import { GATEWAY_PROVIDER_ID } from './constants';
import { SecretInput } from '../gateway/components/secrets/SecretInput';

const PROVIDER_KEY_PLACEHOLDERS = {
  openai: 'sk-...',
  anthropic: 'sk-ant-...',
} satisfies Record<string, string>;
const DEFAULT_KEY_PLACEHOLDER = 'API key';

const keyPlaceholderFor = (providerId: string): string =>
  (PROVIDER_KEY_PLACEHOLDERS as Record<string, string | undefined>)[providerId] ?? DEFAULT_KEY_PLACEHOLDER;

interface ApiKeyPromptProps {
  providerId: string;
  providerName: string;
  gatewayVendor?: string;
  providerModel?: string | null;
  /** Called after the key was saved successfully. */
  onSaved: () => void;
}

export const ApiKeyPrompt = ({
  providerId,
  providerName,
  gatewayVendor,
  providerModel,
  onSaved,
}: ApiKeyPromptProps) => {
  const { theme } = useDesignSystemTheme();
  const { isLocalServer, canEditServerSettings } = useAssistant();
  const [apiKey, setApiKey] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSave = useCallback(async () => {
    if (!apiKey.trim()) {
      return;
    }
    setIsSaving(true);
    setError(null);
    try {
      if (gatewayVendor) {
        await updateConfig({
          providers: {
            [GATEWAY_PROVIDER_ID]: {
              gateway_vendor: gatewayVendor,
              api_key: apiKey.trim(),
              model: providerModel ?? undefined,
              selected: true,
            },
          },
        });
      } else {
        await updateConfig({ providers: { [providerId]: { api_key: apiKey.trim() } } });
      }
      setIsSaving(false);
      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save the API key');
      setIsSaving(false);
    }
  }, [apiKey, gatewayVendor, providerId, providerModel, onSaved]);

  const containerCss = {
    border: `1px solid ${theme.colors.borderWarning}`,
    borderRadius: theme.borders.borderRadiusMd,
    backgroundColor: theme.colors.backgroundValidationWarning,
    padding: theme.spacing.md,
    marginBottom: theme.spacing.sm,
    display: 'flex',
    flexDirection: 'column' as const,
    gap: theme.spacing.sm,
  };

  // API keys are stored as server-wide LLM connections, which this caller cannot add.
  if (!canEditServerSettings) {
    return (
      <div css={containerCss}>
        <Typography.Text>
          {isLocalServer ? (
            <FormattedMessage
              defaultMessage="Only an administrator can add the {provider} API key on this server. Ask an administrator, or pick another provider below."
              description="Inline assistant API key prompt shown to a non-admin user, who cannot add API keys"
              values={{ provider: providerName }}
            />
          ) : (
            <FormattedMessage
              defaultMessage="The {provider} API key can only be added from the MLflow server host. Pick another provider below."
              description="Inline assistant API key prompt shown to a remote user, who cannot add API keys"
              values={{ provider: providerName }}
            />
          )}
        </Typography.Text>
      </div>
    );
  }

  return (
    <div css={containerCss}>
      <Typography.Text>
        <FormattedMessage
          defaultMessage="Add your {provider} API key to continue, or pick another provider below."
          description="Explanation shown in the inline assistant API key prompt"
          values={{ provider: providerName }}
        />
      </Typography.Text>

      <SecretInput
        componentId="mlflow.assistant.api_key_prompt.input"
        value={apiKey}
        onChange={(e) => {
          setApiKey(e.target.value);
          if (error) setError(null);
        }}
        placeholder={keyPlaceholderFor(gatewayVendor ?? providerId)}
        allowClear={false}
      />

      <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm - 1 }}>
        <FormattedMessage
          defaultMessage="Your key is stored in MLflow LLM Connections on this server."
          description="Note in the inline assistant API key prompt about how the key is stored"
        />
      </Typography.Text>

      {error && (
        <Typography.Text css={{ color: theme.colors.textValidationDanger, fontSize: theme.typography.fontSizeSm }}>
          {error}
        </Typography.Text>
      )}

      <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          componentId="mlflow.assistant.api_key_prompt.save"
          type="primary"
          onClick={handleSave}
          loading={isSaving}
          disabled={!apiKey.trim()}
        >
          <FormattedMessage
            defaultMessage="Continue"
            description="Confirm button of the inline assistant API key prompt"
          />
        </Button>
      </div>
    </div>
  );
};
