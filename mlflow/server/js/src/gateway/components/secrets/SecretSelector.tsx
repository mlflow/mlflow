import {
  SimpleSelect,
  SimpleSelectOption,
  FormUI,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useMemo } from 'react';
import { useSecretsQuery } from '../../hooks/useSecretsQuery';
import { timestampToDate } from '../../utils/dateUtils';
import { formatAuthMethodName, formatCredentialFieldName } from '../../utils/providerUtils';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import { MaskedValueDisplay } from './MaskedValueDisplay';
import type { SecretInfo } from '../../types';

interface SecretSelectorProps {
  provider?: string;
  value: string;
  onChange: (secretId: string) => void;
  disabled?: boolean;
  error?: string;
}

export const SecretSelector = ({ provider, value, onChange, disabled, error }: SecretSelectorProps) => {
  const { theme } = useDesignSystemTheme();
  const { data: secrets, isLoading } = useSecretsQuery({ provider });

  const filteredSecrets = useMemo(
    () => (provider ? secrets?.filter((s) => s.provider === provider) : secrets),
    [provider, secrets],
  );

  // Find the selected secret for displaying details
  const selectedSecret = useMemo(
    () => (value ? filteredSecrets?.find((s) => s.secret_id === value) : undefined),
    [value, filteredSecrets],
  );

  if (isLoading) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Spinner size="small" />
        <FormattedMessage defaultMessage="Loading API keys..." description="Loading message for API keys" />
      </div>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, width: '100%' }}>
      <div>
        <FormUI.Label htmlFor="mlflow.gateway.create-endpoint.secret-select">
          <FormattedMessage defaultMessage="API key" description="Label for API key selector" />
        </FormUI.Label>
        <SimpleSelect
          id="mlflow.gateway.create-endpoint.secret-select"
          componentId="mlflow.gateway.create-endpoint.secret-select"
          value={value}
          onChange={({ target }) => onChange(target.value)}
          disabled={disabled || !filteredSecrets?.length}
          placeholder={filteredSecrets?.length ? 'Select an API key' : 'No API keys available for this provider'}
          validationState={error ? 'error' : undefined}
          contentProps={{
            matchTriggerWidth: true,
            maxHeight: 300,
          }}
          css={{ width: '100%' }}
        >
          {filteredSecrets?.map((secret) => (
            <SimpleSelectOption key={secret.secret_id} value={secret.secret_id}>
              {secret.secret_name}
            </SimpleSelectOption>
          ))}
        </SimpleSelect>
        {error && <FormUI.Message type="error" message={error} />}
      </div>

      {/* Selected API key details */}
      {selectedSecret && (
        <div
          css={{
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: theme.general.borderRadiusBase,
            padding: theme.spacing.md,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <Typography.Text bold size="sm" color="secondary">
            <FormattedMessage defaultMessage="API key details" description="Header for API key details section" />
          </Typography.Text>
          <div
            css={{
              display: 'grid',
              gridTemplateColumns: 'auto 1fr',
              gap: `${theme.spacing.xs}px ${theme.spacing.md}px`,
              fontSize: theme.typography.fontSizeSm,
              marginTop: theme.spacing.md,
            }}
          >
            {selectedSecret.auth_config?.['auth_mode'] && (
              <>
                <Typography.Text color="secondary">
                  <FormattedMessage defaultMessage="Auth method:" description="Label for auth method" />
                </Typography.Text>
                <Typography.Text>
                  {formatAuthMethodName(String(selectedSecret.auth_config['auth_mode']))}
                </Typography.Text>
              </>
            )}
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Masked key:" description="Label for masked key value" />
            </Typography.Text>
            <MaskedValueDisplay maskedValue={selectedSecret.masked_value} />
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Last updated:" description="Label for last updated" />
            </Typography.Text>
            <Typography.Text>
              <TimeAgo date={timestampToDate(selectedSecret.last_updated_at)} />
            </Typography.Text>
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Created:" description="Label for created date" />
            </Typography.Text>
            <Typography.Text>
              <TimeAgo date={timestampToDate(selectedSecret.created_at)} />
            </Typography.Text>
            {selectedSecret.created_by && (
              <>
                <Typography.Text color="secondary">
                  <FormattedMessage defaultMessage="Created by:" description="Label for created by" />
                </Typography.Text>
                <Typography.Text>{selectedSecret.created_by}</Typography.Text>
              </>
            )}
            <AuthConfigDisplay secret={selectedSecret} />
          </div>
        </div>
      )}
    </div>
  );
};

/** Helper component to display auth config from secret */
const AuthConfigDisplay = ({ secret }: { secret: SecretInfo | undefined }) => {
  const { theme } = useDesignSystemTheme();

  const authConfig = useMemo(() => {
    if (!secret) return null;

    // Parse auth_config_json if it exists, otherwise use auth_config
    if (secret.auth_config_json) {
      try {
        return JSON.parse(secret.auth_config_json) as Record<string, unknown>;
      } catch {
        return null;
      }
    }
    return secret.auth_config ?? null;
  }, [secret]);

  // Filter out auth_mode since it's already shown separately as "Auth method"
  if (!authConfig) return null;
  const configEntries = Object.entries(authConfig).filter(([key]) => key !== 'auth_mode');
  if (configEntries.length === 0) return null;

  return (
    <>
      <Typography.Text color="secondary">
        <FormattedMessage defaultMessage="Config:" description="Auth config label" />
      </Typography.Text>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs / 2 }}>
        {configEntries.map(([key, value]) => (
          <div key={key} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
              {formatCredentialFieldName(key)}:
            </Typography.Text>
            <Typography.Text css={{ fontFamily: 'monospace', fontSize: theme.typography.fontSizeSm }}>
              {String(value)}
            </Typography.Text>
          </div>
        ))}
      </div>
    </>
  );
};
