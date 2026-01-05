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
import { formatAuthMethodName, formatCredentialFieldName } from '../../utils/providerUtils';
import { parseAuthConfig } from '../../utils/secretUtils';
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

const AuthConfigDisplay = ({ secret }: { secret: SecretInfo | undefined }) => {
  const { theme } = useDesignSystemTheme();

  const authConfig = useMemo(() => {
    if (!secret) return null;
    return secret.auth_config ?? null;
  }, [secret]);

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

export const SecretSelector = ({ provider, value, onChange, disabled, error }: SecretSelectorProps) => {
  const { theme } = useDesignSystemTheme();
  const { data: secrets, isLoading } = useSecretsQuery({ provider });

  const filteredSecrets = useMemo(
    () => (provider ? secrets?.filter((s) => s.provider === provider) : secrets),
    [provider, secrets],
  );

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
            {parseAuthConfig(selectedSecret)?.['auth_mode'] && (
              <>
                <Typography.Text color="secondary">
                  <FormattedMessage defaultMessage="Auth Type:" description="Auth type label" />
                </Typography.Text>
                <Typography.Text>
                  {formatAuthMethodName(String(parseAuthConfig(selectedSecret)!['auth_mode']))}
                </Typography.Text>
              </>
            )}
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Masked Key:" description="Masked API key label" />
            </Typography.Text>
            <MaskedValueDisplay maskedValue={selectedSecret.masked_values} />
            <AuthConfigDisplay secret={selectedSecret} />
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Last Updated:" description="Label for last updated" />
            </Typography.Text>
            <Typography.Text>
              <TimeAgo date={new Date(selectedSecret.last_updated_at)} />
            </Typography.Text>
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Created:" description="Label for created date" />
            </Typography.Text>
            <Typography.Text>
              <TimeAgo date={new Date(selectedSecret.created_at)} />
            </Typography.Text>
            {selectedSecret.created_by && (
              <>
                <Typography.Text color="secondary">
                  <FormattedMessage defaultMessage="Created by:" description="Label for created by" />
                </Typography.Text>
                <Typography.Text>{selectedSecret.created_by}</Typography.Text>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
