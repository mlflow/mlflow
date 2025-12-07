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
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import { formatAuthMethodName } from '../../utils/providerUtils';
import { parseAuthConfig } from '../../utils/secretUtils';

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
        <FormattedMessage defaultMessage="Loading secrets..." description="Loading message for secrets" />
      </div>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, width: '100%' }}>
      <div>
        <FormUI.Label htmlFor="mlflow.gateway.create-endpoint.secret-select">
          <FormattedMessage defaultMessage="Secret" description="Label for secret selector" />
        </FormUI.Label>
        <SimpleSelect
          id="mlflow.gateway.create-endpoint.secret-select"
          componentId="mlflow.gateway.create-endpoint.secret-select"
          value={value}
          onChange={({ target }) => onChange(target.value)}
          disabled={disabled || !filteredSecrets?.length}
          placeholder={filteredSecrets?.length ? 'Select a secret' : 'No secrets available for this provider'}
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

      {/* Selected secret details */}
      {selectedSecret && (
        <div
          css={{
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: theme.general.borderRadiusBase,
            padding: theme.spacing.md,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.xs,
          }}
        >
          <Typography.Text bold size="sm" color="secondary">
            <FormattedMessage defaultMessage="Secret details" description="Header for secret details section" />
          </Typography.Text>
          <div
            css={{
              display: 'grid',
              gridTemplateColumns: 'auto 1fr',
              gap: `${theme.spacing.xs}px ${theme.spacing.md}px`,
              fontSize: theme.typography.fontSizeSm,
            }}
          >
            {parseAuthConfig(selectedSecret)?.['auth_mode'] && (
              <>
                <Typography.Text color="secondary">
                  <FormattedMessage defaultMessage="Auth method:" description="Label for auth method" />
                </Typography.Text>
                <Typography.Text>
                  {formatAuthMethodName(String(parseAuthConfig(selectedSecret)!['auth_mode']))}
                </Typography.Text>
              </>
            )}
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
          </div>
        </div>
      )}
    </div>
  );
};
