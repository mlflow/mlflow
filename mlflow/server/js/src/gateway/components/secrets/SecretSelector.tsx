import { SimpleSelect, SimpleSelectOption, FormUI, Spinner, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useSecretsQuery } from '../../hooks/useSecretsQuery';

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

  if (isLoading) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Spinner size="small" />
        <FormattedMessage defaultMessage="Loading secrets..." description="Loading message for secrets" />
      </div>
    );
  }

  const filteredSecrets = provider ? secrets?.filter((s) => s.provider === provider) : secrets;

  return (
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
        css={{ width: '100%' }}
        contentProps={{
          matchTriggerWidth: true,
          maxHeight: 300,
        }}
      >
        {filteredSecrets?.map((secret) => (
          <SimpleSelectOption key={secret.secret_id} value={secret.secret_id}>
            {secret.secret_name}
          </SimpleSelectOption>
        ))}
      </SimpleSelect>
      {error && <FormUI.Message type="error" message={error} />}
    </div>
  );
};
