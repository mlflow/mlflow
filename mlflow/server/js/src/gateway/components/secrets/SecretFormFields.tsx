import { Input, FormUI, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useProviderConfigQuery } from '../../hooks/useProviderConfigQuery';
import type { SecretFormData } from './types';

export interface SecretFormFieldsProps {
  /** Provider to fetch auth field configuration for */
  provider: string;
  /** Current form values */
  value: SecretFormData;
  /** Callback when any field changes */
  onChange: (value: SecretFormData) => void;
  /** Field-level errors */
  errors?: {
    name?: string;
    value?: string;
    authConfig?: Record<string, string>;
  };
  /** Whether the form is disabled */
  disabled?: boolean;
  /** Component ID prefix for telemetry */
  componentIdPrefix?: string;
  /** Whether to hide the secret name field (useful for editing where name is fixed) */
  hideNameField?: boolean;
}

/**
 * Controlled form fields for creating or editing a secret.
 * This component can be used standalone or within a larger form.
 */
export const SecretFormFields = ({
  provider,
  value,
  onChange,
  errors,
  disabled,
  componentIdPrefix = 'mlflow.gateway.secret-form',
  hideNameField = false,
}: SecretFormFieldsProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();
  const { data: providerConfig } = useProviderConfigQuery({ provider });

  if (!provider) {
    return (
      <div css={{ color: theme.colors.textSecondary }}>
        <FormattedMessage
          defaultMessage="Select a provider to configure secret"
          description="Message when no provider selected for secret form"
        />
      </div>
    );
  }

  const handleFieldChange = <K extends keyof SecretFormData>(field: K, fieldValue: SecretFormData[K]) => {
    onChange({ ...value, [field]: fieldValue });
  };

  const handleAuthConfigChange = (fieldName: string, fieldValue: string) => {
    onChange({
      ...value,
      authConfig: { ...value.authConfig, [fieldName]: fieldValue },
    });
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {!hideNameField && (
        <div>
          <FormUI.Label htmlFor={`${componentIdPrefix}.name`}>
            <FormattedMessage defaultMessage="Secret name" description="Label for secret name input" />
          </FormUI.Label>
          <Input
            id={`${componentIdPrefix}.name`}
            componentId={`${componentIdPrefix}.name`}
            value={value.name}
            onChange={(e) => handleFieldChange('name', e.target.value)}
            placeholder={formatMessage({
              defaultMessage: 'my-api-key',
              description: 'Placeholder for secret name input',
            })}
            validationState={errors?.name ? 'error' : undefined}
            disabled={disabled}
          />
          {errors?.name && <FormUI.Message type="error" message={errors.name} />}
        </div>
      )}

      <div>
        <FormUI.Label htmlFor={`${componentIdPrefix}.value`}>
          {providerConfig?.credential_name ?? 'API Key'}
        </FormUI.Label>
        <Input
          id={`${componentIdPrefix}.value`}
          componentId={`${componentIdPrefix}.value`}
          type="password"
          autoComplete="off"
          data-1p-ignore
          data-lpignore="true"
          data-form-type="other"
          value={value.value}
          onChange={(e) => handleFieldChange('value', e.target.value)}
          placeholder={formatMessage({
            defaultMessage: 'Enter your API key',
            description: 'Placeholder for secret value input',
          })}
          validationState={errors?.value ? 'error' : undefined}
          disabled={disabled}
        />
        {errors?.value && <FormUI.Message type="error" message={errors.value} />}
      </div>

      {providerConfig?.auth_fields?.map((authField) => (
        <div key={authField.name}>
          <FormUI.Label htmlFor={`${componentIdPrefix}.auth-config.${authField.name}`}>{authField.name}</FormUI.Label>
          <Input
            id={`${componentIdPrefix}.auth-config.${authField.name}`}
            componentId={`${componentIdPrefix}.auth-config.${authField.name}`}
            value={value.authConfig[authField.name] ?? ''}
            onChange={(e) => handleAuthConfigChange(authField.name, e.target.value)}
            placeholder={authField.description}
            validationState={errors?.authConfig?.[authField.name] ? 'error' : undefined}
            disabled={disabled}
          />
          {authField.description && !errors?.authConfig?.[authField.name] && (
            <FormUI.Hint>{authField.description}</FormUI.Hint>
          )}
          {errors?.authConfig?.[authField.name] && (
            <FormUI.Message type="error" message={errors.authConfig[authField.name]} />
          )}
        </div>
      ))}
    </div>
  );
};
