import type { ChangeEvent } from 'react';
import { FormUI, Spinner, useDesignSystemTheme, Radio, type RadioChangeEvent } from '@databricks/design-system';
import { GatewayInput } from '../common';
import { useMemo, useCallback, useRef } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useProviderConfigQuery } from '../../hooks/useProviderConfigQuery';
import { formatCredentialFieldName, sortFieldsByProvider } from '../../utils/providerUtils';
import { SecretInput } from './SecretInput';
import type { AuthMode } from '../../types';
import type { SecretFormData } from './types';

export interface SecretFormFieldsProps {
  provider: string;
  value: SecretFormData;
  onChange: (value: SecretFormData) => void;
  errors?: {
    name?: string;
    secretFields?: Record<string, string>;
    configFields?: Record<string, string>;
  };
  disabled?: boolean;
  componentId?: string;
  hideNameField?: boolean;
  secretPlaceholders?: Record<string, string>;
}

export const SecretFormFields = ({
  provider,
  value,
  onChange,
  errors,
  disabled,
  componentId = 'mlflow.gateway.secret-form',
  hideNameField = false,
  secretPlaceholders,
}: SecretFormFieldsProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();
  const domId = useRef(`secret-form-${Math.random().toString(36).slice(2, 9)}`).current;
  const { data: providerConfig, isLoading } = useProviderConfigQuery({ provider });

  const authModes = useMemo(() => providerConfig?.auth_modes ?? [], [providerConfig?.auth_modes]);
  const hasMultipleModes = authModes.length > 1;

  const selectedAuthMode = useMemo((): AuthMode | undefined => {
    if (!authModes.length) return undefined;
    if (value.authMode) {
      const matched = authModes.find((m) => m.mode === value.authMode);
      if (matched) return matched;
    }
    return authModes.find((m) => m.mode === providerConfig?.default_mode) ?? authModes[0];
  }, [authModes, value.authMode, providerConfig?.default_mode]);

  const effectiveAuthMode = value.authMode || selectedAuthMode?.mode || '';

  const sortedFields = useMemo(() => {
    const secretFields = (selectedAuthMode?.secret_fields ?? []).map((field) => ({
      ...field,
      fieldType: 'secret' as const,
    }));
    const configFields = (selectedAuthMode?.config_fields ?? []).map((field) => ({
      ...field,
      fieldType: 'config' as const,
    }));
    const allFields = [...secretFields, ...configFields];
    return sortFieldsByProvider(allFields, provider);
  }, [selectedAuthMode?.secret_fields, selectedAuthMode?.config_fields, provider]);

  const handleNameChange = useCallback(
    (newName: string) => {
      onChange({ ...value, name: newName });
    },
    [onChange, value],
  );

  const handleSecretFieldChange = useCallback(
    (fieldName: string, fieldValue: string) => {
      onChange({
        ...value,
        secretFields: { ...value.secretFields, [fieldName]: fieldValue },
      });
    },
    [onChange, value],
  );

  const handleConfigFieldChange = useCallback(
    (fieldName: string, fieldValue: string) => {
      onChange({
        ...value,
        configFields: { ...value.configFields, [fieldName]: fieldValue },
      });
    },
    [onChange, value],
  );

  const handleAuthModeChange = useCallback(
    (mode: string) => {
      onChange({
        ...value,
        authMode: mode,
        secretFields: {},
        configFields: {},
      });
    },
    [onChange, value],
  );

  if (!provider) {
    return (
      <div css={{ color: theme.colors.textSecondary }}>
        <FormattedMessage
          defaultMessage="Select a provider to configure API key"
          description="Message when no provider selected for API key form"
        />
      </div>
    );
  }

  if (isLoading) {
    return (
      <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg }}>
        <Spinner size="small" />
      </div>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {!hideNameField && (
        <div>
          <FormUI.Label htmlFor={`${domId}.name`}>
            <FormattedMessage defaultMessage="API key name" description="Label for API key name input" />
            <span css={{ color: theme.colors.textValidationDanger }}> *</span>
          </FormUI.Label>
          <GatewayInput
            id={`${domId}.name`}
            componentId={`${componentId}.name`}
            value={value.name}
            onChange={(e: ChangeEvent<HTMLInputElement>) => handleNameChange(e.target.value)}
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

      {hasMultipleModes && (
        <div>
          <FormUI.Label>
            <FormattedMessage defaultMessage="Authentication method" description="Label for auth mode selector" />
          </FormUI.Label>
          <Radio.Group
            name={`${domId}.auth-mode`}
            componentId={`${componentId}.auth-mode`}
            value={effectiveAuthMode}
            onChange={(e: RadioChangeEvent) => handleAuthModeChange(e.target.value)}
            disabled={disabled}
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              {authModes.map((mode) => (
                <Radio key={mode.mode} value={mode.mode}>
                  <div>
                    <div css={{ fontWeight: theme.typography.typographyBoldFontWeight }}>{mode.display_name}</div>
                    {mode.description && (
                      <div css={{ color: theme.colors.textSecondary, fontSize: theme.typography.fontSizeSm }}>
                        {mode.description}
                      </div>
                    )}
                  </div>
                </Radio>
              ))}
            </div>
          </Radio.Group>
        </div>
      )}

      {sortedFields.map((field) => (
        <div key={field.name}>
          <FormUI.Label htmlFor={`${domId}.${field.fieldType}.${field.name}`}>
            {formatCredentialFieldName(field.name)}
            {field.required && <span css={{ color: theme.colors.textValidationDanger }}> *</span>}
          </FormUI.Label>
          {field.fieldType === 'secret' ? (
            <SecretInput
              id={`${domId}.secret.${field.name}`}
              componentId={`${componentId}.secret`}
              value={value.secretFields[field.name] ?? ''}
              onChange={(e: ChangeEvent<HTMLInputElement>) => handleSecretFieldChange(field.name, e.target.value)}
              placeholder={secretPlaceholders?.[field.name] ?? field.description}
              validationState={errors?.secretFields?.[field.name] ? 'error' : undefined}
              disabled={disabled}
            />
          ) : (
            <GatewayInput
              id={`${domId}.config.${field.name}`}
              componentId={`${componentId}.config`}
              value={value.configFields[field.name] ?? ''}
              onChange={(e: ChangeEvent<HTMLInputElement>) => handleConfigFieldChange(field.name, e.target.value)}
              placeholder={field.description}
              validationState={errors?.configFields?.[field.name] ? 'error' : undefined}
              disabled={disabled}
            />
          )}
          {field.fieldType === 'secret' && errors?.secretFields?.[field.name] && (
            <FormUI.Message type="error" message={errors.secretFields[field.name]} />
          )}
          {field.fieldType === 'config' && errors?.configFields?.[field.name] && (
            <FormUI.Message type="error" message={errors.configFields[field.name]} />
          )}
        </div>
      ))}
    </div>
  );
};
