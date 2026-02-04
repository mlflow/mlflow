import { useCallback, useMemo, useEffect } from 'react';
import { FormUI, Radio, Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { GatewayInput } from '../../common';
import { SecretInput } from '../../secrets/SecretInput';
import { SecretSelector } from '../../secrets/SecretSelector';
import { formatCredentialFieldName, sortFieldsByProvider } from '../../../utils/providerUtils';
import type { ApiKeyConfiguration, NewSecretData } from '../types';
import type { SecretInfo, AuthMode, SecretField, ConfigField } from '../../../types';

interface ApiKeyConfiguratorProps {
  value: ApiKeyConfiguration;
  onChange: (value: ApiKeyConfiguration) => void;
  provider: string;
  existingSecrets: SecretInfo[];
  isLoadingSecrets: boolean;
  authModes: AuthMode[];
  defaultAuthMode: string | undefined;
  isLoadingProviderConfig: boolean;
  errors?: {
    existingSecretId?: string;
    newSecret?: {
      name?: string;
      secretFields?: Record<string, string>;
      configFields?: Record<string, string>;
    };
  };
  disabled?: boolean;
  componentIdPrefix?: string;
}

export function ApiKeyConfigurator({
  value,
  onChange,
  provider,
  existingSecrets,
  isLoadingSecrets,
  authModes,
  defaultAuthMode,
  isLoadingProviderConfig,
  errors,
  disabled,
  componentIdPrefix = 'mlflow.gateway.api-key-config',
}: ApiKeyConfiguratorProps) {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();

  const hasExistingSecrets = existingSecrets.length > 0;
  const hasMultipleAuthModes = authModes.length > 1;

  const selectedAuthMode = useMemo((): AuthMode | undefined => {
    if (!authModes.length) return undefined;
    if (value.newSecret.authMode) {
      const matched = authModes.find((m) => m.mode === value.newSecret.authMode);
      if (matched) return matched;
    }
    return authModes.find((m) => m.mode === defaultAuthMode) ?? authModes[0];
  }, [authModes, value.newSecret.authMode, defaultAuthMode]);

  // Auto-set authMode when it's empty and auth modes are available to ensure the form validation passes
  useEffect(() => {
    if (value.mode === 'new' && !value.newSecret.authMode && selectedAuthMode) {
      onChange({
        ...value,
        newSecret: {
          ...value.newSecret,
          authMode: selectedAuthMode.mode,
        },
      });
    }
  }, [value, selectedAuthMode, onChange]);

  const handleModeChange = useCallback(
    (mode: 'new' | 'existing') => {
      onChange({ ...value, mode });
    },
    [onChange, value],
  );

  const handleExistingSecretSelect = useCallback(
    (secretId: string) => {
      onChange({ ...value, existingSecretId: secretId });
    },
    [onChange, value],
  );

  const handleNewSecretChange = useCallback(
    (newSecret: NewSecretData) => {
      onChange({ ...value, newSecret });
    },
    [onChange, value],
  );

  const handleNameChange = useCallback(
    (name: string) => {
      handleNewSecretChange({ ...value.newSecret, name });
    },
    [handleNewSecretChange, value.newSecret],
  );

  const handleAuthModeChange = useCallback(
    (authMode: string) => {
      handleNewSecretChange({
        ...value.newSecret,
        authMode,
        secretFields: {},
        configFields: {},
      });
    },
    [handleNewSecretChange, value.newSecret],
  );

  const handleSecretFieldChange = useCallback(
    (fieldName: string, fieldValue: string) => {
      handleNewSecretChange({
        ...value.newSecret,
        secretFields: { ...value.newSecret.secretFields, [fieldName]: fieldValue },
      });
    },
    [handleNewSecretChange, value.newSecret],
  );

  const handleConfigFieldChange = useCallback(
    (fieldName: string, fieldValue: string) => {
      handleNewSecretChange({
        ...value.newSecret,
        configFields: { ...value.newSecret.configFields, [fieldName]: fieldValue },
      });
    },
    [handleNewSecretChange, value.newSecret],
  );

  if (!provider) {
    return (
      <Typography.Text color="secondary">
        <FormattedMessage
          defaultMessage="Select a provider and model to configure API key"
          description="Message when no provider selected for API key form"
        />
      </Typography.Text>
    );
  }

  if (isLoadingProviderConfig) {
    return (
      <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.md }}>
        <Spinner size="small" />
      </div>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <Radio.Group
        componentId={`${componentIdPrefix}.mode`}
        name={`${componentIdPrefix}.mode`}
        value={value.mode}
        onChange={(e) => handleModeChange(e.target.value as 'new' | 'existing')}
        layout="horizontal"
        disabled={disabled}
      >
        <Radio value="new">
          <FormattedMessage defaultMessage="Create new API key" description="Option to create new API key" />
        </Radio>
        <Radio value="existing" disabled={!hasExistingSecrets}>
          <FormattedMessage defaultMessage="Use existing API key" description="Option to use existing API key" />
        </Radio>
      </Radio.Group>

      {!hasExistingSecrets && (
        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
          <FormattedMessage
            defaultMessage="No existing API keys for this provider."
            description="Message when no existing API keys"
          />
        </Typography.Text>
      )}

      {value.mode === 'existing' ? (
        <SecretSelector
          provider={provider}
          value={value.existingSecretId}
          onChange={handleExistingSecretSelect}
          disabled={disabled}
          error={errors?.existingSecretId}
        />
      ) : (
        <NewSecretForm
          value={value.newSecret}
          provider={provider}
          selectedAuthMode={selectedAuthMode}
          authModes={authModes}
          hasMultipleAuthModes={hasMultipleAuthModes}
          defaultAuthMode={defaultAuthMode}
          errors={errors?.newSecret}
          disabled={disabled}
          componentIdPrefix={componentIdPrefix}
          onNameChange={handleNameChange}
          onAuthModeChange={handleAuthModeChange}
          onSecretFieldChange={handleSecretFieldChange}
          onConfigFieldChange={handleConfigFieldChange}
        />
      )}
    </div>
  );
}

interface NewSecretFormProps {
  value: NewSecretData;
  provider: string;
  selectedAuthMode: AuthMode | undefined;
  authModes: AuthMode[];
  hasMultipleAuthModes: boolean;
  defaultAuthMode: string | undefined;
  errors?: {
    name?: string;
    secretFields?: Record<string, string>;
    configFields?: Record<string, string>;
  };
  disabled?: boolean;
  componentIdPrefix: string;
  onNameChange: (name: string) => void;
  onAuthModeChange: (mode: string) => void;
  onSecretFieldChange: (fieldName: string, fieldValue: string) => void;
  onConfigFieldChange: (fieldName: string, fieldValue: string) => void;
}

function NewSecretForm({
  value,
  provider,
  selectedAuthMode,
  authModes,
  hasMultipleAuthModes,
  defaultAuthMode,
  errors,
  disabled,
  componentIdPrefix,
  onNameChange,
  onAuthModeChange,
  onSecretFieldChange,
  onConfigFieldChange,
}: NewSecretFormProps) {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();

  // Combine secret and config fields into a single sorted list
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

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div>
        <FormUI.Label htmlFor={`${componentIdPrefix}.name`}>
          <FormattedMessage defaultMessage="API key name" description="Label for API key name input" />
          <span css={{ color: theme.colors.textValidationDanger }}> *</span>
        </FormUI.Label>
        <GatewayInput
          id={`${componentIdPrefix}.name`}
          componentId={`${componentIdPrefix}.name`}
          value={value.name}
          onChange={(e) => onNameChange(e.target.value)}
          placeholder={formatMessage({
            defaultMessage: 'my-api-key',
            description: 'Placeholder for API key name input',
          })}
          validationState={errors?.name ? 'error' : undefined}
          disabled={disabled}
        />
        {errors?.name ? (
          <FormUI.Message type="error" message={errors.name} />
        ) : (
          <FormUI.Hint>
            <FormattedMessage
              defaultMessage="A unique name to identify this API key for reuse across endpoints"
              description="Hint text explaining API key name field"
            />
          </FormUI.Hint>
        )}
      </div>

      {hasMultipleAuthModes && (
        <div>
          <FormUI.Label>
            <FormattedMessage defaultMessage="Authentication method" description="Label for auth mode selector" />
          </FormUI.Label>
          <Radio.Group
            name={`${componentIdPrefix}.auth-mode`}
            componentId={`${componentIdPrefix}.auth-mode`}
            value={value.authMode || defaultAuthMode}
            onChange={(e) => onAuthModeChange(e.target.value)}
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
          <FormUI.Label htmlFor={`${componentIdPrefix}.${field.fieldType}.${field.name}`}>
            {formatCredentialFieldName(field.name)}
            {field.required && <span css={{ color: theme.colors.textValidationDanger }}> *</span>}
          </FormUI.Label>
          {field.fieldType === 'secret' ? (
            <SecretInput
              id={`${componentIdPrefix}.secret.${field.name}`}
              componentId={`${componentIdPrefix}.secret.${field.name}`}
              value={value.secretFields[field.name] ?? ''}
              onChange={(e) => onSecretFieldChange(field.name, e.target.value)}
              placeholder={field.description}
              validationState={errors?.secretFields?.[field.name] ? 'error' : undefined}
              disabled={disabled}
            />
          ) : (
            <GatewayInput
              id={`${componentIdPrefix}.config.${field.name}`}
              componentId={`${componentIdPrefix}.config.${field.name}`}
              value={value.configFields[field.name] ?? ''}
              onChange={(e) => onConfigFieldChange(field.name, e.target.value)}
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
}
