import { useState, useCallback, useMemo, useEffect } from 'react';
import {
  FormUI,
  Radio,
  Spinner,
  Typography,
  useDesignSystemTheme,
  ChevronDownIcon,
  ChevronRightIcon,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { SecretSelector } from '../../../../../gateway/components/secrets/SecretSelector';
import { SecretInput } from '../../../../../gateway/components/secrets/SecretInput';
import { GatewayInput } from '../../../../../gateway/components/common';
import { formatCredentialFieldName, sortFieldsByProvider } from '../../../../../gateway/utils/providerUtils';
import type { ApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/types';
import type { AuthMode } from '../../../../../gateway/types';

interface IssueDetectionApiKeyConfiguratorProps {
  value: ApiKeyConfiguration;
  onChange: (value: ApiKeyConfiguration) => void;
  provider: string;
  authModes: AuthMode[];
  defaultAuthMode: string | undefined;
  isLoadingProviderConfig: boolean;
  hasExistingSecrets: boolean;
  disabled?: boolean;
  componentIdPrefix?: string;
}

/**
 * Main API key configurator for issue detection.
 * Shows the new/existing mode selector and required credential fields.
 */
export function IssueDetectionApiKeyConfigurator({
  value,
  onChange,
  provider,
  authModes,
  defaultAuthMode,
  isLoadingProviderConfig,
  hasExistingSecrets,
  disabled,
  componentIdPrefix = 'mlflow.traces.issue-detection.api-key',
}: IssueDetectionApiKeyConfiguratorProps) {
  const { theme } = useDesignSystemTheme();

  const selectedAuthMode = useMemo((): AuthMode | undefined => {
    if (!authModes.length) return undefined;
    if (value.newSecret.authMode) {
      const matched = authModes.find((m) => m.mode === value.newSecret.authMode);
      if (matched) return matched;
    }
    return authModes.find((m) => m.mode === defaultAuthMode) ?? authModes[0];
  }, [authModes, value.newSecret.authMode, defaultAuthMode]);

  const requiredFields = useMemo(() => {
    const secretFields = (selectedAuthMode?.secret_fields ?? []).map((field) => ({
      ...field,
      fieldType: 'secret' as const,
    }));
    const configFields = (selectedAuthMode?.config_fields ?? []).map((field) => ({
      ...field,
      fieldType: 'config' as const,
    }));
    const allFields = [...secretFields, ...configFields];
    const sorted = sortFieldsByProvider(allFields, provider);
    return sorted.filter((field) => field.required);
  }, [selectedAuthMode?.secret_fields, selectedAuthMode?.config_fields, provider]);

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

  const handleAuthModeChange = useCallback(
    (authMode: string) => {
      onChange({
        ...value,
        newSecret: {
          ...value.newSecret,
          authMode,
          secretFields: {},
          configFields: {},
        },
      });
    },
    [onChange, value],
  );

  const handleSecretFieldChange = useCallback(
    (fieldName: string, fieldValue: string) => {
      onChange({
        ...value,
        newSecret: {
          ...value.newSecret,
          secretFields: { ...value.newSecret.secretFields, [fieldName]: fieldValue },
        },
      });
    },
    [onChange, value],
  );

  const handleConfigFieldChange = useCallback(
    (fieldName: string, fieldValue: string) => {
      onChange({
        ...value,
        newSecret: {
          ...value.newSecret,
          configFields: { ...value.newSecret.configFields, [fieldName]: fieldValue },
        },
      });
    },
    [onChange, value],
  );

  if (!provider) {
    return (
      <Typography.Text color="secondary">
        <FormattedMessage
          defaultMessage="Select a provider to configure API key"
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
          hideDetails
        />
      ) : (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {requiredFields.map((field) => (
            <FieldInput
              key={field.name}
              field={field}
              value={field.fieldType === 'secret' ? value.newSecret.secretFields : value.newSecret.configFields}
              onChange={field.fieldType === 'secret' ? handleSecretFieldChange : handleConfigFieldChange}
              componentIdPrefix={componentIdPrefix}
              disabled={disabled}
            />
          ))}
          <AuthMethodSelector
            authModes={authModes}
            value={value.newSecret.authMode}
            defaultAuthMode={defaultAuthMode}
            onChange={handleAuthModeChange}
            componentIdPrefix={componentIdPrefix}
            disabled={disabled}
          />
        </div>
      )}
    </div>
  );
}

interface FieldInputProps {
  field: {
    name: string;
    required?: boolean;
    description?: string;
    fieldType: 'secret' | 'config';
  };
  value: Record<string, string>;
  onChange: (fieldName: string, fieldValue: string) => void;
  componentIdPrefix: string;
  disabled?: boolean;
}

function FieldInput({ field, value, onChange, componentIdPrefix, disabled }: FieldInputProps) {
  const { theme } = useDesignSystemTheme();

  return (
    <div>
      <FormUI.Label htmlFor={`${componentIdPrefix}.${field.fieldType}.${field.name}`}>
        {formatCredentialFieldName(field.name)}
        {field.required && <span css={{ color: theme.colors.textValidationDanger }}> *</span>}
      </FormUI.Label>
      {field.fieldType === 'secret' ? (
        <SecretInput
          id={`${componentIdPrefix}.secret.${field.name}`}
          componentId={`${componentIdPrefix}.secret.${field.name}`}
          value={value[field.name] ?? ''}
          onChange={(e) => onChange(field.name, e.target.value)}
          placeholder={field.description}
          disabled={disabled}
        />
      ) : (
        <GatewayInput
          id={`${componentIdPrefix}.config.${field.name}`}
          componentId={`${componentIdPrefix}.config.${field.name}`}
          value={value[field.name] ?? ''}
          onChange={(e) => onChange(field.name, e.target.value)}
          placeholder={field.description}
          disabled={disabled}
        />
      )}
    </div>
  );
}

interface AuthMethodSelectorProps {
  authModes: AuthMode[];
  value: string;
  defaultAuthMode: string | undefined;
  onChange: (mode: string) => void;
  componentIdPrefix: string;
  disabled?: boolean;
}

function AuthMethodSelector({
  authModes,
  value,
  defaultAuthMode,
  onChange,
  componentIdPrefix,
  disabled,
}: AuthMethodSelectorProps) {
  const { theme } = useDesignSystemTheme();
  const [isExpanded, setIsExpanded] = useState(false);

  if (authModes.length <= 1) {
    return null;
  }

  const selectedMode = authModes.find((m) => m.mode === (value || defaultAuthMode));

  return (
    <div css={{ fontSize: theme.typography.fontSizeSm }}>
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          background: 'none',
          border: 'none',
          padding: 0,
          cursor: 'pointer',
          color: theme.colors.textSecondary,
          fontSize: theme.typography.fontSizeSm,
          '&:hover': {
            color: theme.colors.textPrimary,
          },
        }}
      >
        {isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
        <FormattedMessage defaultMessage="Authentication method" description="Label for auth mode selector" />
        {!isExpanded && selectedMode && (
          <span css={{ color: theme.colors.textPrimary }}>: {selectedMode.display_name}</span>
        )}
      </button>
      {isExpanded && (
        <div css={{ marginTop: theme.spacing.sm, marginLeft: theme.spacing.md }}>
          <Radio.Group
            name={`${componentIdPrefix}.auth-mode`}
            componentId={`${componentIdPrefix}.auth-mode-radio-group`}
            value={value || defaultAuthMode}
            onChange={(e) => onChange(e.target.value)}
            disabled={disabled}
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              {authModes.map((mode) => (
                <Radio key={mode.mode} value={mode.mode}>
                  <div>
                    <div css={{ fontWeight: theme.typography.typographyBoldFontWeight }}>{mode.display_name}</div>
                    {mode.description && <div css={{ color: theme.colors.textSecondary }}>{mode.description}</div>}
                  </div>
                </Radio>
              ))}
            </div>
          </Radio.Group>
        </div>
      )}
    </div>
  );
}
