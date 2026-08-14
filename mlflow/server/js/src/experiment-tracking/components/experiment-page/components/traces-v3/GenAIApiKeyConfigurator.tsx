import { useState, useCallback, useMemo, useEffect } from 'react';
import {
  FormUI,
  Radio,
  Spinner,
  Typography,
  useDesignSystemTheme,
  ChevronDownIcon,
  ChevronRightIcon,
  TypeaheadComboboxRoot,
  TypeaheadComboboxInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxMenuItem,
  useComboboxState,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { SecretSelector } from '../../../../../gateway/components/secrets/SecretSelector';
import { SecretInput } from '../../../../../gateway/components/secrets/SecretInput';
import { GatewayInput } from '../../../../../gateway/components/common';
import { formatCredentialFieldName, sortFieldsByProvider } from '../../../../../gateway/utils/providerUtils';
import type { ApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/types';
import type { AuthMode, SecretInfo } from '../../../../../gateway/types';
import { useSecretsQuery } from '../../../../../gateway/hooks/useSecretsQuery';

interface ApiKeyOption {
  type: 'existing' | 'new';
  secretId?: string;
  secretName?: string;
  value: string;
}

/**
 * Helper function to get the selected auth mode from available modes.
 * Returns the matched mode if currentAuthMode is provided and found,
 * otherwise returns the default mode or the first available mode.
 */
function getSelectedAuthMode(
  authModes: AuthMode[],
  currentAuthMode: string | undefined,
  defaultAuthMode: string | undefined,
): AuthMode | undefined {
  if (!authModes.length) return undefined;
  if (currentAuthMode) {
    const matched = authModes.find((m) => m.mode === currentAuthMode);
    if (matched) return matched;
  }
  return authModes.find((m) => m.mode === defaultAuthMode) ?? authModes[0];
}

interface GenAIApiKeyConfiguratorProps {
  value: ApiKeyConfiguration;
  onChange: (value: ApiKeyConfiguration) => void;
  provider: string;
  authModes: AuthMode[];
  defaultAuthMode: string | undefined;
  isLoadingProviderConfig: boolean;
  hasExistingSecrets: boolean;
  disabled?: boolean;
}

/**
 * Main API key configurator for GenAI model selection.
 * Shows the new/existing mode selector and required credential fields.
 */
export function GenAIApiKeyConfigurator({
  value,
  onChange,
  provider,
  authModes,
  defaultAuthMode,
  isLoadingProviderConfig,
  hasExistingSecrets,
  disabled,
}: GenAIApiKeyConfiguratorProps) {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { data: secrets } = useSecretsQuery({ provider });

  const filteredSecrets = useMemo(
    () => (provider ? secrets?.filter((s) => s.provider === provider) : secrets) ?? [],
    [provider, secrets],
  );

  const selectedAuthMode = useMemo(
    () => getSelectedAuthMode(authModes, value.newSecret.authMode, defaultAuthMode),
    [authModes, value.newSecret.authMode, defaultAuthMode],
  );

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

  const isSingleApiKeyField = useMemo(() => {
    return requiredFields.length === 1 && requiredFields[0].fieldType === 'secret' && authModes.length <= 1;
  }, [requiredFields, authModes.length]);

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
      onChange({ ...value, mode: 'existing', existingSecretId: secretId });
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

  if (isSingleApiKeyField) {
    return (
      <SimplifiedApiKeyInput
        value={value}
        onChange={onChange}
        fieldName={requiredFields[0].name}
        secrets={filteredSecrets}
        disabled={disabled}
      />
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <Radio.Group
        componentId="mlflow.traces.issue-detection.api-key.mode"
        name="mlflow.traces.issue-detection.api-key.mode"
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
              disabled={disabled}
            />
          ))}
          <AuthMethodSelector
            authModes={authModes}
            value={value.newSecret.authMode}
            defaultAuthMode={defaultAuthMode}
            onChange={handleAuthModeChange}
            disabled={disabled}
          />
        </div>
      )}
    </div>
  );
}

interface GenAIAdvancedApiKeySettingsProps {
  value: ApiKeyConfiguration;
  onChange: (value: ApiKeyConfiguration) => void;
  provider: string;
  authModes: AuthMode[];
  defaultAuthMode: string | undefined;
  disabled?: boolean;
}

/**
 * Advanced API key settings for GenAI model selection.
 * Shows optional credential fields only.
 */
export function GenAIAdvancedApiKeySettings({
  value,
  onChange,
  provider,
  authModes,
  defaultAuthMode,
  disabled,
}: GenAIAdvancedApiKeySettingsProps) {
  const { theme } = useDesignSystemTheme();

  const selectedAuthMode = useMemo(
    () => getSelectedAuthMode(authModes, value.newSecret.authMode, defaultAuthMode),
    [authModes, value.newSecret.authMode, defaultAuthMode],
  );

  const optionalFields = useMemo(() => {
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
    return sorted.filter((field) => !field.required);
  }, [selectedAuthMode?.secret_fields, selectedAuthMode?.config_fields, provider]);

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

  if (optionalFields.length === 0) {
    return null;
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {optionalFields.map((field) => (
        <FieldInput
          key={field.name}
          field={field}
          value={field.fieldType === 'secret' ? value.newSecret.secretFields : value.newSecret.configFields}
          onChange={field.fieldType === 'secret' ? handleSecretFieldChange : handleConfigFieldChange}
          disabled={disabled}
        />
      ))}
    </div>
  );
}

interface SimplifiedApiKeyInputProps {
  value: ApiKeyConfiguration;
  onChange: (value: ApiKeyConfiguration) => void;
  fieldName: string;
  secrets: SecretInfo[];
  disabled?: boolean;
}

function SimplifiedApiKeyInput({ value, onChange, fieldName, secrets, disabled }: SimplifiedApiKeyInputProps) {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const items = useMemo((): ApiKeyOption[] => {
    return secrets.map((secret) => ({
      type: 'existing' as const,
      secretId: secret.secret_id,
      secretName: secret.secret_name,
      value: secret.secret_id,
    }));
  }, [secrets]);

  const [filteredItems, setFilteredItems] = useState<(ApiKeyOption | null)[]>(items);
  const [inputValue, setInputValue] = useState(
    value.mode === 'existing'
      ? (secrets.find((s) => s.secret_id === value.existingSecretId)?.secret_name ?? '')
      : (value.newSecret.secretFields[fieldName] ?? ''),
  );

  useEffect(() => {
    setFilteredItems(items);
  }, [items]);

  useEffect(() => {
    if (value.mode === 'existing') {
      const secretName = secrets.find((s) => s.secret_id === value.existingSecretId)?.secret_name ?? '';
      setInputValue(secretName);
    }
  }, [value.mode, value.existingSecretId, secrets]);

  const selectedItem = useMemo((): ApiKeyOption | null => {
    if (value.mode === 'existing' && value.existingSecretId) {
      const secret = secrets.find((s) => s.secret_id === value.existingSecretId);
      if (secret) {
        return {
          type: 'existing',
          secretId: secret.secret_id,
          secretName: secret.secret_name,
          value: secret.secret_id,
        };
      }
    }
    return null;
  }, [value.mode, value.existingSecretId, secrets]);

  const handleFormChange = useCallback(
    (item: ApiKeyOption | null) => {
      if (item?.type === 'existing' && item.secretId) {
        onChange({
          ...value,
          mode: 'existing',
          existingSecretId: item.secretId,
        });
      }
    },
    [onChange, value],
  );

  const handleInputValueChange = useCallback(
    (newInputValue: string) => {
      const matchingSecret = secrets.find(
        (s) => s.secret_name.toLowerCase() === newInputValue.toLowerCase() || s.secret_id === newInputValue,
      );

      if (matchingSecret) {
        onChange({
          ...value,
          mode: 'existing',
          existingSecretId: matchingSecret.secret_id,
        });
      } else {
        onChange({
          ...value,
          mode: 'new',
          existingSecretId: '',
          newSecret: {
            ...value.newSecret,
            secretFields: { [fieldName]: newInputValue },
          },
        });
      }
    },
    [onChange, value, fieldName, secrets],
  );

  const wrappedSetInputValue: React.Dispatch<React.SetStateAction<string>> = useCallback(
    (action) => {
      const newValue = typeof action === 'function' ? action(inputValue) : action;
      setInputValue(newValue);
      handleInputValueChange(newValue);
    },
    [inputValue, handleInputValueChange],
  );

  const comboboxComponentId = `mlflow.traces.issue-detection.api-key.api-key-combobox`;

  const comboboxState = useComboboxState<ApiKeyOption | null>({
    componentId: comboboxComponentId,
    allItems: items,
    items: filteredItems,
    setItems: setFilteredItems,
    multiSelect: false,
    setInputValue: wrappedSetInputValue,
    itemToString: (item) => item?.secretName ?? '',
    matcher: (item, query) => item?.secretName?.toLowerCase().includes(query.toLowerCase()) ?? false,
    formValue: selectedItem,
    formOnChange: handleFormChange,
  });

  const isUsingExistingKey = value.mode === 'existing' && value.existingSecretId;

  return (
    <div>
      <Typography.Text
        color="secondary"
        css={{ display: 'block', marginBottom: theme.spacing.xs, fontSize: theme.typography.fontSizeSm }}
      >
        {formatCredentialFieldName(fieldName)}
      </Typography.Text>
      <TypeaheadComboboxRoot id={comboboxComponentId} comboboxState={comboboxState}>
        <TypeaheadComboboxInput
          placeholder={intl.formatMessage({
            defaultMessage: 'Enter API key or select saved key',
            description: 'Placeholder for API key combobox input',
          })}
          comboboxState={comboboxState}
          formOnChange={handleFormChange}
          disabled={disabled}
          showComboboxToggleButton={items.length > 0}
          type={isUsingExistingKey ? 'text' : 'password'}
        />
        {filteredItems.length > 0 && (
          <TypeaheadComboboxMenu comboboxState={comboboxState}>
            {filteredItems.map(
              (item, index) =>
                item && (
                  <TypeaheadComboboxMenuItem
                    key={item.secretId}
                    item={item}
                    index={index}
                    comboboxState={comboboxState}
                  >
                    {item.secretName}
                  </TypeaheadComboboxMenuItem>
                ),
            )}
          </TypeaheadComboboxMenu>
        )}
      </TypeaheadComboboxRoot>
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
  disabled?: boolean;
}

function FieldInput({ field, value, onChange, disabled }: FieldInputProps) {
  const { theme } = useDesignSystemTheme();
  const fieldId = `mlflow.traces.issue-detection.api-key.${field.fieldType}.${field.name}`;

  return (
    <div>
      <Typography.Text
        color="secondary"
        css={{ display: 'block', marginBottom: theme.spacing.xs, fontSize: theme.typography.fontSizeSm }}
      >
        {formatCredentialFieldName(field.name)}
        {field.required && <span css={{ color: theme.colors.textValidationDanger }}> *</span>}
      </Typography.Text>
      {field.fieldType === 'secret' ? (
        <SecretInput
          id={fieldId}
          componentId="mlflow.traces.issue-detection.api-key.secret-input"
          value={value[field.name] ?? ''}
          onChange={(e) => onChange(field.name, e.target.value)}
          placeholder={field.description}
          disabled={disabled}
        />
      ) : (
        <GatewayInput
          id={fieldId}
          componentId="mlflow.traces.issue-detection.api-key.config-input"
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
  disabled?: boolean;
}

function AuthMethodSelector({ authModes, value, defaultAuthMode, onChange, disabled }: AuthMethodSelectorProps) {
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
            name="mlflow.traces.issue-detection.api-key.auth-mode"
            componentId="mlflow.traces.issue-detection.api-key.auth-mode-radio-group"
            value={value || defaultAuthMode}
            onChange={(e) => onChange(e.target.value)}
            disabled={disabled}
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              {authModes.map((mode) => (
                <Radio key={mode.mode} value={mode.mode}>
                  <div css={{ fontSize: theme.typography.fontSizeSm }}>
                    <div>{mode.display_name}</div>
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
    </div>
  );
}
