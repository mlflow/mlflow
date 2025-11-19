import {
  Button,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  FormUI,
  Input,
  Modal,
  Radio,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useState, useMemo, useEffect } from 'react';
import type { Endpoint } from '../types';
import { useListSecrets } from '../hooks/useListSecrets';
import { useListBindings } from '../hooks/useListBindings';
import { useListEndpoints } from '../hooks/useListEndpoints';
import { MaskedApiKeyInput } from './MaskedApiKeyInput';
import { AuthConfigFields } from './AuthConfigFields';
import { PROVIDERS } from './routeConstants';

export interface UpdateRouteModalProps {
  route: Endpoint | null;
  visible: boolean;
  onCancel: () => void;
  onUpdate?: (
    routeId: string,
    updateData: {
      secret_id?: string;
      secret_name?: string;
      secret_value?: string;
      provider?: string;
      auth_config?: string;
    },
  ) => void;
}

type SecretSource = 'existing' | 'new';

export const UpdateRouteModal = ({ route, visible, onCancel, onUpdate }: UpdateRouteModalProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { secrets = [] } = useListSecrets({ enabled: visible });
  const { endpoints = [] } = useListEndpoints({ enabled: visible });

  // Get bindings for the current secret to show affected resources
  const { bindings = [] } = useListBindings({
    secretId: route?.secret_id || '',
    enabled: visible && !!route?.secret_id,
  });

  const [secretSource, setSecretSource] = useState<SecretSource>('existing');
  const [selectedSecretId, setSelectedSecretId] = useState<string>('');
  const [newSecretName, setNewSecretName] = useState<string>('');
  const [newSecretValue, setNewSecretValue] = useState<string>('');
  const [newSecretProvider, setNewSecretProvider] = useState<string>('');
  const [authConfig, setAuthConfig] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<{
    secretId?: string;
    secretName?: string;
    secretValue?: string;
    provider?: string;
  }>({});
  const [isLoading, setIsLoading] = useState(false);

  // Reset state when modal opens with a new route
  useEffect(() => {
    if (route && visible) {
      setSelectedSecretId(''); // Start with no selection
      setSecretSource('existing');
      setNewSecretName('');
      setNewSecretValue('');
      // Initialize provider from current secret or route
      const currentSecret = secrets.find((s) => s.secret_id === route.secret_id);
      setNewSecretProvider(route.provider || currentSecret?.provider || '');
      setAuthConfig({});
      setErrors({});
    }
  }, [route, visible, secrets]);

  // Filter secrets to only show those from the same provider
  const compatibleSecrets = useMemo(() => {
    if (!route) return [];

    // If route has no provider, show all secrets
    if (!route.provider) return secrets;

    const routeProvider = route.provider;

    return secrets.filter((secret) => {
      // Use the provider field from the backend if available
      if (secret.provider) {
        return secret.provider.toLowerCase() === routeProvider.toLowerCase();
      }

      // Fallback to pattern matching on secret_name for backwards compatibility
      const secretNameLower = secret.secret_name.toLowerCase();
      const providerLower = routeProvider.toLowerCase();

      if (providerLower === 'openai') return secretNameLower.includes('openai');
      if (providerLower === 'anthropic')
        return secretNameLower.includes('anthropic') || secretNameLower.includes('claude');
      if (providerLower === 'bedrock') return secretNameLower.includes('bedrock') || secretNameLower.includes('aws');
      if (providerLower === 'vertex_ai')
        return (
          secretNameLower.includes('vertex') || secretNameLower.includes('gemini') || secretNameLower.includes('google')
        );
      if (providerLower === 'azure') return secretNameLower.includes('azure');
      if (providerLower === 'databricks') return secretNameLower.includes('databricks');

      return true; // For custom providers, show all
    });
  }, [secrets, route]);

  // Find endpoints that use the current secret (will be affected by changing the secret)
  const endpointsUsingCurrentSecret = useMemo(() => {
    if (!route) return [];
    return endpoints.filter((r) => r.secret_id === route.secret_id && r.endpoint_id !== route.endpoint_id);
  }, [endpoints, route]);

  // Find bindings for this specific route (resources bound to this route)
  const routeBindings = useMemo(() => {
    if (!route) return [];
    return bindings.filter((b) => b.endpoint_id === route.endpoint_id);
  }, [bindings, route]);

  // Get the selected provider info with auth config fields
  const selectedProviderInfo = useMemo(() => {
    return PROVIDERS.find((p) => p.value === newSecretProvider);
  }, [newSecretProvider]);

  const handleUpdate = async () => {
    if (!route) return;

    const newErrors: { secretId?: string; secretName?: string; secretValue?: string; provider?: string } = {};

    if (secretSource === 'existing') {
      if (!selectedSecretId) {
        newErrors.secretId = intl.formatMessage({
          defaultMessage: 'Please select an API key',
          description: 'Secret selection required error',
        });
      } else if (selectedSecretId === route.secret_id) {
        // No change, just close
        onCancel();
        return;
      }
    } else {
      // New secret validation
      if (!newSecretProvider) {
        newErrors.provider = intl.formatMessage({
          defaultMessage: 'Provider is required',
          description: 'Provider required error',
        });
      }
      if (!newSecretName.trim()) {
        newErrors.secretName = intl.formatMessage({
          defaultMessage: 'Secret name is required',
          description: 'Secret name required error',
        });
      }
      if (!newSecretValue.trim()) {
        newErrors.secretValue = intl.formatMessage({
          defaultMessage: 'API key is required',
          description: 'API key required error',
        });
      }
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    setIsLoading(true);
    try {
      if (secretSource === 'existing') {
        await onUpdate?.(route.endpoint_id, { secret_id: selectedSecretId });
      } else {
        await onUpdate?.(route.endpoint_id, {
          secret_name: newSecretName,
          secret_value: newSecretValue,
          provider: newSecretProvider || undefined,
          auth_config: Object.keys(authConfig).length > 0 ? JSON.stringify(authConfig) : undefined,
        });
      }
      onCancel();
    } finally {
      setIsLoading(false);
    }
  };

  const selectedSecret = compatibleSecrets.find((s) => s.secret_id === selectedSecretId);
  const hasChanges =
    secretSource === 'new'
      ? newSecretName.trim().length > 0 || newSecretValue.trim().length > 0
      : selectedSecretId !== '' && selectedSecretId !== route?.secret_id;

  return (
    <Modal
      componentId="mlflow.routes.update_route_modal"
      visible={visible}
      onCancel={onCancel}
      okText={intl.formatMessage({ defaultMessage: 'Update Endpoint', description: 'Update endpoint button' })}
      cancelText={intl.formatMessage({ defaultMessage: 'Cancel', description: 'Cancel button' })}
      onOk={handleUpdate}
      okButtonProps={{ loading: isLoading, disabled: !hasChanges }}
      title={<FormattedMessage defaultMessage="Update Endpoint API Key" description="Update endpoint modal title" />}
      size="wide"
    >
      {route && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
          {/* Route info */}
          <div>
            <Typography.Title level={4} css={{ marginTop: 0, marginBottom: theme.spacing.sm }}>
              {route.name || route.endpoint_id}
            </Typography.Title>
            <Typography.Text color="secondary" size="sm">
              <FormattedMessage
                defaultMessage="Select a new API key for this endpoint. The endpoint will use the selected key for authentication."
                description="Update endpoint modal description"
              />
            </Typography.Text>
          </div>

          {/* Secret source toggle */}
          <div>
            <FormUI.Label>
              <FormattedMessage defaultMessage="API Key Source" description="API key source label" />
            </FormUI.Label>
            <Radio.Group
              componentId="mlflow.routes.update_route_modal.secret_source"
              name="secret-source"
              value={secretSource}
              onChange={(e) => {
                setSecretSource(e.target.value as SecretSource);
                setErrors({});
              }}
            >
              <Radio value="existing">
                <FormattedMessage defaultMessage="Use Existing Key" description="Use existing secret option" />
              </Radio>
              <Radio value="new">
                <FormattedMessage defaultMessage="Create New Key" description="Create new secret option" />
              </Radio>
            </Radio.Group>
          </div>

          {/* Current API Key Display */}
          <div
            css={{
              padding: theme.spacing.md,
              borderRadius: theme.borders.borderRadiusMd,
              backgroundColor: theme.colors.backgroundSecondary,
              border: `1px solid ${theme.colors.border}`,
            }}
          >
            <Typography.Text color="secondary" size="sm" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
              <FormattedMessage defaultMessage="Current API Key:" description="Current API key label" />
            </Typography.Text>
            <Typography.Text css={{ fontWeight: 500 }}>
              {secrets.find((s) => s.secret_id === route?.secret_id)?.secret_name || route?.secret_id}
            </Typography.Text>
          </div>

          {/* Existing secret selection */}
          {secretSource === 'existing' && (
            <div>
              <DialogCombobox
                componentId="mlflow.routes.update_route_modal.secret"
                label={intl.formatMessage({
                  defaultMessage: 'Select API Key',
                  description: 'Select API key label',
                })}
                value={
                  selectedSecretId
                    ? [compatibleSecrets.find((s) => s.secret_id === selectedSecretId)?.secret_name || '']
                    : []
                }
              >
                <DialogComboboxTrigger allowClear={false} placeholder="Select Existing API Key" />
                <DialogComboboxContent>
                  <DialogComboboxOptionList>
                    {compatibleSecrets.map((secret) => (
                      <DialogComboboxOptionListSelectItem
                        key={secret.secret_id}
                        checked={selectedSecretId === secret.secret_id}
                        value={secret.secret_name}
                        onChange={() => {
                          setSelectedSecretId(secret.secret_id);
                          setErrors((prev) => ({ ...prev, secretId: undefined }));
                        }}
                      >
                        <Typography.Text>{secret.secret_name}</Typography.Text>
                      </DialogComboboxOptionListSelectItem>
                    ))}
                  </DialogComboboxOptionList>
                </DialogComboboxContent>
              </DialogCombobox>
              {errors.secretId && <FormUI.Message type="error" message={errors.secretId} />}
            </div>
          )}

          {/* New secret creation */}
          {secretSource === 'new' && (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
              {/* Provider selector */}
              <div>
                <FormUI.Label htmlFor="update-route-provider">
                  <FormattedMessage defaultMessage="Provider" description="Provider select label" />
                  <span css={{ color: theme.colors.textValidationDanger }}> *</span>
                </FormUI.Label>
                <SimpleSelect
                  componentId="mlflow.routes.update_route_modal.provider"
                  id="update-route-provider"
                  label=""
                  value={newSecretProvider}
                  onChange={(e) => {
                    setNewSecretProvider(e.target.value);
                    setAuthConfig({}); // Reset auth config when provider changes
                    setErrors((prev) => ({ ...prev, provider: undefined }));
                  }}
                  validationState={errors.provider ? 'error' : undefined}
                  placeholder={intl.formatMessage({
                    defaultMessage: 'Select provider',
                    description: 'Provider select placeholder',
                  })}
                >
                  {PROVIDERS.map((provider) => (
                    <SimpleSelectOption key={provider.value} value={provider.value}>
                      {provider.label}
                    </SimpleSelectOption>
                  ))}
                </SimpleSelect>
                {errors.provider && <FormUI.Message type="error" message={errors.provider} />}
              </div>

              <div>
                <FormUI.Label htmlFor="update-route-new-secret-name">
                  <FormattedMessage defaultMessage="Secret Name" description="New secret name label" />
                  <span css={{ color: theme.colors.textValidationDanger }}> *</span>
                </FormUI.Label>
                <Input
                  componentId="mlflow.routes.update_route_modal.new_secret_name"
                  id="update-route-new-secret-name"
                  placeholder={intl.formatMessage({
                    defaultMessage: 'e.g., my-openai-key',
                    description: 'Secret name placeholder',
                  })}
                  value={newSecretName}
                  onChange={(e) => {
                    setNewSecretName(e.target.value);
                    setErrors((prev) => ({ ...prev, secretName: undefined }));
                  }}
                  validationState={errors.secretName ? 'error' : undefined}
                />
                {errors.secretName && <FormUI.Message type="error" message={errors.secretName} />}
              </div>

              <div>
                <FormUI.Label htmlFor="update-route-new-secret-value">
                  <FormattedMessage defaultMessage="API Key" description="New API key label" />
                  <span css={{ color: theme.colors.textValidationDanger }}> *</span>
                </FormUI.Label>
                <MaskedApiKeyInput
                  value={newSecretValue}
                  onChange={(value) => {
                    setNewSecretValue(value);
                    setErrors((prev) => ({ ...prev, secretValue: undefined }));
                  }}
                  placeholder={intl.formatMessage({
                    defaultMessage: 'Enter your API key',
                    description: 'API key placeholder',
                  })}
                  id="update-route-new-secret-value"
                  componentId="mlflow.routes.update_route_modal.new_secret_value"
                />
                {errors.secretValue && <FormUI.Message type="error" message={errors.secretValue} />}
              </div>

              {/* Auth configuration fields */}
              <AuthConfigFields
                fields={selectedProviderInfo?.authConfigFields || []}
                values={authConfig}
                onChange={(name, value) => {
                  setAuthConfig((prev) => ({ ...prev, [name]: value }));
                }}
                componentIdPrefix="mlflow.routes.update_route_modal.auth_config"
              />
            </div>
          )}

          {/* Impact warning */}
          {hasChanges && (endpointsUsingCurrentSecret.length > 0 || routeBindings.length > 0) && (
            <div
              css={{
                padding: theme.spacing.md,
                borderRadius: theme.borders.borderRadiusMd,
                backgroundColor: theme.colors.backgroundWarning,
                border: `1px solid ${theme.colors.borderWarning}`,
              }}
            >
              <Typography.Title level={4} css={{ marginTop: 0, marginBottom: theme.spacing.sm }}>
                <FormattedMessage defaultMessage="Impact Analysis" description="Impact analysis section title" />
              </Typography.Title>

              {/* Resources bound to this endpoint */}
              {routeBindings.length > 0 && (
                <div css={{ marginBottom: theme.spacing.md }}>
                  <Typography.Text css={{ fontWeight: 600, display: 'block', marginBottom: theme.spacing.xs }}>
                    <FormattedMessage
                      defaultMessage="Resources Using This Endpoint ({count})"
                      description="Resources bound to endpoint count"
                      values={{ count: routeBindings.length }}
                    />
                  </Typography.Text>
                  <Typography.Text
                    color="secondary"
                    size="sm"
                    css={{ display: 'block', marginBottom: theme.spacing.xs }}
                  >
                    <FormattedMessage
                      defaultMessage="These resources will use the new API key:"
                      description="Resources impact description"
                    />
                  </Typography.Text>
                  <ul css={{ margin: 0, paddingLeft: theme.spacing.lg }}>
                    {routeBindings.slice(0, 5).map((binding) => (
                      <li key={binding.binding_id}>
                        <Typography.Text size="sm">
                          {binding.resource_type}: {binding.resource_id}
                        </Typography.Text>
                      </li>
                    ))}
                    {routeBindings.length > 5 && (
                      <li>
                        <Typography.Text size="sm" color="secondary">
                          <FormattedMessage
                            defaultMessage="... and {count} more"
                            description="More resources indicator"
                            values={{ count: routeBindings.length - 5 }}
                          />
                        </Typography.Text>
                      </li>
                    )}
                  </ul>
                </div>
              )}

              {/* Other endpoints using the same key */}
              {endpointsUsingCurrentSecret.length > 0 && (
                <div>
                  <Typography.Text css={{ fontWeight: 600, display: 'block', marginBottom: theme.spacing.xs }}>
                    <FormattedMessage
                      defaultMessage="Other Endpoints Using Current Key ({count})"
                      description="Endpoints using same key count"
                      values={{ count: endpointsUsingCurrentSecret.length }}
                    />
                  </Typography.Text>
                  <Typography.Text
                    color="secondary"
                    size="sm"
                    css={{ display: 'block', marginBottom: theme.spacing.xs }}
                  >
                    <FormattedMessage
                      defaultMessage="These endpoints will continue using the current key:"
                      description="Endpoints impact description"
                    />
                  </Typography.Text>
                  <ul css={{ margin: 0, paddingLeft: theme.spacing.lg }}>
                    {endpointsUsingCurrentSecret.slice(0, 5).map((r) => (
                      <li key={r.endpoint_id}>
                        <Typography.Text size="sm">
                          {r.name || r.endpoint_id} ({r.model_name})
                        </Typography.Text>
                      </li>
                    ))}
                    {endpointsUsingCurrentSecret.length > 5 && (
                      <li>
                        <Typography.Text size="sm" color="secondary">
                          <FormattedMessage
                            defaultMessage="... and {count} more"
                            description="More endpoints indicator"
                            values={{ count: endpointsUsingCurrentSecret.length - 5 }}
                          />
                        </Typography.Text>
                      </li>
                    )}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </Modal>
  );
};
