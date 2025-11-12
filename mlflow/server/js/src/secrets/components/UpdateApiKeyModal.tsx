import {
  FormUI,
  Input,
  Modal,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useState, useCallback, useEffect } from 'react';
import type { Secret } from '../types';
import { useUpdateSecretMutation } from '../hooks/useUpdateSecretMutation';
import { BindingsTable } from './BindingsTable';
import { Descriptions } from '@mlflow/mlflow/src/common/components/Descriptions';
import { useListBindings } from '../hooks/useListBindings';

const DEFAULT_PROVIDERS = [
  {
    provider_id: 'anthropic',
    display_name: 'Anthropic',
  },
  {
    provider_id: 'openai',
    display_name: 'OpenAI',
  },
  {
    provider_id: 'vertex_ai',
    display_name: 'Google Vertex AI',
  },
  {
    provider_id: 'bedrock',
    display_name: 'AWS Bedrock',
  },
  {
    provider_id: 'databricks',
    display_name: 'Databricks',
  },
];

export interface UpdateApiKeyModalProps {
  secret: Secret | null;
  visible: boolean;
  onCancel: () => void;
  onSuccess?: (secretName: string) => void;
}

export const UpdateApiKeyModal = ({ secret, visible, onCancel, onSuccess }: UpdateApiKeyModalProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [apiKeyValue, setApiKeyValue] = useState('');
  const [error, setError] = useState<string>();

  const providerDisplayName = DEFAULT_PROVIDERS.find((p) => p.provider_id === secret?.provider)?.display_name || secret?.provider || '';

  // Fetch bindings to get the environment variable name
  const { bindings } = useListBindings({ secretId: secret?.secret_id || '' });

  const { updateSecret, isLoading } = useUpdateSecretMutation({
    onSuccess: () => {
      const secretName = secret?.secret_name || '';
      onSuccess?.(secretName);
      setApiKeyValue('');
      setError(undefined);
      onCancel();
    },
    onError: (err: Error) => {
      setError(
        err.message ||
          intl.formatMessage({
            defaultMessage: 'Failed to update API key. Please try again.',
            description: 'Update API key modal > update failed error message',
          }),
      );
    },
  });

  const handleUpdate = useCallback(() => {
    if (!secret) return;

    if (!apiKeyValue) {
      setError(
        intl.formatMessage({
          defaultMessage: 'API key value is required',
          description: 'Update API key modal > API key value required validation',
        }),
      );
      return;
    }

    updateSecret({
      secret_id: secret.secret_id,
      secret_value: apiKeyValue,
      provider: secret.provider,
      model: secret.model,
    });
  }, [secret, apiKeyValue, updateSecret, intl]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && apiKeyValue && !isLoading) {
      e.preventDefault();
      handleUpdate();
    }
  }, [apiKeyValue, isLoading, handleUpdate]);

  const handleCancel = useCallback(() => {
    setApiKeyValue('');
    setError(undefined);
    onCancel();
  }, [onCancel]);

  // Reset state when modal opens
  useEffect(() => {
    if (visible) {
      setApiKeyValue('');
      setError(undefined);
    }
  }, [visible]);

  if (!secret) return null;

  return (
    <Modal
      componentId="mlflow.secrets.update_api_key_modal"
      visible={visible}
      onCancel={handleCancel}
      okText={intl.formatMessage({
        defaultMessage: 'Update API Key',
        description: 'Update API key modal > update button text',
      })}
      cancelText={intl.formatMessage({
        defaultMessage: 'Cancel',
        description: 'Update API key modal > cancel button text',
      })}
      onOk={handleUpdate}
      okButtonProps={{ loading: isLoading, disabled: !apiKeyValue }}
      title={
        <FormattedMessage
          defaultMessage="Update API Key"
          description="Update API key modal > modal title"
        />
      }
      size="wide"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        {/* Secret Name Header */}
        <div
          css={{
            padding: theme.spacing.md,
            borderRadius: theme.borders.borderRadiusMd,
            backgroundColor: theme.colors.backgroundSecondary,
            border: `1px solid ${theme.colors.border}`,
          }}
        >
          <Typography.Title level={3} css={{ margin: 0 }}>
            {secret.secret_name}
          </Typography.Title>
        </div>

        {/* Model Details */}
        <div>
          <Descriptions columns={1}>
            <Descriptions.Item
              label={
                <FormattedMessage
                  defaultMessage="Provider"
                  description="Update API key modal > provider label"
                />
              }
            >
              <Typography.Text>{providerDisplayName}</Typography.Text>
            </Descriptions.Item>
            <Descriptions.Item
              label={
                <FormattedMessage
                  defaultMessage="Model"
                  description="Update API key modal > model label"
                />
              }
            >
              <Typography.Text>{secret.model || '-'}</Typography.Text>
            </Descriptions.Item>
            {bindings && bindings.length > 0 && (
              <Descriptions.Item
                label={
                  <FormattedMessage
                    defaultMessage="Environment Variable"
                    description="Update API key modal > environment variable label"
                  />
                }
              >
                <Typography.Text css={{ fontFamily: 'monospace' }}>{bindings[0].field_name}</Typography.Text>
              </Descriptions.Item>
            )}
          </Descriptions>
        </div>

        {/* Bindings Warning */}
        <div>
          <BindingsTable secretId={secret.secret_id} variant="warning" isSharedSecret={secret.is_shared} />
        </div>

        {/* API Key Input */}
        <div>
          <FormUI.Label htmlFor="update-api-key-input">
            <FormattedMessage
              defaultMessage="New API Key"
              description="Update API key modal > new API key label"
            />
          </FormUI.Label>
          <Input
            componentId="mlflow.secrets.update_api_key_modal.value"
            id="update-api-key-input"
            type="password"
            autoComplete="off"
            data-form-type="other"
            data-lpignore="true"
            data-1p-ignore="true"
            data-bwignore="true"
            placeholder={intl.formatMessage({
              defaultMessage: 'Enter new API key',
              description: 'Update API key modal > API key placeholder',
            })}
            value={apiKeyValue}
            onChange={(e) => {
              setApiKeyValue(e.target.value);
              setError(undefined);
            }}
            onKeyDown={handleKeyDown}
          />
          {error && <FormUI.Message type="error" message={error} />}
        </div>
      </div>
    </Modal>
  );
};
