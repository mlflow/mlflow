import {
  Button,
  Modal,
  Typography,
  useDesignSystemTheme,
  FormUI,
  TrashIcon,
  RefreshIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useState } from 'react';
import { MaskedApiKeyInput } from './MaskedApiKeyInput';
import type { Secret, SecretBinding, Endpoint } from '../types';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';

export interface SecretManagementModalProps {
  secret: Secret | null;
  visible: boolean;
  onClose: () => void;
  onRotate?: (secretId: string, newValue: string) => Promise<void>;
  onDelete?: (secretId: string) => Promise<void>;
  routes?: Endpoint[];
  bindings?: SecretBinding[];
}

export const SecretManagementModal = ({
  secret,
  visible,
  onClose,
  onRotate,
  onDelete,
  routes = [],
  bindings = [],
}: SecretManagementModalProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const [mode, setMode] = useState<'view' | 'rotate' | 'delete'>('view');
  const [newSecretValue, setNewSecretValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | undefined>(undefined);

  const handleReset = () => {
    setMode('view');
    setNewSecretValue('');
    setError(undefined);
    onClose();
  };

  const handleRotate = async () => {
    if (!secret) return;

    if (!newSecretValue.trim()) {
      setError(
        intl.formatMessage({
          defaultMessage: 'New API key is required',
          description: 'Secret management > rotate key validation error',
        }),
      );
      return;
    }

    setIsLoading(true);
    setError(undefined);
    try {
      await onRotate?.(secret.secret_id, newSecretValue);
      handleReset();
    } catch (err: any) {
      setError(err.message || 'Failed to rotate key');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!secret) return;

    setIsLoading(true);
    setError(undefined);
    try {
      await onDelete?.(secret.secret_id);
      handleReset();
    } catch (err: any) {
      setError(err.message || 'Failed to delete key');
    } finally {
      setIsLoading(false);
    }
  };

  // Find endpoints using this secret (check each model's secret_id)
  const routesUsingSecret = routes.filter((endpoint) =>
    endpoint.models.some((model) => model.secret_id === secret?.secret_id),
  );

  // Get unique resource bindings (not route bindings)
  const resourceBindings = bindings.filter((b) => b.secret_id === secret?.secret_id);

  const renderViewMode = () => (
    <>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        {/* Secret details */}
        <div>
          <Typography.Title level={4} css={{ marginTop: 0, marginBottom: theme.spacing.md }}>
            <FormattedMessage defaultMessage="Secret Details" description="Secret management > details section title" />
          </Typography.Title>
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            <div>
              <Typography.Text color="secondary" size="sm">
                <FormattedMessage defaultMessage="Secret Name" description="Secret management > name label" />
              </Typography.Text>
              <Typography.Paragraph css={{ marginTop: theme.spacing.xs, marginBottom: 0, fontWeight: 500 }}>
                {secret?.secret_name}
              </Typography.Paragraph>
            </div>
            <div>
              <Typography.Text color="secondary" size="sm">
                <FormattedMessage defaultMessage="Masked Value" description="Secret management > masked value label" />
              </Typography.Text>
              <Typography.Paragraph css={{ marginTop: theme.spacing.xs, marginBottom: 0, fontFamily: 'monospace' }}>
                {secret?.masked_value}
              </Typography.Paragraph>
            </div>
            {secret?.provider && (
              <div>
                <Typography.Text color="secondary" size="sm">
                  <FormattedMessage defaultMessage="Provider" description="Secret management > provider label" />
                </Typography.Text>
                <Typography.Paragraph css={{ marginTop: theme.spacing.xs, marginBottom: 0 }}>
                  {secret.provider}
                </Typography.Paragraph>
              </div>
            )}
            <div>
              <Typography.Text color="secondary" size="sm">
                <FormattedMessage defaultMessage="Created At" description="Secret management > created at label" />
              </Typography.Text>
              <Typography.Paragraph css={{ marginTop: theme.spacing.xs, marginBottom: 0 }}>
                {secret && Utils.formatTimestamp(secret.created_at)}
              </Typography.Paragraph>
            </div>
            <div>
              <Typography.Text color="secondary" size="sm">
                <FormattedMessage defaultMessage="Last Updated" description="Secret management > last updated label" />
              </Typography.Text>
              <Typography.Paragraph css={{ marginTop: theme.spacing.xs, marginBottom: 0 }}>
                {secret && Utils.formatTimestamp(secret.last_updated_at)}
              </Typography.Paragraph>
            </div>
          </div>
        </div>

        {/* Routes using this secret */}
        {routesUsingSecret.length > 0 && (
          <div>
            <Typography.Title level={4} css={{ marginTop: 0, marginBottom: theme.spacing.md }}>
              <FormattedMessage
                defaultMessage="Routes Using This Key ({count})"
                description="Secret management > routes section title"
                values={{ count: routesUsingSecret.length }}
              />
            </Typography.Title>
            <ul css={{ margin: 0, paddingLeft: theme.spacing.lg }}>
              {routesUsingSecret.map((endpoint) => (
                <li key={endpoint.endpoint_id}>
                  <Typography.Text>
                    {endpoint.name || endpoint.endpoint_id} (
                    {endpoint.models
                      .filter((model) => model.secret_id === secret?.secret_id)
                      .map((model) => model.model_name)
                      .join(', ')}
                    )
                  </Typography.Text>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Resource bindings */}
        {resourceBindings.length > 0 && (
          <div>
            <Typography.Title level={4} css={{ marginTop: 0, marginBottom: theme.spacing.md }}>
              <FormattedMessage
                defaultMessage="Resource Bindings ({count})"
                description="Secret management > bindings section title"
                values={{ count: resourceBindings.length }}
              />
            </Typography.Title>
            <ul css={{ margin: 0, paddingLeft: theme.spacing.lg }}>
              {resourceBindings.slice(0, 10).map((binding) => (
                <li key={binding.binding_id}>
                  <Typography.Text size="sm">
                    {binding.resource_type}: {binding.resource_id}
                  </Typography.Text>
                </li>
              ))}
              {resourceBindings.length > 10 && (
                <li>
                  <Typography.Text size="sm" color="secondary">
                    <FormattedMessage
                      defaultMessage="... and {count} more"
                      description="Secret management > more bindings indicator"
                      values={{ count: resourceBindings.length - 10 }}
                    />
                  </Typography.Text>
                </li>
              )}
            </ul>
          </div>
        )}

        {/* Action buttons */}
        <div css={{ display: 'flex', gap: theme.spacing.sm }}>
          <Button
            componentId="mlflow.secrets.management_modal.rotate_button"
            icon={<RefreshIcon />}
            onClick={() => setMode('rotate')}
          >
            <FormattedMessage defaultMessage="Rotate Key" description="Secret management > rotate button" />
          </Button>
          <Button
            componentId="mlflow.secrets.management_modal.delete_button"
            icon={<TrashIcon />}
            danger
            onClick={() => setMode('delete')}
          >
            <FormattedMessage defaultMessage="Delete Key" description="Secret management > delete button" />
          </Button>
        </div>
      </div>
    </>
  );

  const renderRotateMode = () => (
    <>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        <div>
          <Typography.Title level={4} css={{ marginTop: 0, marginBottom: theme.spacing.sm }}>
            <FormattedMessage defaultMessage="Rotate Key" description="Secret management > rotate key title" />
          </Typography.Title>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Enter a new API key value. All routes and resources using this key will automatically use the new value."
              description="Secret management > rotate key description"
            />
          </Typography.Text>
        </div>

        <div>
          <FormUI.Label htmlFor="new-secret-value">
            <FormattedMessage defaultMessage="New API Key" description="Secret management > new key label" />
            <span css={{ color: theme.colors.textValidationDanger }}> *</span>
          </FormUI.Label>
          <MaskedApiKeyInput
            value={newSecretValue}
            onChange={setNewSecretValue}
            placeholder={intl.formatMessage({
              defaultMessage: 'Enter new API key',
              description: 'Secret management > new key placeholder',
            })}
            id="new-secret-value"
            componentId="mlflow.secrets.management_modal.new_value"
          />
          {error && <FormUI.Message type="error" message={error} />}
        </div>

        {/* Impact warning */}
        {(routesUsingSecret.length > 0 || resourceBindings.length > 0) && (
          <div
            css={{
              padding: theme.spacing.md,
              borderRadius: theme.borders.borderRadiusMd,
              backgroundColor: theme.colors.backgroundWarning,
              border: `1px solid ${theme.colors.borderWarning}`,
            }}
          >
            <Typography.Title level={4} css={{ marginTop: 0, marginBottom: theme.spacing.sm }}>
              <FormattedMessage defaultMessage="Impact Analysis" description="Secret management > impact title" />
            </Typography.Title>
            {routesUsingSecret.length > 0 && (
              <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                <FormattedMessage
                  defaultMessage="{count} {count, plural, one {route} other {routes}} will use the new key"
                  description="Secret management > routes impact"
                  values={{ count: routesUsingSecret.length }}
                />
              </Typography.Text>
            )}
            {resourceBindings.length > 0 && (
              <Typography.Text css={{ display: 'block' }}>
                <FormattedMessage
                  defaultMessage="{count} {count, plural, one {resource binding} other {resource bindings}} will use the new key"
                  description="Secret management > bindings impact"
                  values={{ count: resourceBindings.length }}
                />
              </Typography.Text>
            )}
          </div>
        )}
      </div>
    </>
  );

  const renderDeleteMode = () => (
    <>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        <div>
          <Typography.Title level={4} css={{ marginTop: 0, marginBottom: theme.spacing.sm }}>
            <FormattedMessage defaultMessage="Delete Key" description="Secret management > delete key title" />
          </Typography.Title>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Are you sure you want to delete this key? This action cannot be undone."
              description="Secret management > delete key description"
            />
          </Typography.Text>
        </div>

        {/* Danger warning */}
        <div
          css={{
            padding: theme.spacing.md,
            borderRadius: theme.borders.borderRadiusMd,
            backgroundColor: theme.colors.backgroundDanger,
            border: `1px solid ${theme.colors.borderDanger}`,
          }}
        >
          <Typography.Title
            level={4}
            css={{ marginTop: 0, marginBottom: theme.spacing.sm, color: theme.colors.textValidationDanger }}
          >
            <FormattedMessage
              defaultMessage="Warning: Cascade Delete"
              description="Secret management > cascade delete title"
            />
          </Typography.Title>
          <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Deleting this key will permanently delete:"
              description="Secret management > cascade delete intro"
            />
          </Typography.Text>
          <ul css={{ margin: 0, paddingLeft: theme.spacing.lg }}>
            <li>
              <Typography.Text>
                <FormattedMessage
                  defaultMessage="The secret and its value"
                  description="Secret management > delete secret item"
                />
              </Typography.Text>
            </li>
            {routesUsingSecret.length > 0 && (
              <li>
                <Typography.Text>
                  <FormattedMessage
                    defaultMessage="{count} {count, plural, one {route} other {routes}}"
                    description="Secret management > delete routes item"
                    values={{ count: routesUsingSecret.length }}
                  />
                </Typography.Text>
              </li>
            )}
            {resourceBindings.length > 0 && (
              <li>
                <Typography.Text>
                  <FormattedMessage
                    defaultMessage="{count} {count, plural, one {resource binding} other {resource bindings}}"
                    description="Secret management > delete bindings item"
                    values={{ count: resourceBindings.length }}
                  />
                </Typography.Text>
              </li>
            )}
          </ul>
        </div>

        {error && <FormUI.Message type="error" message={error} />}
      </div>
    </>
  );

  const getModalTitle = () => {
    switch (mode) {
      case 'rotate':
        return intl.formatMessage({
          defaultMessage: 'Rotate Key',
          description: 'Secret management > rotate modal title',
        });
      case 'delete':
        return intl.formatMessage({
          defaultMessage: 'Delete Key',
          description: 'Secret management > delete modal title',
        });
      default:
        return intl.formatMessage({
          defaultMessage: 'Manage Key',
          description: 'Secret management > view modal title',
        });
    }
  };

  const getOkText = () => {
    switch (mode) {
      case 'rotate':
        return intl.formatMessage({
          defaultMessage: 'Rotate Key',
          description: 'Secret management > rotate ok button',
        });
      case 'delete':
        return intl.formatMessage({
          defaultMessage: 'Delete Key',
          description: 'Secret management > delete ok button',
        });
      default:
        return intl.formatMessage({
          defaultMessage: 'Close',
          description: 'Secret management > close button',
        });
    }
  };

  const getCancelText = () => {
    if (mode === 'view') {
      return intl.formatMessage({
        defaultMessage: 'Close',
        description: 'Secret management > view cancel button',
      });
    }
    return intl.formatMessage({
      defaultMessage: 'Back',
      description: 'Secret management > edit cancel button',
    });
  };

  const handleOk = () => {
    switch (mode) {
      case 'rotate':
        return handleRotate();
      case 'delete':
        return handleDelete();
      default:
        return handleReset();
    }
  };

  return (
    <Modal
      componentId="mlflow.secrets.management_modal"
      visible={visible}
      onCancel={mode === 'view' ? handleReset : () => setMode('view')}
      okText={getOkText()}
      cancelText={getCancelText()}
      onOk={handleOk}
      okButtonProps={{
        loading: isLoading,
        danger: mode === 'delete',
        disabled: mode === 'rotate' && !newSecretValue.trim(),
      }}
      title={getModalTitle()}
      size="wide"
    >
      {secret && (
        <>
          {mode === 'view' && renderViewMode()}
          {mode === 'rotate' && renderRotateMode()}
          {mode === 'delete' && renderDeleteMode()}
        </>
      )}
    </Modal>
  );
};
