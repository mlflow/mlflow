import { useMemo, useCallback } from 'react';
import { Link } from '../../../common/utils/RoutingUtils';
import {
  Alert,
  Breadcrumb,
  Button,
  FormUI,
  Spinner,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { GatewayInput } from '../common';
import { FormattedMessage, useIntl } from 'react-intl';
import { Controller, UseFormReturn } from 'react-hook-form';
import { ProviderSelect } from '../create-endpoint/ProviderSelect';
import { ModelSelect } from '../create-endpoint/ModelSelect';
import { ApiKeyConfigurator, useApiKeyConfiguration } from '../model-configuration';
import type { ApiKeyConfiguration } from '../model-configuration';
import GatewayRoutes from '../../routes';
import { LongFormSection, LongFormSummary } from '../../../common/components/long-form';
import { formatProviderName } from '../../utils/providerUtils';
import type { EditEndpointFormData } from '../../hooks/useEditEndpointForm';

const LONG_FORM_TITLE_WIDTH = 200;

export interface EditEndpointFormRendererProps {
  form: UseFormReturn<EditEndpointFormData>;
  isLoadingEndpoint: boolean;
  isSubmitting: boolean;
  loadError: Error | null;
  mutationError: Error | null;
  errorMessage: string | null;
  resetErrors: () => void;
  endpointName: string | undefined;
  isFormComplete: boolean;
  hasChanges: boolean;
  onSubmit: (values: EditEndpointFormData) => Promise<void>;
  onCancel: () => void;
  onNameBlur: () => void;
}

export const EditEndpointFormRenderer = ({
  form,
  isLoadingEndpoint,
  isSubmitting,
  loadError,
  mutationError,
  errorMessage,
  resetErrors,
  endpointName,
  isFormComplete,
  hasChanges,
  onSubmit,
  onCancel,
  onNameBlur,
}: EditEndpointFormRendererProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const provider = form.watch('provider');
  const modelName = form.watch('modelName');
  const secretMode = form.watch('secretMode');
  const existingSecretId = form.watch('existingSecretId');
  const newSecret = form.watch('newSecret');

  const { existingSecrets, isLoadingSecrets, authModes, defaultAuthMode, isLoadingProviderConfig } =
    useApiKeyConfiguration({ provider });

  const selectedSecretName = existingSecrets.find((s) => s.secret_id === existingSecretId)?.secret_name;

  const apiKeyConfig: ApiKeyConfiguration = useMemo(
    () => ({
      mode: secretMode,
      existingSecretId: existingSecretId,
      newSecret: newSecret,
    }),
    [secretMode, existingSecretId, newSecret],
  );

  const handleApiKeyChange = useCallback(
    (config: ApiKeyConfiguration) => {
      if (config.mode !== secretMode) {
        form.setValue('secretMode', config.mode);
      }
      if (config.existingSecretId !== existingSecretId) {
        form.setValue('existingSecretId', config.existingSecretId);
      }
      if (config.newSecret !== newSecret) {
        form.setValue('newSecret', config.newSecret);
      }
    },
    [form, secretMode, existingSecretId, newSecret],
  );

  if (isLoadingEndpoint) {
    return (
      <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, padding: theme.spacing.md }}>
          <Spinner size="small" />
          <FormattedMessage defaultMessage="Loading endpoint..." description="Loading message for endpoint" />
        </div>
      </div>
    );
  }

  if (loadError) {
    return (
      <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div css={{ padding: theme.spacing.md }}>
          <Alert
            componentId="mlflow.gateway.edit-endpoint.error"
            type="error"
            message={loadError.message ?? 'Endpoint not found'}
          />
        </div>
      </div>
    );
  }

  return (
    <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
      <div css={{ padding: theme.spacing.md }}>
        <Breadcrumb includeTrailingCaret>
          <Breadcrumb.Item>
            <Link to={GatewayRoutes.gatewayPageRoute}>
              <FormattedMessage defaultMessage="AI Gateway" description="Breadcrumb link to gateway page" />
            </Link>
          </Breadcrumb.Item>
          <Breadcrumb.Item>
            <Link to={GatewayRoutes.gatewayPageRoute}>
              <FormattedMessage defaultMessage="Endpoints" description="Breadcrumb link to endpoints list" />
            </Link>
          </Breadcrumb.Item>
        </Breadcrumb>
        <Typography.Title level={2} css={{ marginTop: theme.spacing.sm }}>
          <FormattedMessage defaultMessage="Edit endpoint" description="Page title for edit endpoint" />
        </Typography.Title>
        <div
          css={{
            marginTop: theme.spacing.md,
            borderBottom: `1px solid ${theme.colors.border}`,
          }}
        />
      </div>

      {mutationError && (
        <div css={{ padding: `0 ${theme.spacing.md}px` }}>
          <Alert
            componentId="mlflow.gateway.edit-endpoint.mutation-error"
            closable={false}
            message={errorMessage}
            type="error"
            css={{ marginBottom: theme.spacing.md }}
          />
        </div>
      )}

      <div
        css={{
          flex: 1,
          display: 'flex',
          gap: theme.spacing.md,
          padding: `0 ${theme.spacing.md}px`,
          overflow: 'auto',
        }}
      >
        <div css={{ flex: 1, maxWidth: 700 }}>
          <LongFormSection
            titleWidth={LONG_FORM_TITLE_WIDTH}
            title={intl.formatMessage({
              defaultMessage: 'General',
              description: 'Section title for general settings',
            })}
          >
            <Controller
              control={form.control}
              name="name"
              render={({ field, fieldState }) => (
                <div>
                  <FormUI.Label htmlFor="mlflow.gateway.edit-endpoint.name">
                    <FormattedMessage defaultMessage="Name" description="Label for endpoint name input" />
                  </FormUI.Label>
                  <GatewayInput
                    id="mlflow.gateway.edit-endpoint.name"
                    componentId="mlflow.gateway.edit-endpoint.name"
                    {...field}
                    onChange={(e) => {
                      field.onChange(e);
                      form.clearErrors('name');
                      resetErrors();
                    }}
                    onBlur={() => {
                      field.onBlur();
                      onNameBlur();
                    }}
                    placeholder={intl.formatMessage({
                      defaultMessage: 'my-endpoint',
                      description: 'Placeholder for endpoint name input',
                    })}
                    validationState={fieldState.error ? 'error' : undefined}
                  />
                  {fieldState.error && <FormUI.Message type="error" message={fieldState.error.message} />}
                </div>
              )}
            />
          </LongFormSection>

          <LongFormSection
            titleWidth={LONG_FORM_TITLE_WIDTH}
            title={intl.formatMessage({
              defaultMessage: 'Model configuration',
              description: 'Section title for model configuration',
            })}
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
              <Controller
                control={form.control}
                name="provider"
                rules={{ required: 'Provider is required' }}
                render={({ field, fieldState }) => (
                  <ProviderSelect
                    value={field.value}
                    onChange={(value) => {
                      field.onChange(value);
                      form.setValue('modelName', '');
                      form.setValue('existingSecretId', '');
                      form.setValue('secretMode', 'new');
                      form.setValue('newSecret', {
                        name: '',
                        authMode: '',
                        secretFields: {},
                        configFields: {},
                      });
                    }}
                    error={fieldState.error?.message}
                    componentIdPrefix="mlflow.gateway.edit-endpoint.provider"
                  />
                )}
              />

              <Controller
                control={form.control}
                name="modelName"
                rules={{ required: 'Model is required' }}
                render={({ field, fieldState }) => (
                  <ModelSelect
                    provider={provider}
                    value={field.value}
                    onChange={field.onChange}
                    error={fieldState.error?.message}
                    componentIdPrefix="mlflow.gateway.edit-endpoint.model"
                  />
                )}
              />
            </div>
          </LongFormSection>

          <LongFormSection
            titleWidth={LONG_FORM_TITLE_WIDTH}
            title={intl.formatMessage({
              defaultMessage: 'Connections',
              description: 'Section title for authentication',
            })}
            hideDivider
          >
            <ApiKeyConfigurator
              value={apiKeyConfig}
              onChange={handleApiKeyChange}
              provider={provider}
              existingSecrets={existingSecrets}
              isLoadingSecrets={isLoadingSecrets}
              authModes={authModes}
              defaultAuthMode={defaultAuthMode}
              isLoadingProviderConfig={isLoadingProviderConfig}
              componentIdPrefix="mlflow.gateway.edit-endpoint.api-key"
            />
          </LongFormSection>
        </div>

        <div
          css={{
            width: 280,
            flexShrink: 0,
            position: 'sticky',
            top: 0,
            alignSelf: 'flex-start',
          }}
        >
          <LongFormSummary
            title={intl.formatMessage({
              defaultMessage: 'Summary',
              description: 'Summary sidebar title',
            })}
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <Typography.Text bold color="secondary">
                  <FormattedMessage defaultMessage="Provider" description="Summary provider label" />
                </Typography.Text>
                {provider ? (
                  <Typography.Text>{formatProviderName(provider)}</Typography.Text>
                ) : (
                  <Typography.Text color="secondary">
                    <FormattedMessage defaultMessage="Not configured" description="Summary not configured" />
                  </Typography.Text>
                )}
              </div>

              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <Typography.Text bold color="secondary">
                  <FormattedMessage defaultMessage="Model" description="Summary model label" />
                </Typography.Text>
                {modelName ? (
                  <Typography.Text>{modelName}</Typography.Text>
                ) : (
                  <Typography.Text color="secondary">
                    <FormattedMessage defaultMessage="Not configured" description="Summary not configured" />
                  </Typography.Text>
                )}
              </div>

              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <Typography.Text bold color="secondary">
                  <FormattedMessage defaultMessage="Connections" description="Summary connections label" />
                </Typography.Text>
                {secretMode === 'existing' && selectedSecretName ? (
                  <Typography.Text>{selectedSecretName}</Typography.Text>
                ) : secretMode === 'new' && newSecret?.name ? (
                  <Typography.Text>
                    {newSecret.name}{' '}
                    <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                      <FormattedMessage defaultMessage="(new)" description="Summary new secret indicator" />
                    </Typography.Text>
                  </Typography.Text>
                ) : (
                  <Typography.Text color="secondary">
                    <FormattedMessage defaultMessage="Not configured" description="Summary not configured" />
                  </Typography.Text>
                )}
              </div>
            </div>
          </LongFormSummary>
        </div>
      </div>

      <div
        css={{
          display: 'flex',
          justifyContent: 'flex-end',
          gap: theme.spacing.sm,
          padding: theme.spacing.md,
          borderTop: `1px solid ${theme.colors.border}`,
          flexShrink: 0,
        }}
      >
        <Button componentId="mlflow.gateway.edit-endpoint.cancel" onClick={onCancel}>
          <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
        </Button>
        <Tooltip
          componentId="mlflow.gateway.edit-endpoint.save-tooltip"
          content={
            !isFormComplete
              ? intl.formatMessage({
                  defaultMessage: 'Please select a provider, model, and configure authentication',
                  description: 'Tooltip shown when save button is disabled due to incomplete form',
                })
              : !hasChanges
              ? intl.formatMessage({
                  defaultMessage: 'No changes to save',
                  description: 'Tooltip shown when save button is disabled due to no changes',
                })
              : undefined
          }
        >
          <Button
            componentId="mlflow.gateway.edit-endpoint.save"
            type="primary"
            onClick={form.handleSubmit(onSubmit)}
            loading={isSubmitting}
            disabled={!isFormComplete || !hasChanges}
          >
            <FormattedMessage defaultMessage="Save changes" description="Save changes button" />
          </Button>
        </Tooltip>
      </div>
    </div>
  );
};
