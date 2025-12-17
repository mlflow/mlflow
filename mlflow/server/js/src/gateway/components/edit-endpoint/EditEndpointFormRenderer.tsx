import { Link } from '../../../common/utils/RoutingUtils';
import {
  Alert,
  Breadcrumb,
  Button,
  FormUI,
  Radio,
  Spinner,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { GatewayInput } from '../common';
import { FormattedMessage, useIntl } from 'react-intl';
import { Controller, FormProvider, UseFormReturn } from 'react-hook-form';
import { ProviderSelect } from '../create-endpoint/ProviderSelect';
import { ModelSelect } from '../create-endpoint/ModelSelect';
import { SecretConfigSection, type SecretMode } from '../secrets/SecretConfigSection';
import { ModelDefinitionSelector } from '../model-definitions/ModelDefinitionSelector';
import GatewayRoutes from '../../routes';
import { LongFormSection, LongFormSummary } from '../../../common/components/long-form';
import { formatProviderName } from '../../utils/providerUtils';
import type { EditEndpointFormData } from '../../hooks/useEditEndpointForm';
import type { ModelDefinition } from '../../types';

type ModelDefinitionMode = 'new' | 'existing';

const LONG_FORM_TITLE_WIDTH = 200;

export interface EditEndpointFormRendererProps {
  form: UseFormReturn<EditEndpointFormData>;
  isLoadingEndpoint: boolean;
  isSubmitting: boolean;
  loadError: Error | null;
  mutationError: Error | null;
  errorMessage: string | null;
  resetErrors: () => void;
  endpointId: string;
  endpointName: string | undefined;
  selectedModelDefinition?: ModelDefinition;
  hasProviderModelDefinitions: boolean;
  isFormComplete: boolean;
  hasChanges: boolean;
  onSubmit: (values: EditEndpointFormData) => Promise<void>;
  onCancel: () => void;
  onNameBlur: () => void;
}

/**
 * Pure presentational component for the edit endpoint form.
 * All business logic is handled by the container (useEditEndpointForm hook).
 */
export const EditEndpointFormRenderer = ({
  form,
  isLoadingEndpoint,
  isSubmitting,
  loadError,
  mutationError,
  errorMessage,
  resetErrors,
  endpointId,
  endpointName,
  selectedModelDefinition,
  hasProviderModelDefinitions,
  isFormComplete,
  hasChanges,
  onSubmit,
  onCancel,
  onNameBlur,
}: EditEndpointFormRendererProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const modelDefinitionMode = form.watch('modelDefinitionMode');
  const provider = form.watch('provider');
  const modelName = form.watch('modelName');
  const secretMode = form.watch('secretMode');

  // Loading state
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

  // Error state
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
      <FormProvider {...form}>
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
          {/* Form content */}
          <div css={{ flex: 1, maxWidth: 700 }}>
            {/* General Section */}
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

            {/* Model Section */}
            <LongFormSection
              titleWidth={LONG_FORM_TITLE_WIDTH}
              title={intl.formatMessage({
                defaultMessage: 'Model configuration',
                description: 'Section title for model configuration',
              })}
            >
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                {/* Provider is always selected first */}
                <Controller
                  control={form.control}
                  name="provider"
                  rules={{ required: 'Provider is required' }}
                  render={({ field, fieldState }) => (
                    <ProviderSelect
                      value={field.value}
                      onChange={(value) => {
                        field.onChange(value);
                        // Reset all dependent fields when provider changes
                        form.setValue('modelName', '');
                        form.setValue('existingSecretId', '');
                        form.setValue('existingModelDefinitionId', '');
                        form.setValue('modelDefinitionMode', 'new');
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

                {/* Mode selector */}
                <div>
                  <Radio.Group
                    componentId="mlflow.gateway.edit-endpoint.model-definition-mode"
                    name="modelDefinitionMode"
                    value={modelDefinitionMode}
                    onChange={(e) => {
                      form.setValue('modelDefinitionMode', e.target.value as ModelDefinitionMode);
                      if (e.target.value === 'existing') {
                        form.setValue('modelName', '');
                        form.setValue('existingSecretId', '');
                      } else {
                        form.setValue('existingModelDefinitionId', '');
                      }
                    }}
                    layout="horizontal"
                  >
                    <Radio value="new">
                      <FormattedMessage
                        defaultMessage="Edit current model"
                        description="Option to edit current model definition"
                      />
                    </Radio>
                    <Radio value="existing" disabled={!hasProviderModelDefinitions}>
                      <FormattedMessage
                        defaultMessage="Use existing model definition"
                        description="Option to use existing model definition"
                      />
                    </Radio>
                  </Radio.Group>
                  {provider && !hasProviderModelDefinitions && (
                    <Typography.Text
                      color="secondary"
                      css={{ display: 'block', marginTop: theme.spacing.xs, fontSize: theme.typography.fontSizeSm }}
                    >
                      <FormattedMessage
                        defaultMessage="No other model definitions for this provider."
                        description="Message when no other model definitions for provider"
                      />
                    </Typography.Text>
                  )}
                </div>

                {/* Model selection based on mode */}
                {modelDefinitionMode === 'existing' ? (
                  <ModelDefinitionSelector
                    value={form.watch('existingModelDefinitionId')}
                    onChange={(id) => form.setValue('existingModelDefinitionId', id)}
                    provider={provider}
                  />
                ) : (
                  <Controller
                    control={form.control}
                    name="modelName"
                    rules={{ required: modelDefinitionMode === 'new' ? 'Model is required' : false }}
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
                )}
              </div>
            </LongFormSection>

            {/* Connections Section - only show when editing current model */}
            {modelDefinitionMode === 'new' && (
              <LongFormSection
                titleWidth={LONG_FORM_TITLE_WIDTH}
                title={intl.formatMessage({
                  defaultMessage: 'Connections',
                  description: 'Section title for authentication',
                })}
                hideDivider
              >
                <SecretConfigSection
                  provider={provider}
                  mode={secretMode}
                  onModeChange={(mode) => form.setValue('secretMode', mode)}
                  selectedSecretId={form.watch('existingSecretId')}
                  onSecretSelect={(secretId) => form.setValue('existingSecretId', secretId)}
                  newSecretFieldPrefix="newSecret"
                  componentIdPrefix="mlflow.gateway.edit-endpoint"
                />
              </LongFormSection>
            )}
          </div>

          {/* Summary sidebar */}
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
                {/* Provider */}
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  <Typography.Text bold color="secondary">
                    <FormattedMessage defaultMessage="Provider" description="Summary provider label" />
                  </Typography.Text>
                  {provider ? (
                    <Tag componentId="mlflow.gateway.edit-endpoint.summary.provider">
                      {formatProviderName(provider)}
                    </Tag>
                  ) : (
                    <Typography.Text color="secondary">
                      <FormattedMessage defaultMessage="Not configured" description="Summary not configured" />
                    </Typography.Text>
                  )}
                </div>

                {/* Model - different display based on mode */}
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  <Typography.Text bold color="secondary">
                    <FormattedMessage defaultMessage="Model" description="Summary model label" />
                  </Typography.Text>
                  {modelDefinitionMode === 'existing' ? (
                    selectedModelDefinition ? (
                      <>
                        <Tag componentId="mlflow.gateway.edit-endpoint.summary.model-def">
                          {selectedModelDefinition.name}
                        </Tag>
                        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                          {selectedModelDefinition.model_name}
                        </Typography.Text>
                      </>
                    ) : (
                      <Typography.Text color="secondary">
                        <FormattedMessage
                          defaultMessage="Select a model definition"
                          description="Summary select model def"
                        />
                      </Typography.Text>
                    )
                  ) : modelName ? (
                    <Typography.Text>{modelName}</Typography.Text>
                  ) : (
                    <Typography.Text color="secondary">
                      <FormattedMessage defaultMessage="Not configured" description="Summary not configured" />
                    </Typography.Text>
                  )}
                </div>

                {/* Connections */}
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  <Typography.Text bold color="secondary">
                    <FormattedMessage defaultMessage="Connections" description="Summary connections label" />
                  </Typography.Text>
                  {modelDefinitionMode === 'existing' ? (
                    selectedModelDefinition ? (
                      <Typography.Text>{selectedModelDefinition.secret_name}</Typography.Text>
                    ) : (
                      <Typography.Text color="secondary">—</Typography.Text>
                    )
                  ) : (
                    <Typography.Text>
                      {secretMode === 'new' ? (
                        <FormattedMessage defaultMessage="New secret" description="Summary new secret" />
                      ) : (
                        <FormattedMessage defaultMessage="Existing secret" description="Summary existing secret" />
                      )}
                    </Typography.Text>
                  )}
                </div>
              </div>
            </LongFormSummary>
          </div>
        </div>

        {/* Footer buttons */}
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
      </FormProvider>
    </div>
  );
};
