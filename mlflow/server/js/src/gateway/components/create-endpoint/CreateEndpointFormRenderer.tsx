import { Link } from '../../../common/utils/RoutingUtils';
import { ScrollablePageWrapper } from '../../../common/components/ScrollablePageWrapper';
import {
  Alert,
  Breadcrumb,
  Button,
  FormUI,
  Input,
  Radio,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Controller, FormProvider, UseFormReturn } from 'react-hook-form';
import { ProviderSelect } from './ProviderSelect';
import { ModelSelect } from './ModelSelect';
import { SecretConfigSection, type SecretMode } from '../secrets/SecretConfigSection';
import { ModelDefinitionSelector } from '../model-definitions/ModelDefinitionSelector';
import { EndpointSummary } from '../endpoints/EndpointSummary';
import GatewayRoutes from '../../routes';
import { LongFormLayout, LongFormSection } from '../../../common/components/long-form';
import type { Model, ModelDefinition } from '../../types';
import type { CreateEndpointFormData } from '../../hooks/useCreateEndpointForm';

type ModelDefinitionMode = 'new' | 'existing';

const LONG_FORM_TITLE_WIDTH = 200;

export interface CreateEndpointFormRendererProps {
  form: UseFormReturn<CreateEndpointFormData>;
  isLoading: boolean;
  error: Error | null;
  errorMessage: string | null;
  resetErrors: () => void;
  selectedModelDefinition?: ModelDefinition;
  selectedModel?: Model;
  selectedSecretName?: string;
  hasProviderModelDefinitions: boolean;
  isFormComplete: boolean;
  onSubmit: (values: CreateEndpointFormData) => Promise<void>;
  onCancel: () => void;
  onNameBlur: () => void;
}

/**
 * Pure presentational component for the create endpoint form.
 * All business logic is handled by the container (useCreateEndpointForm hook).
 */
export const CreateEndpointFormRenderer = ({
  form,
  isLoading,
  error,
  errorMessage,
  resetErrors,
  selectedModelDefinition,
  selectedModel,
  selectedSecretName,
  hasProviderModelDefinitions,
  isFormComplete,
  onSubmit,
  onCancel,
  onNameBlur,
}: CreateEndpointFormRendererProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const modelDefinitionMode = form.watch('modelDefinitionMode');
  const provider = form.watch('provider');
  const modelName = form.watch('modelName');
  const secretMode = form.watch('secretMode');

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <FormProvider {...form}>
        <div css={{ padding: theme.spacing.md }}>
          <Breadcrumb includeTrailingCaret>
            <Breadcrumb.Item>
              <Link to={GatewayRoutes.gatewayPageRoute}>
                <FormattedMessage defaultMessage="Gateway" description="Breadcrumb link to gateway page" />
              </Link>
            </Breadcrumb.Item>
          </Breadcrumb>
          <Typography.Title level={2} css={{ marginTop: theme.spacing.sm }}>
            <FormattedMessage defaultMessage="Create endpoint" description="Page title for create endpoint" />
          </Typography.Title>
        </div>

        {error && (
          <div css={{ padding: `0 ${theme.spacing.md}px` }}>
            <Alert
              componentId="mlflow.gateway.create-endpoint.error"
              closable={false}
              message={errorMessage}
              type="error"
              css={{ marginBottom: theme.spacing.md }}
            />
          </div>
        )}

        <LongFormLayout
          sidebar={
            <EndpointSummary
              provider={provider}
              modelName={modelDefinitionMode === 'existing' ? selectedModelDefinition?.model_name : modelName}
              modelMetadata={selectedModel}
              selectedSecretName={selectedSecretName}
              showConnection={modelDefinitionMode === 'new'}
              connectionMode={secretMode === 'new' ? 'new' : 'existing'}
              componentIdPrefix="mlflow.gateway.create-endpoint.summary"
            />
          }
        >
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
              rules={{ required: 'Name is required' }}
              render={({ field, fieldState }) => (
                <div>
                  <FormUI.Label htmlFor="mlflow.gateway.create-endpoint.name">
                    <FormattedMessage defaultMessage="Name" description="Label for endpoint name input" />
                  </FormUI.Label>
                  <Input
                    id="mlflow.gateway.create-endpoint.name"
                    componentId="mlflow.gateway.create-endpoint.name"
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

          {/* Model Configuration Section */}
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
                      // Reset dependent fields when provider changes
                      form.setValue('modelName', '');
                      form.setValue('existingSecretId', '');
                      form.setValue('existingModelDefinitionId', '');
                      // Reset to 'new' mode when provider changes
                      form.setValue('modelDefinitionMode', 'new');
                    }}
                    error={fieldState.error?.message}
                  />
                )}
              />

              {/* Mode selector - always shown, "existing" disabled if no model definitions for provider */}
              <div>
                <Radio.Group
                  componentId="mlflow.gateway.create-endpoint.model-definition-mode"
                  name="modelDefinitionMode"
                  value={modelDefinitionMode}
                  onChange={(e) => {
                    form.setValue('modelDefinitionMode', e.target.value as ModelDefinitionMode);
                    // Reset the other mode's values when switching
                    if (e.target.value === 'existing') {
                      form.setValue('modelName', '');
                      form.setValue('existingSecretId', '');
                    } else {
                      form.setValue('existingModelDefinitionId', '');
                    }
                  }}
                  layout="vertical"
                >
                  <Radio value="new">
                    <FormattedMessage
                      defaultMessage="Configure new model"
                      description="Option to configure new model"
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
                      defaultMessage="No existing model definitions for this provider."
                      description="Message when no existing model definitions for provider"
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
                <>
                  <Controller
                    control={form.control}
                    name="modelDefinitionName"
                    render={({ field }) => (
                      <div>
                        <FormUI.Label htmlFor="mlflow.gateway.create-endpoint.model-definition-name">
                          <FormattedMessage
                            defaultMessage="Model definition name"
                            description="Label for model definition name input"
                          />
                          <Typography.Text color="secondary" css={{ marginLeft: theme.spacing.xs }}>
                            <FormattedMessage defaultMessage="(optional)" description="Optional field indicator" />
                          </Typography.Text>
                        </FormUI.Label>
                        <Input
                          id="mlflow.gateway.create-endpoint.model-definition-name"
                          componentId="mlflow.gateway.create-endpoint.model-definition-name"
                          {...field}
                          placeholder={intl.formatMessage({
                            defaultMessage: 'Auto-generated if empty',
                            description: 'Placeholder for model definition name input',
                          })}
                        />
                      </div>
                    )}
                  />
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
                      />
                    )}
                  />
                </>
              )}
            </div>
          </LongFormSection>

          {/* Authentication Section - only show when creating new model definition */}
          {modelDefinitionMode === 'new' && (
            <LongFormSection
              titleWidth={LONG_FORM_TITLE_WIDTH}
              title={intl.formatMessage({
                defaultMessage: 'Authentication',
                description: 'Section title for authentication',
              })}
            >
              <SecretConfigSection
                provider={provider}
                mode={secretMode}
                onModeChange={(mode) => form.setValue('secretMode', mode)}
                selectedSecretId={form.watch('existingSecretId')}
                onSecretSelect={(secretId) => form.setValue('existingSecretId', secretId)}
                newSecretFieldPrefix="newSecret"
              />
            </LongFormSection>
          )}
        </LongFormLayout>

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
          <Button componentId="mlflow.gateway.create-endpoint.cancel" onClick={onCancel}>
            <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
          </Button>
          <Tooltip
            componentId="mlflow.gateway.create-endpoint.submit-tooltip"
            content={
              !isFormComplete
                ? intl.formatMessage({
                    defaultMessage: 'Please complete all required fields',
                    description: 'Tooltip shown when create button is disabled',
                  })
                : undefined
            }
          >
            <Button
              componentId="mlflow.gateway.create-endpoint.submit"
              type="primary"
              onClick={form.handleSubmit(onSubmit)}
              loading={isLoading}
              disabled={!isFormComplete}
            >
              <FormattedMessage defaultMessage="Create" description="Create button" />
            </Button>
          </Tooltip>
        </div>
      </FormProvider>
    </ScrollablePageWrapper>
  );
};
