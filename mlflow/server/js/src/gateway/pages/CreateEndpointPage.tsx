import { useNavigate, Link } from '../../common/utils/RoutingUtils';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import {
  Alert,
  Breadcrumb,
  Button,
  FormUI,
  Input,
  Radio,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { useForm, Controller, FormProvider } from 'react-hook-form';
import { useCreateEndpointMutation } from '../hooks/useCreateEndpointMutation';
import { useCreateSecretMutation } from '../hooks/useCreateSecretMutation';
import { useCreateModelDefinitionMutation } from '../hooks/useCreateModelDefinitionMutation';
import { ProviderSelect } from '../components/create-endpoint/ProviderSelect';
import { ModelSelect } from '../components/create-endpoint/ModelSelect';
import { SecretConfigSection, type SecretMode } from '../components/secrets/SecretConfigSection';
import { ModelDefinitionSelector } from '../components/model-definitions/ModelDefinitionSelector';
import { useModelDefinitionsQuery } from '../hooks/useModelDefinitionsQuery';
import GatewayRoutes from '../routes';
import { formatProviderName } from '../utils/providerUtils';
import { LongFormSection, LongFormSummary } from '../../common/components/long-form';
import { useModelsQuery } from '../hooks/useModelsQuery';
import { useEndpointsQuery } from '../hooks/useEndpointsQuery';
import type { Model } from '../types';

type ModelDefinitionMode = 'new' | 'existing';

const LONG_FORM_TITLE_WIDTH = 200;

/**
 * Check if error message indicates a unique constraint violation across different DB backends
 */
const isUniqueConstraintError = (message: string): boolean => {
  const lowerMessage = message.toLowerCase();
  return (
    // SQLite
    lowerMessage.includes('unique constraint failed') ||
    // PostgreSQL
    lowerMessage.includes('duplicate key value violates unique constraint') ||
    // MySQL
    lowerMessage.includes('duplicate entry') ||
    // SQL Server
    lowerMessage.includes('violation of unique key constraint') ||
    // Generic patterns
    lowerMessage.includes('uniqueviolation') ||
    lowerMessage.includes('integrityerror')
  );
};

/**
 * Check if the error is related to a specific field (endpoint name or secret name)
 */
const isEndpointNameError = (message: string): boolean => {
  const lowerMessage = message.toLowerCase();
  return lowerMessage.includes('endpoints.name') || lowerMessage.includes('endpoint_name');
};

const isSecretNameError = (message: string): boolean => {
  const lowerMessage = message.toLowerCase();
  return lowerMessage.includes('secrets.secret_name') || lowerMessage.includes('secret_name');
};

/**
 * Parse backend error messages and return user-friendly versions
 */
const getReadableErrorMessage = (error: Error | null): string | null => {
  if (!error?.message) return null;

  const message = error.message;

  if (isUniqueConstraintError(message)) {
    if (isEndpointNameError(message)) {
      return 'An endpoint with this name already exists. Please choose a different name.';
    }
    if (isSecretNameError(message)) {
      return 'A secret with this name already exists. Please choose a different name or use an existing secret.';
    }
    // Generic unique constraint fallback
    return 'A record with this value already exists. Please use a unique value.';
  }

  // Return original message if no pattern matched (but truncate if too long)
  if (message.length > 200) {
    return 'An error occurred while creating the endpoint. Please try again.';
  }

  return message;
};

interface CreateEndpointFormData {
  name: string;
  modelDefinitionMode: ModelDefinitionMode;
  existingModelDefinitionId: string;
  // Fields for new model definition
  modelDefinitionName: string;
  provider: string;
  modelName: string;
  secretMode: SecretMode;
  existingSecretId: string;
  newSecret: {
    name: string;
    value: string;
    authConfig: Record<string, string>;
  };
}

const CreateEndpointPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();

  const form = useForm<CreateEndpointFormData>({
    defaultValues: {
      name: '',
      modelDefinitionMode: 'new',
      existingModelDefinitionId: '',
      modelDefinitionName: '',
      provider: '',
      modelName: '',
      secretMode: 'new',
      existingSecretId: '',
      newSecret: {
        name: '',
        value: '',
        authConfig: {},
      },
    },
  });

  // Fetch existing model definitions
  const { data: existingModelDefinitions } = useModelDefinitionsQuery();

  const {
    mutateAsync: createEndpoint,
    error: createEndpointError,
    isLoading: isCreatingEndpoint,
    reset: resetEndpointError,
  } = useCreateEndpointMutation();

  const {
    mutateAsync: createSecret,
    error: createSecretError,
    isLoading: isCreatingSecret,
    reset: resetSecretError,
  } = useCreateSecretMutation();

  const {
    mutateAsync: createModelDefinition,
    error: createModelDefinitionError,
    isLoading: isCreatingModelDefinition,
    reset: resetModelDefinitionError,
  } = useCreateModelDefinitionMutation();

  const resetErrors = () => {
    resetEndpointError();
    resetSecretError();
    resetModelDefinitionError();
  };

  const isLoading = isCreatingEndpoint || isCreatingSecret || isCreatingModelDefinition;
  const error = (createEndpointError || createSecretError || createModelDefinitionError) as Error | null;

  const handleSubmit = async (values: CreateEndpointFormData) => {
    try {
      let modelDefinitionId: string;

      if (values.modelDefinitionMode === 'existing') {
        // Use existing model definition
        modelDefinitionId = values.existingModelDefinitionId;
      } else {
        // Create new model definition
        let secretId = values.existingSecretId;

        if (values.secretMode === 'new') {
          const authConfigJson =
            Object.keys(values.newSecret.authConfig).length > 0
              ? JSON.stringify(values.newSecret.authConfig)
              : undefined;

          const secretResponse = await createSecret({
            secret_name: values.newSecret.name,
            secret_value: values.newSecret.value,
            provider: values.provider,
            auth_config_json: authConfigJson,
          });

          secretId = secretResponse.secret.secret_id;
        }

        // Create a model definition
        const modelDefinitionResponse = await createModelDefinition({
          name: values.modelDefinitionName || `${values.name || 'endpoint'}-${values.modelName}`,
          secret_id: secretId,
          provider: values.provider,
          model_name: values.modelName,
        });

        modelDefinitionId = modelDefinitionResponse.model_definition.model_definition_id;
      }

      // Create the endpoint with the model definition ID
      const endpointResponse = await createEndpoint({
        name: values.name || undefined,
        model_definition_ids: [modelDefinitionId],
      });

      navigate(GatewayRoutes.getEndpointDetailsRoute(endpointResponse.endpoint.endpoint_id));
    } catch {
      // Error is captured by the mutation's error state and displayed via the Alert component
    }
  };

  const handleCancel = () => {
    navigate(GatewayRoutes.gatewayPageRoute);
  };

  const modelDefinitionMode = form.watch('modelDefinitionMode');
  const existingModelDefinitionId = form.watch('existingModelDefinitionId');
  const modelDefinitionName = form.watch('modelDefinitionName');
  const provider = form.watch('provider');
  const modelName = form.watch('modelName');
  const secretMode = form.watch('secretMode');
  const existingSecretId = form.watch('existingSecretId');
  const newSecretName = form.watch('newSecret.name');
  const newSecretValue = form.watch('newSecret.value');

  // Filter model definitions by selected provider
  const providerModelDefinitions = provider ? existingModelDefinitions?.filter((md) => md.provider === provider) : [];
  const hasProviderModelDefinitions = providerModelDefinitions && providerModelDefinitions.length > 0;

  // Find selected model definition for display
  const selectedModelDefinition = existingModelDefinitions?.find(
    (md) => md.model_definition_id === existingModelDefinitionId,
  );

  // Check if the form is complete enough to enable the Create button
  const isNewModelConfigured = () => {
    const isSecretConfigured =
      secretMode === 'existing' ? Boolean(existingSecretId) : Boolean(newSecretName) && Boolean(newSecretValue);
    return Boolean(provider) && Boolean(modelName) && isSecretConfigured;
  };

  const isFormComplete =
    modelDefinitionMode === 'existing'
      ? Boolean(provider) && Boolean(existingModelDefinitionId)
      : isNewModelConfigured();

  // Get the selected model's full data for the summary
  const { data: models } = useModelsQuery({ provider: provider || undefined });
  const selectedModel = models?.find((m) => m.model === modelName);

  // Fetch existing endpoints to check for name conflicts on blur
  const { data: existingEndpoints } = useEndpointsQuery();

  const handleNameBlur = () => {
    const name = form.getValues('name');
    if (name && existingEndpoints?.some((e) => e.name === name)) {
      form.setError('name', {
        type: 'manual',
        message: 'An endpoint with this name already exists',
      });
    }
  };

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
              message={getReadableErrorMessage(error)}
              type="error"
              css={{ marginBottom: theme.spacing.md }}
            />
          </div>
        )}

        <div
          css={{
            flex: 1,
            display: 'flex',
            gap: theme.spacing.lg,
            padding: `0 ${theme.spacing.md}px`,
            overflow: 'auto',
            // Stack vertically on narrow screens
            '@media (max-width: 1023px)': {
              flexDirection: 'column',
              gap: theme.spacing.md,
            },
          }}
        >
          {/* Main form column */}
          <div
            css={{
              flexGrow: 1,
              maxWidth: 900,
              minWidth: 0,
              '@media (max-width: 1023px)': {
                maxWidth: '100%',
              },
            }}
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
                      onBlur={(e) => {
                        field.onBlur();
                        handleNameBlur();
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
                    layout="horizontal"
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
                    value={existingModelDefinitionId}
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
          </div>

          {/* Summary sidebar */}
          <div
            css={{
              flexShrink: 0,
              width: 320,
              position: 'sticky',
              top: 0,
              alignSelf: 'flex-start',
              '@media (max-width: 1023px)': {
                width: '100%',
                position: 'static',
              },
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
                    <Tag componentId="mlflow.gateway.create-endpoint.summary.provider">
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
                        <Tag componentId="mlflow.gateway.create-endpoint.summary.model-def">
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
                    <ModelSummary model={selectedModel} modelName={modelName} />
                  ) : (
                    <Typography.Text color="secondary">
                      <FormattedMessage defaultMessage="Not configured" description="Summary not configured" />
                    </Typography.Text>
                  )}
                </div>

                {/* Authentication */}
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  <Typography.Text bold color="secondary">
                    <FormattedMessage defaultMessage="Authentication" description="Summary secret label" />
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
          <Button componentId="mlflow.gateway.create-endpoint.cancel" onClick={handleCancel}>
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
              onClick={form.handleSubmit(handleSubmit)}
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

/** Helper component to display model metadata in the summary */
const ModelSummary = ({ model, modelName }: { model: Model | undefined; modelName: string }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const formatTokens = (tokens: number | null) => {
    if (tokens === null || tokens === undefined) return null;
    if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
    if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(0)}K`;
    return tokens.toString();
  };

  const formatCost = (cost: number | null) => {
    if (cost === null || cost === undefined) return null;
    if (cost === 0) return 'Free';
    const perMillion = cost * 1_000_000;
    if (perMillion < 0.01) return `$${perMillion.toFixed(4)}/1M`;
    return `$${perMillion.toFixed(2)}/1M`;
  };

  const capabilities: string[] = [];
  if (model?.supports_function_calling) capabilities.push('Tools');
  if (model?.supports_reasoning) capabilities.push('Reasoning');

  const contextWindow = formatTokens(model?.max_input_tokens ?? null);
  const inputCost = formatCost(model?.input_cost_per_token ?? null);
  const outputCost = formatCost(model?.output_cost_per_token ?? null);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      {/* Model name */}
      <Tag componentId="mlflow.gateway.create-endpoint.summary.model">{modelName}</Tag>

      {/* Capabilities */}
      {capabilities.length > 0 && (
        <div css={{ display: 'flex', gap: theme.spacing.xs, flexWrap: 'wrap' }}>
          {capabilities.map((cap) => (
            <Tag key={cap} color="turquoise" componentId={`mlflow.gateway.create-endpoint.summary.capability.${cap}`}>
              {cap}
            </Tag>
          ))}
        </div>
      )}

      {/* Context & Cost info */}
      {model && (contextWindow || inputCost || outputCost) && (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
            marginTop: theme.spacing.xs,
            fontSize: theme.typography.fontSizeSm,
            color: theme.colors.textSecondary,
          }}
        >
          {contextWindow && (
            <span>
              {intl.formatMessage(
                { defaultMessage: 'Context: {tokens}', description: 'Context window size' },
                { tokens: contextWindow },
              )}
            </span>
          )}
          {(inputCost || outputCost) && (
            <span>
              {intl.formatMessage(
                { defaultMessage: 'Cost: {input} in / {output} out', description: 'Model cost per token' },
                { input: inputCost ?? '-', output: outputCost ?? '-' },
              )}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, CreateEndpointPage);
