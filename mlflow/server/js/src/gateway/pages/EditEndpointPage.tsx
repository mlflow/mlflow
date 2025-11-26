import { useParams, Link, useNavigate } from '../../common/utils/RoutingUtils';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import {
  Alert,
  Breadcrumb,
  Button,
  Spinner,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { useForm, Controller, FormProvider } from 'react-hook-form';
import { useQuery, useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useEffect } from 'react';
import { GatewayApi } from '../api';
import GatewayRoutes from '../routes';
import { ProviderSelect } from '../components/create-endpoint/ProviderSelect';
import { ModelSelect } from '../components/create-endpoint/ModelSelect';
import { SecretConfigSection, type SecretMode } from '../components/secrets/SecretConfigSection';
import { LongFormSection } from '../../common/components/long-form';
import { useCreateSecretMutation } from '../hooks/useCreateSecretMutation';
import { useEndpointsQuery } from '../hooks/useEndpointsQuery';
import { FormUI, Input } from '@databricks/design-system';

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
    return 'An error occurred while updating the endpoint. Please try again.';
  }

  return message;
};

interface EditEndpointFormData {
  name: string;
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

const useEndpointQuery = (endpointId: string) => {
  return useQuery(['gateway_endpoint', endpointId], {
    queryFn: () => GatewayApi.getEndpoint(endpointId),
    retry: false,
    enabled: Boolean(endpointId),
  });
};

const useUpdateEndpointMutation = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: { endpointId: string; name?: string }) =>
      GatewayApi.updateEndpoint({ endpoint_id: data.endpointId, name: data.name }),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_endpoints']);
      queryClient.invalidateQueries(['gateway_endpoint']);
    },
  });
};

const useUpdateModelDefinitionMutation = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: { modelDefinitionId: string; secretId?: string; provider?: string; modelName?: string }) =>
      GatewayApi.updateModelDefinition({
        model_definition_id: data.modelDefinitionId,
        secret_id: data.secretId,
        provider: data.provider,
        model_name: data.modelName,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_endpoints']);
      queryClient.invalidateQueries(['gateway_endpoint']);
      queryClient.invalidateQueries(['gateway_model_definitions']);
    },
  });
};

const EditEndpointPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();
  const { endpointId } = useParams<{ endpointId: string }>();

  const { data: endpointData, isLoading: isLoadingEndpoint, error: loadError } = useEndpointQuery(endpointId ?? '');
  const endpoint = endpointData?.endpoint;
  const primaryMapping = endpoint?.model_mappings?.[0];
  const primaryModelDef = primaryMapping?.model_definition;

  const form = useForm<EditEndpointFormData>({
    defaultValues: {
      name: '',
      provider: '',
      modelName: '',
      secretMode: 'existing',
      existingSecretId: '',
      newSecret: {
        name: '',
        value: '',
        authConfig: {},
      },
    },
  });

  // Reset form when endpoint data loads
  useEffect(() => {
    if (endpoint && primaryModelDef) {
      form.reset({
        name: endpoint.name ?? '',
        provider: primaryModelDef.provider ?? '',
        modelName: primaryModelDef.model_name ?? '',
        secretMode: 'existing',
        existingSecretId: primaryModelDef.secret_id ?? '',
        newSecret: {
          name: '',
          value: '',
          authConfig: {},
        },
      });
    }
  }, [endpoint, primaryModelDef, form]);

  const {
    mutateAsync: updateEndpoint,
    error: updateEndpointError,
    isLoading: isUpdatingEndpoint,
    reset: resetEndpointError,
  } = useUpdateEndpointMutation();
  const {
    mutateAsync: updateModelDefinition,
    error: updateModelDefError,
    isLoading: isUpdatingModelDef,
    reset: resetModelDefError,
  } = useUpdateModelDefinitionMutation();
  const {
    mutateAsync: createSecret,
    error: createSecretError,
    isLoading: isCreatingSecret,
    reset: resetSecretError,
  } = useCreateSecretMutation();

  const resetErrors = () => {
    resetEndpointError();
    resetModelDefError();
    resetSecretError();
  };

  const isLoading = isUpdatingEndpoint || isUpdatingModelDef || isCreatingSecret;
  const mutationError = (updateEndpointError || updateModelDefError || createSecretError) as Error | null;

  const handleSubmit = async (values: EditEndpointFormData) => {
    if (!endpoint || !primaryModelDef) return;

    try {
      // Update endpoint name if changed
      if (values.name !== endpoint.name) {
        await updateEndpoint({ endpointId: endpoint.endpoint_id, name: values.name || undefined });
      }

      // Determine the secret ID to use
      let secretId = values.existingSecretId;
      if (values.secretMode === 'new') {
        const authConfigJson =
          Object.keys(values.newSecret.authConfig).length > 0 ? JSON.stringify(values.newSecret.authConfig) : undefined;

        const secretResponse = await createSecret({
          secret_name: values.newSecret.name,
          secret_value: values.newSecret.value,
          provider: values.provider,
          auth_config_json: authConfigJson,
        });

        secretId = (secretResponse as { secret: { secret_id: string } }).secret.secret_id;
      }

      // Update model definition if provider, model name, or secret changed
      const modelDefChanged =
        values.provider !== primaryModelDef.provider ||
        values.modelName !== primaryModelDef.model_name ||
        secretId !== primaryModelDef.secret_id;

      if (modelDefChanged) {
        await updateModelDefinition({
          modelDefinitionId: primaryModelDef.model_definition_id,
          secretId: secretId,
          provider: values.provider,
          modelName: values.modelName,
        });
      }

      navigate(GatewayRoutes.getEndpointDetailsRoute(endpoint.endpoint_id));
    } catch {
      // Error is captured by the mutation's error state and displayed via the Alert component
    }
  };

  const handleCancel = () => {
    navigate(GatewayRoutes.getEndpointDetailsRoute(endpointId ?? ''));
  };

  const provider = form.watch('provider');
  const modelName = form.watch('modelName');
  const secretMode = form.watch('secretMode');
  const existingSecretId = form.watch('existingSecretId');
  const newSecretName = form.watch('newSecret.name');
  const newSecretValue = form.watch('newSecret.value');

  // Check if the form is complete enough to enable the Save button
  const isSecretConfigured =
    secretMode === 'existing' ? Boolean(existingSecretId) : Boolean(newSecretName) && Boolean(newSecretValue);
  const isFormComplete = Boolean(provider) && Boolean(modelName) && isSecretConfigured;

  // Fetch existing endpoints to check for name conflicts on blur
  const { data: existingEndpoints } = useEndpointsQuery();

  const handleNameBlur = () => {
    const name = form.getValues('name');
    // Exclude the current endpoint's name (user can keep the same name)
    const otherEndpoints = existingEndpoints?.filter((e) => e.endpoint_id !== endpointId);
    if (name && otherEndpoints?.some((e) => e.name === name)) {
      form.setError('name', {
        type: 'manual',
        message: 'An endpoint with this name already exists',
      });
    }
  };

  if (isLoadingEndpoint) {
    return (
      <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, padding: theme.spacing.md }}>
          <Spinner size="small" />
          <FormattedMessage defaultMessage="Loading endpoint..." description="Loading message for endpoint" />
        </div>
      </ScrollablePageWrapper>
    );
  }

  if (loadError || !endpoint) {
    return (
      <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div css={{ padding: theme.spacing.md }}>
          <Alert
            componentId="mlflow.gateway.edit-endpoint.error"
            type="error"
            message={(loadError as Error | null)?.message ?? 'Endpoint not found'}
          />
        </div>
      </ScrollablePageWrapper>
    );
  }

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
            <Breadcrumb.Item>
              <Link to={GatewayRoutes.getEndpointDetailsRoute(endpointId ?? '')}>
                {endpoint.name ?? endpoint.endpoint_id}
              </Link>
            </Breadcrumb.Item>
          </Breadcrumb>
          <Typography.Title level={2} css={{ marginTop: theme.spacing.sm }}>
            <FormattedMessage defaultMessage="Edit endpoint" description="Page title for edit endpoint" />
          </Typography.Title>
        </div>

        {mutationError && (
          <div css={{ padding: `0 ${theme.spacing.md}px` }}>
            <Alert
              componentId="mlflow.gateway.edit-endpoint.mutation-error"
              closable={false}
              message={getReadableErrorMessage(mutationError)}
              type="error"
              css={{ marginBottom: theme.spacing.md }}
            />
          </div>
        )}

        <div
          css={{
            flex: 1,
            padding: `0 ${theme.spacing.md}px`,
            overflow: 'auto',
            maxWidth: 900,
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
              render={({ field, fieldState }) => (
                <div>
                  <FormUI.Label htmlFor="mlflow.gateway.edit-endpoint.name">
                    <FormattedMessage defaultMessage="Name" description="Label for endpoint name input" />
                  </FormUI.Label>
                  <Input
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

          {/* Model Section */}
          <LongFormSection
            titleWidth={LONG_FORM_TITLE_WIDTH}
            title={intl.formatMessage({
              defaultMessage: 'Model',
              description: 'Section title for model selection',
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

          {/* Authentication Section */}
          <LongFormSection
            titleWidth={LONG_FORM_TITLE_WIDTH}
            title={intl.formatMessage({
              defaultMessage: 'Authentication',
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
          <Button componentId="mlflow.gateway.edit-endpoint.cancel" onClick={handleCancel}>
            <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
          </Button>
          <Tooltip
            componentId="mlflow.gateway.edit-endpoint.save-tooltip"
            content={
              !isFormComplete
                ? intl.formatMessage({
                    defaultMessage: 'Please select a provider, model, and configure authentication',
                    description: 'Tooltip shown when save button is disabled',
                  })
                : undefined
            }
          >
            <Button
              componentId="mlflow.gateway.edit-endpoint.save"
              type="primary"
              onClick={form.handleSubmit(handleSubmit)}
              loading={isLoading}
              disabled={!isFormComplete}
            >
              <FormattedMessage defaultMessage="Save" description="Save button" />
            </Button>
          </Tooltip>
        </div>
      </FormProvider>
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, EditEndpointPage);
