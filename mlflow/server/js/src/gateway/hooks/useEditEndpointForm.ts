import { useForm } from 'react-hook-form';
import { useNavigate } from '../../common/utils/RoutingUtils';
import { useQuery, useMutation, useQueryClient } from '../../common/utils/reactQueryHooks';
import { useEffect, useCallback, useMemo } from 'react';
import { GatewayApi } from '../api';
import { useCreateSecretMutation } from './useCreateSecretMutation';
import { useEndpointsQuery } from './useEndpointsQuery';
import { useSecretsQuery } from './useSecretsQuery';
import { useModelsQuery } from './useModelsQuery';
import GatewayRoutes from '../routes';
import type { SecretMode } from '../components/secrets/SecretConfigSection';
import type { Endpoint } from '../types';

export interface EditEndpointFormData {
  name: string;
  provider: string;
  modelName: string;
  secretMode: SecretMode;
  existingSecretId: string;
  newSecret: {
    name: string;
    authMode: string;
    secretFields: Record<string, string>;
    configFields: Record<string, string>;
  };
}

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
export const getReadableErrorMessage = (error: Error | null): string | null => {
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

export interface UseEditEndpointFormResult {
  form: ReturnType<typeof useForm<EditEndpointFormData>>;
  // Loading states
  isLoadingEndpoint: boolean;
  isSubmitting: boolean;
  // Errors
  loadError: Error | null;
  mutationError: Error | null;
  resetErrors: () => void;
  // Data
  endpoint: Endpoint | undefined;
  selectedModel: ReturnType<typeof useModelsQuery>['data'] extends (infer T)[] | undefined ? T | undefined : never;
  selectedSecretName: string | undefined;
  // Computed state
  isFormComplete: boolean;
  // Handlers
  handleSubmit: (values: EditEndpointFormData) => Promise<void>;
  handleCancel: () => void;
  handleNameBlur: () => void;
}

/**
 * Custom hook that contains all business logic for editing an endpoint.
 * Separates concerns from the renderer component for easier testing and reuse.
 */
export function useEditEndpointForm(endpointId: string): UseEditEndpointFormResult {
  const navigate = useNavigate();

  // Fetch endpoint data
  const { data: endpointData, isLoading: isLoadingEndpoint, error: loadError } = useEndpointQuery(endpointId);
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
        authMode: '',
        secretFields: {},
        configFields: {},
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
          authMode: '',
          secretFields: {},
          configFields: {},
        },
      });
    }
  }, [endpoint, primaryModelDef, form]);

  // Mutations
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

  const resetErrors = useCallback(() => {
    resetEndpointError();
    resetModelDefError();
    resetSecretError();
  }, [resetEndpointError, resetModelDefError, resetSecretError]);

  const isSubmitting = isUpdatingEndpoint || isUpdatingModelDef || isCreatingSecret;
  const mutationError = (updateEndpointError || updateModelDefError || createSecretError) as Error | null;

  const handleSubmit = useCallback(
    async (values: EditEndpointFormData) => {
      if (!endpoint || !primaryModelDef) return;

      try {
        // Update endpoint name if changed
        if (values.name !== endpoint.name) {
          await updateEndpoint({ endpointId: endpoint.endpoint_id, name: values.name || undefined });
        }

        // Determine the secret ID to use
        let secretId = values.existingSecretId;
        if (values.secretMode === 'new') {
          // Serialize secret fields as JSON for the secret value
          const secretValue = JSON.stringify(values.newSecret.secretFields);
          // Build auth_config with auth_mode included
          const authConfig: Record<string, any> = { ...values.newSecret.configFields };
          if (values.newSecret.authMode) {
            authConfig['auth_mode'] = values.newSecret.authMode;
          }
          const authConfigJson = Object.keys(authConfig).length > 0 ? JSON.stringify(authConfig) : undefined;

          const secretResponse = await createSecret({
            secret_name: values.newSecret.name,
            secret_value: secretValue,
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
    },
    [endpoint, primaryModelDef, updateEndpoint, createSecret, updateModelDefinition, navigate],
  );

  const handleCancel = useCallback(() => {
    navigate(GatewayRoutes.getEndpointDetailsRoute(endpointId));
  }, [navigate, endpointId]);

  // Watch form values for computing isFormComplete
  const provider = form.watch('provider');
  const modelName = form.watch('modelName');
  const secretMode = form.watch('secretMode');
  const existingSecretId = form.watch('existingSecretId');
  const newSecretName = form.watch('newSecret.name');
  const newSecretFields = form.watch('newSecret.secretFields');

  // Check if the form is complete enough to enable the Save button
  const isFormComplete = useMemo(() => {
    const hasSecretFieldValues = Object.values(newSecretFields || {}).some((v) => Boolean(v));
    const isSecretConfigured =
      secretMode === 'existing' ? Boolean(existingSecretId) : Boolean(newSecretName) && hasSecretFieldValues;
    return Boolean(provider) && Boolean(modelName) && isSecretConfigured;
  }, [secretMode, existingSecretId, newSecretName, newSecretFields, provider, modelName]);

  // Fetch existing endpoints to check for name conflicts on blur
  const { data: existingEndpoints } = useEndpointsQuery();

  // Get the selected model's full data for the summary
  const { data: models } = useModelsQuery({ provider: provider || undefined });
  const selectedModel = models?.find((m) => m.model === modelName);

  // Get secrets for the selected provider to find the selected secret name
  const { data: providerSecrets } = useSecretsQuery({ provider: provider || undefined });
  const selectedSecretName = providerSecrets?.find((s) => s.secret_id === existingSecretId)?.secret_name;

  const handleNameBlur = useCallback(() => {
    const name = form.getValues('name');
    // Exclude the current endpoint's name (user can keep the same name)
    const otherEndpoints = existingEndpoints?.filter((e) => e.endpoint_id !== endpointId);
    if (name && otherEndpoints?.some((e) => e.name === name)) {
      form.setError('name', {
        type: 'manual',
        message: 'An endpoint with this name already exists',
      });
    }
  }, [form, existingEndpoints, endpointId]);

  return {
    form,
    isLoadingEndpoint,
    isSubmitting,
    loadError: loadError as Error | null,
    mutationError,
    resetErrors,
    endpoint,
    selectedModel,
    selectedSecretName,
    isFormComplete,
    handleSubmit,
    handleCancel,
    handleNameBlur,
  };
}
