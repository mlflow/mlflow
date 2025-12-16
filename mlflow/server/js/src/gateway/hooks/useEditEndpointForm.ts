import { useForm } from 'react-hook-form';
import { useNavigate } from '../../common/utils/RoutingUtils';
import { useQuery, useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useEffect, useCallback, useMemo, useState } from 'react';
import { GatewayApi } from '../api';
import { useCreateSecretMutation } from './useCreateSecretMutation';
import { useEndpointsQuery } from './useEndpointsQuery';
import { useModelDefinitionsQuery } from './useModelDefinitionsQuery';
import { getReadableErrorMessage, isSecretNameConflict } from '../utils/errorUtils';
import GatewayRoutes from '../routes';
import type { SecretMode } from '../components/secrets/SecretConfigSection';
import type { Endpoint, ModelDefinition } from '../types';

export { getReadableErrorMessage };

type ModelDefinitionMode = 'new' | 'existing';

export interface EditEndpointFormData {
  name: string;
  provider: string;
  modelDefinitionMode: ModelDefinitionMode;
  existingModelDefinitionId: string;
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
  selectedModelDefinition: ModelDefinition | undefined;
  hasProviderModelDefinitions: boolean;
  // Computed state
  isFormComplete: boolean;
  hasChanges: boolean;
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
  const queryClient = useQueryClient();

  // Fetch endpoint data
  const { data: endpointData, isLoading: isLoadingEndpoint, error: loadError } = useEndpointQuery(endpointId);
  const endpoint = endpointData?.endpoint;
  const primaryMapping = endpoint?.model_mappings?.[0];
  const primaryModelDef = primaryMapping?.model_definition;

  const form = useForm<EditEndpointFormData>({
    defaultValues: {
      name: '',
      provider: '',
      modelDefinitionMode: 'new',
      existingModelDefinitionId: '',
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
        modelDefinitionMode: 'new',
        existingModelDefinitionId: '',
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

  // Fetch model definitions for the selected provider
  const { data: allModelDefinitions } = useModelDefinitionsQuery();
  const provider = form.watch('provider');
  const existingModelDefinitionId = form.watch('existingModelDefinitionId');
  const modelDefinitionMode = form.watch('modelDefinitionMode');

  // Filter model definitions for current provider (exclude the one currently attached to this endpoint)
  const providerModelDefinitions = useMemo(() => {
    if (!allModelDefinitions || !provider) return [];
    return allModelDefinitions.filter(
      (md) => md.provider === provider && md.model_definition_id !== primaryModelDef?.model_definition_id,
    );
  }, [allModelDefinitions, provider, primaryModelDef?.model_definition_id]);

  const hasProviderModelDefinitions = providerModelDefinitions.length > 0;

  // Find the selected model definition
  const selectedModelDefinition = useMemo(() => {
    if (!existingModelDefinitionId || !allModelDefinitions) return undefined;
    return allModelDefinitions.find((md) => md.model_definition_id === existingModelDefinitionId);
  }, [existingModelDefinitionId, allModelDefinitions]);

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

  // Custom error state for handling special cases like secret name conflicts
  const [customError, setCustomError] = useState<Error | null>(null);

  const resetErrors = useCallback(() => {
    resetEndpointError();
    resetModelDefError();
    resetSecretError();
    setCustomError(null);
  }, [resetEndpointError, resetModelDefError, resetSecretError]);

  const isSubmitting = isUpdatingEndpoint || isUpdatingModelDef || isCreatingSecret;
  const mutationError = (customError ||
    updateEndpointError ||
    updateModelDefError ||
    createSecretError) as Error | null;

  /**
   * Resolves the secret ID to use - either existing or creates a new one.
   * Returns null if a name conflict was handled and the user needs to retry.
   */
  const resolveSecretId = useCallback(
    async (values: EditEndpointFormData): Promise<string | null> => {
      if (values.secretMode !== 'new') {
        return values.existingSecretId;
      }

      // Serialize secret fields as JSON for the secret value
      const secretValue = JSON.stringify(values.newSecret.secretFields);
      // Build auth_config with auth_mode included
      const authConfig: Record<string, any> = { ...values.newSecret.configFields };
      if (values.newSecret.authMode) {
        authConfig['auth_mode'] = values.newSecret.authMode;
      }
      const authConfigJson = Object.keys(authConfig).length > 0 ? JSON.stringify(authConfig) : undefined;

      try {
        const secretResponse = await createSecret({
          secret_name: values.newSecret.name,
          secret_value: secretValue,
          provider: values.provider,
          auth_config_json: authConfigJson,
        });
        return (secretResponse as { secret: { secret_id: string } }).secret.secret_id;
      } catch (secretError) {
        if (!isSecretNameConflict(secretError)) {
          throw secretError;
        }
        // Handle secret name conflict by finding and selecting the existing secret
        const secretsResponse = await GatewayApi.listSecrets(values.provider);
        const existingSecret = secretsResponse.secrets?.find((s) => s.secret_name === values.newSecret.name);
        if (!existingSecret) {
          throw secretError;
        }
        form.setValue('secretMode', 'existing');
        form.setValue('existingSecretId', existingSecret.secret_id);
        form.setValue('newSecret', { name: '', authMode: '', secretFields: {}, configFields: {} });
        queryClient.invalidateQueries(['gateway_secrets']);
        resetSecretError();
        setCustomError(
          new Error(
            `An API key named "${values.newSecret.name}" already exists. ` +
              `It has been selected for you. Please click Save again to continue.`,
          ),
        );
        return null;
      }
    },
    [createSecret, form, queryClient, resetSecretError],
  );

  const handleSubmit = useCallback(
    async (values: EditEndpointFormData) => {
      if (!endpoint || !primaryModelDef) return;

      try {
        // Update endpoint name if changed
        if (values.name !== endpoint.name) {
          await updateEndpoint({ endpointId: endpoint.endpoint_id, name: values.name || undefined });
        }

        // Handle existing model definition mode - swap model definitions
        if (values.modelDefinitionMode === 'existing' && values.existingModelDefinitionId) {
          // Only swap if selecting a different model definition
          if (values.existingModelDefinitionId !== primaryModelDef.model_definition_id) {
            // Detach current model definition and attach the selected one
            await GatewayApi.detachModelFromEndpoint({
              endpoint_id: endpoint.endpoint_id,
              model_definition_id: primaryModelDef.model_definition_id,
            });
            await GatewayApi.attachModelToEndpoint({
              endpoint_id: endpoint.endpoint_id,
              model_definition_id: values.existingModelDefinitionId,
            });
          }
        } else {
          // Handle new/modified model definition mode
          const secretId = await resolveSecretId(values);
          if (secretId === null) {
            return; // User was notified about conflict, exit early
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
        }

        navigate(GatewayRoutes.getEndpointDetailsRoute(endpoint.endpoint_id));
      } catch {
        // Error is captured by the mutation's error state and displayed via the Alert component
      }
    },
    [endpoint, primaryModelDef, updateEndpoint, updateModelDefinition, navigate, resolveSecretId],
  );

  const handleCancel = useCallback(() => {
    navigate(GatewayRoutes.getEndpointDetailsRoute(endpointId));
  }, [navigate, endpointId]);

  // Watch form values for computing isFormComplete
  const modelName = form.watch('modelName');
  const secretMode = form.watch('secretMode');
  const existingSecretId = form.watch('existingSecretId');
  const newSecretName = form.watch('newSecret.name');
  const newSecretFields = form.watch('newSecret.secretFields');

  // Check if the form is complete enough to enable the Save button
  const isFormComplete = useMemo(() => {
    // Existing model definition mode: just need to select one
    if (modelDefinitionMode === 'existing') {
      return Boolean(existingModelDefinitionId);
    }

    // New model definition mode: need provider, model, and secret
    const hasSecretFieldValues = Object.values(newSecretFields || {}).some((v) => Boolean(v));
    const isSecretConfigured =
      secretMode === 'existing' ? Boolean(existingSecretId) : Boolean(newSecretName) && hasSecretFieldValues;
    return Boolean(provider) && Boolean(modelName) && isSecretConfigured;
  }, [
    modelDefinitionMode,
    existingModelDefinitionId,
    secretMode,
    existingSecretId,
    newSecretName,
    newSecretFields,
    provider,
    modelName,
  ]);

  // Watch the name field for hasChanges computation
  const name = form.watch('name');

  // Check if anything has changed from the original values
  const hasChanges = useMemo(() => {
    if (!endpoint || !primaryModelDef) return false;

    const originalName = endpoint.name ?? '';

    // Check if name changed
    if (name !== originalName) return true;

    // Existing model definition mode: check if a different model definition is selected
    if (modelDefinitionMode === 'existing') {
      return Boolean(existingModelDefinitionId) && existingModelDefinitionId !== primaryModelDef.model_definition_id;
    }

    // New model definition mode: check provider, model, or secret changes
    const originalProvider = primaryModelDef.provider ?? '';
    const originalModelName = primaryModelDef.model_name ?? '';
    const originalSecretId = primaryModelDef.secret_id ?? '';

    // Check if creating a new secret (always a change)
    if (secretMode === 'new') {
      const hasNewSecretData = Boolean(newSecretName) || Object.values(newSecretFields || {}).some((v) => Boolean(v));
      if (hasNewSecretData) return true;
    }

    // Check if any field has changed
    return (
      provider !== originalProvider ||
      modelName !== originalModelName ||
      (secretMode === 'existing' && existingSecretId !== originalSecretId)
    );
  }, [
    endpoint,
    primaryModelDef,
    name,
    modelDefinitionMode,
    existingModelDefinitionId,
    provider,
    modelName,
    secretMode,
    existingSecretId,
    newSecretName,
    newSecretFields,
  ]);

  // Fetch existing endpoints to check for name conflicts on blur
  const { data: existingEndpoints } = useEndpointsQuery();

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
    selectedModelDefinition,
    hasProviderModelDefinitions,
    isFormComplete,
    hasChanges,
    handleSubmit,
    handleCancel,
    handleNameBlur,
  };
}
