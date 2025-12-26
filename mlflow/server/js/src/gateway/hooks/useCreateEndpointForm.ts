import { useForm } from 'react-hook-form';
import { useState, useCallback } from 'react';
import { useNavigate } from '../../common/utils/RoutingUtils';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useCreateEndpointMutation } from './useCreateEndpointMutation';
import { useCreateSecret } from './useCreateSecret';
import { useCreateModelDefinitionMutation } from './useCreateModelDefinitionMutation';
import { useModelsQuery } from './useModelsQuery';
import { useEndpointsQuery } from './useEndpointsQuery';
import GatewayRoutes from '../routes';
import type { ProviderModel } from '../types';
import type { SecretMode } from '../components/secrets/SecretConfigSection';

export interface CreateEndpointFormData {
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

export interface UseCreateEndpointFormResult {
  form: ReturnType<typeof useForm<CreateEndpointFormData>>;
  // Mutations and state
  isLoading: boolean;
  error: Error | null;
  resetErrors: () => void;
  // Data
  existingEndpoints: ReturnType<typeof useEndpointsQuery>['data'];
  selectedModel: ProviderModel | undefined;
  // Computed state
  isFormComplete: boolean;
  // Handlers
  handleSubmit: (values: CreateEndpointFormData) => Promise<void>;
  handleCancel: () => void;
  handleNameBlur: () => void;
}

/**
 * Custom hook that contains all business logic for creating an endpoint.
 * Separates concerns from the renderer component for easier testing and reuse.
 */
export function useCreateEndpointForm(): UseCreateEndpointFormResult {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const form = useForm<CreateEndpointFormData>({
    defaultValues: {
      name: '',
      provider: '',
      modelName: '',
      secretMode: 'new',
      existingSecretId: '',
      newSecret: {
        name: '',
        authMode: '',
        secretFields: {},
        configFields: {},
      },
    },
  });

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
  } = useCreateSecret();

  const {
    mutateAsync: createModelDefinition,
    error: createModelDefinitionError,
    isLoading: isCreatingModelDefinition,
    reset: resetModelDefinitionError,
  } = useCreateModelDefinitionMutation();

  // Custom error state for handling special cases like secret name conflicts
  const [customError, setCustomError] = useState<Error | null>(null);

  const resetErrors = useCallback(() => {
    resetEndpointError();
    resetSecretError();
    resetModelDefinitionError();
    setCustomError(null);
  }, [resetEndpointError, resetSecretError, resetModelDefinitionError]);

  const isLoading = isCreatingEndpoint || isCreatingSecret || isCreatingModelDefinition;
  const error = (customError || createEndpointError || createSecretError || createModelDefinitionError) as Error | null;

  const handleSubmit = async (values: CreateEndpointFormData) => {
    try {
      let secretId = values.existingSecretId;

      if (values.secretMode === 'new') {
        // Build auth_config with auth_mode included
        const authConfig: Record<string, string> = { ...values.newSecret.configFields };
        if (values.newSecret.authMode) {
          authConfig['auth_mode'] = values.newSecret.authMode;
        }

        const secretResponse = await createSecret({
          secret_name: values.newSecret.name,
          secret_value: values.newSecret.secretFields,
          provider: values.provider,
          auth_config: Object.keys(authConfig).length > 0 ? authConfig : undefined,
        });

        secretId = secretResponse.secret.secret_id;
      }

      // Create a model definition (auto-generate name from endpoint name and model)
      const modelDefinitionResponse = await createModelDefinition({
        name: `${values.name || 'endpoint'}-${values.modelName}`,
        secret_id: secretId,
        provider: values.provider,
        model_name: values.modelName,
      });

      const modelDefinitionId = modelDefinitionResponse.model_definition.model_definition_id;

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

  // Watch form values
  const provider = form.watch('provider');
  const modelName = form.watch('modelName');
  const secretMode = form.watch('secretMode');
  const existingSecretId = form.watch('existingSecretId');
  const newSecretName = form.watch('newSecret.name');
  const newSecretFields = form.watch('newSecret.secretFields');

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

  // Check if the form is complete enough to enable the Create button
  const hasSecretFieldValues = Object.values(newSecretFields || {}).some((v) => !!v);
  const isSecretConfigured = secretMode === 'existing' ? !!existingSecretId : !!newSecretName && hasSecretFieldValues;
  const isFormComplete = !!provider && !!modelName && isSecretConfigured;

  return {
    form,
    isLoading,
    error,
    resetErrors,
    existingEndpoints,
    selectedModel,
    isFormComplete,
    handleSubmit,
    handleCancel,
    handleNameBlur,
  };
}
