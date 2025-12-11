import { useForm } from 'react-hook-form';
import { useNavigate } from '../../common/utils/RoutingUtils';
import { useCreateEndpointMutation } from './useCreateEndpointMutation';
import { useCreateSecretMutation } from './useCreateSecretMutation';
import { useCreateModelDefinitionMutation } from './useCreateModelDefinitionMutation';
import { useModelDefinitionsQuery } from './useModelDefinitionsQuery';
import { useModelsQuery } from './useModelsQuery';
import { useEndpointsQuery } from './useEndpointsQuery';
import GatewayRoutes from '../routes';
import type { SecretMode } from '../components/secrets/SecretConfigSection';

type ModelDefinitionMode = 'new' | 'existing';

export interface CreateEndpointFormData {
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
    return 'An error occurred while creating the endpoint. Please try again.';
  }

  return message;
};

export interface UseCreateEndpointFormResult {
  form: ReturnType<typeof useForm<CreateEndpointFormData>>;
  // Mutations and state
  isLoading: boolean;
  error: Error | null;
  resetErrors: () => void;
  // Data
  existingModelDefinitions: ReturnType<typeof useModelDefinitionsQuery>['data'];
  existingEndpoints: ReturnType<typeof useEndpointsQuery>['data'];
  selectedModelDefinition: ReturnType<typeof useModelDefinitionsQuery>['data'] extends (infer T)[] | undefined
    ? T | undefined
    : never;
  selectedModel: ReturnType<typeof useModelsQuery>['data'] extends (infer T)[] | undefined ? T | undefined : never;
  providerModelDefinitions: ReturnType<typeof useModelDefinitionsQuery>['data'];
  hasProviderModelDefinitions: boolean;
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
        authMode: '',
        secretFields: {},
        configFields: {},
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
          // Serialize secret fields as JSON for the secret value
          const secretValue = JSON.stringify(values.newSecret.secretFields);
          // Serialize config fields as JSON for auth config
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

  // Watch form values
  const modelDefinitionMode = form.watch('modelDefinitionMode');
  const existingModelDefinitionId = form.watch('existingModelDefinitionId');
  const provider = form.watch('provider');
  const modelName = form.watch('modelName');
  const secretMode = form.watch('secretMode');
  const existingSecretId = form.watch('existingSecretId');
  const newSecretName = form.watch('newSecret.name');
  const newSecretFields = form.watch('newSecret.secretFields');

  // Filter model definitions by selected provider
  const providerModelDefinitions = provider ? existingModelDefinitions?.filter((md) => md.provider === provider) : [];
  const hasProviderModelDefinitions = providerModelDefinitions !== undefined && providerModelDefinitions.length > 0;

  // Find selected model definition for display
  const selectedModelDefinition = existingModelDefinitions?.find(
    (md) => md.model_definition_id === existingModelDefinitionId,
  );

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
  const isNewModelConfigured = () => {
    // Check if at least one secret field has a value
    const hasSecretFieldValues = Object.values(newSecretFields || {}).some((v) => Boolean(v));
    const isSecretConfigured =
      secretMode === 'existing' ? Boolean(existingSecretId) : Boolean(newSecretName) && hasSecretFieldValues;
    return Boolean(provider) && Boolean(modelName) && isSecretConfigured;
  };

  const isFormComplete =
    modelDefinitionMode === 'existing'
      ? Boolean(provider) && Boolean(existingModelDefinitionId)
      : isNewModelConfigured();

  return {
    form,
    isLoading,
    error,
    resetErrors,
    existingModelDefinitions,
    existingEndpoints,
    selectedModelDefinition,
    selectedModel,
    providerModelDefinitions,
    hasProviderModelDefinitions,
    isFormComplete,
    handleSubmit,
    handleCancel,
    handleNameBlur,
  };
}
