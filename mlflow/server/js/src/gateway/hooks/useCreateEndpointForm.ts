import { useForm } from 'react-hook-form';
import { useCallback, useEffect, useRef } from 'react';
import { useCreateEndpointMutation } from './useCreateEndpointMutation';
import { useCreateSecret } from './useCreateSecret';
import { useCreateModelDefinitionMutation } from './useCreateModelDefinitionMutation';
import { useModelsQuery } from './useModelsQuery';
import { useEndpointsQuery } from './useEndpointsQuery';
import type { ProviderModel, Endpoint } from '../types';
import type { SecretMode } from '../components/model-configuration/types';

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
  experimentId: string;
}

export interface UseCreateEndpointFormOptions {
  onSuccess?: (endpoint: Endpoint) => void;
  onCancel?: () => void;
}

export interface UseCreateEndpointFormResult {
  form: ReturnType<typeof useForm<CreateEndpointFormData>>;
  isLoading: boolean;
  error: Error | null;
  resetErrors: () => void;
  existingEndpoints: ReturnType<typeof useEndpointsQuery>['data'];
  selectedModel: ProviderModel | undefined;
  isFormComplete: boolean;
  handleSubmit: (values: CreateEndpointFormData) => Promise<void>;
  handleCancel: () => void;
  handleNameBlur: () => void;
}

export function useCreateEndpointForm({
  onSuccess,
  onCancel,
}: UseCreateEndpointFormOptions = {}): UseCreateEndpointFormResult {
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
      experimentId: '',
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

  const resetErrors = useCallback(() => {
    resetEndpointError();
    resetSecretError();
    resetModelDefinitionError();
  }, [resetEndpointError, resetSecretError, resetModelDefinitionError]);

  const isLoading = isCreatingEndpoint || isCreatingSecret || isCreatingModelDefinition;
  const error = (createEndpointError || createSecretError || createModelDefinitionError) as Error | null;

  const handleSubmit = async (values: CreateEndpointFormData) => {
    try {
      let secretId = values.existingSecretId;

      if (values.secretMode === 'new') {
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

      const modelDefinitionResponse = await createModelDefinition({
        name: `${values.name || 'endpoint'}-${values.modelName}`,
        secret_id: secretId,
        provider: values.provider,
        model_name: values.modelName,
      });

      const modelDefinitionId = modelDefinitionResponse.model_definition.model_definition_id;

      const endpointResponse = await createEndpoint({
        name: values.name || undefined,
        model_configs: [
          {
            model_definition_id: modelDefinitionId,
            linkage_type: 'PRIMARY',
            weight: 1.0,
          },
        ],
        experiment_id: values.experimentId || undefined,
      });

      onSuccess?.(endpointResponse.endpoint);
    } catch {
      // Errors are handled by mutation error state
    }
  };

  const handleCancel = () => {
    onCancel?.();
  };

  const provider = form.watch('provider');
  const modelName = form.watch('modelName');
  const secretMode = form.watch('secretMode');
  const existingSecretId = form.watch('existingSecretId');
  const newSecretName = form.watch('newSecret.name');
  const newSecretFields = form.watch('newSecret.secretFields');

  const prevProviderRef = useRef(provider);

  useEffect(() => {
    const prevProvider = prevProviderRef.current;
    prevProviderRef.current = provider;

    if (!prevProvider || !provider || prevProvider === provider) {
      return;
    }

    form.setValue('modelName', '');
    form.setValue('secretMode', 'new');
    form.setValue('existingSecretId', '');
    form.setValue('newSecret', {
      name: '',
      authMode: '',
      secretFields: {},
      configFields: {},
    });
    resetErrors();
  }, [provider, form, resetErrors]);

  const { data: models } = useModelsQuery({ provider: provider || undefined });
  const selectedModel = models?.find((m) => m.model === modelName);

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
