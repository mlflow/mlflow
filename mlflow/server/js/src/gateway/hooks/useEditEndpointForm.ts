import { useForm } from 'react-hook-form';
import { useNavigate } from '../../common/utils/RoutingUtils';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useEffect, useCallback, useMemo, useState } from 'react';
import { useEndpointsQuery } from './useEndpointsQuery';
import { useEndpointQuery } from './useEndpointQuery';
import { useUpdateEndpointMutation } from './useUpdateEndpointMutation';
import { useCreateModelDefinitionMutation } from './useCreateModelDefinitionMutation';
import { useUpdateModelDefinitionMutation } from './useUpdateModelDefinitionMutation';
import { useCreateSecret } from './useCreateSecret';
import { getReadableErrorMessage } from '../utils/errorUtils';
import { isValidEndpointName } from '../utils/gatewayUtils';
import GatewayRoutes from '../routes';
import type { Endpoint } from '../types';

export { getReadableErrorMessage };

export interface TrafficSplitModel {
  modelDefinitionId?: string;
  modelDefinitionName: string;
  provider: string;
  modelName: string;
  secretMode: 'new' | 'existing';
  existingSecretId: string;
  newSecret: {
    name: string;
    authMode: string;
    secretFields: Record<string, string>;
    configFields: Record<string, string>;
  };
  weight: number;
}

export interface FallbackModel {
  modelDefinitionId?: string;
  modelDefinitionName: string;
  provider: string;
  modelName: string;
  secretMode: 'new' | 'existing';
  existingSecretId: string;
  newSecret: {
    name: string;
    authMode: string;
    secretFields: Record<string, string>;
    configFields: Record<string, string>;
  };
  fallbackOrder: number;
}

export interface EditEndpointFormData {
  name: string;
  trafficSplitModels: TrafficSplitModel[];
  fallbackModels: FallbackModel[];
}

export interface UseEditEndpointFormResult {
  form: ReturnType<typeof useForm<EditEndpointFormData>>;
  isLoadingEndpoint: boolean;
  isSubmitting: boolean;
  loadError: Error | null;
  mutationError: Error | null;
  resetErrors: () => void;
  endpoint: Endpoint | undefined;
  isFormComplete: boolean;
  hasChanges: boolean;
  handleSubmit: (values: EditEndpointFormData) => Promise<void>;
  handleCancel: () => void;
  handleNameBlur: () => void;
}

export function useEditEndpointForm(endpointId: string): UseEditEndpointFormResult {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const { data: endpointData, isLoading: isLoadingEndpoint, error: loadError } = useEndpointQuery(endpointId);
  const endpoint = endpointData?.endpoint;

  const form = useForm<EditEndpointFormData>({
    defaultValues: {
      name: '',
      trafficSplitModels: [],
      fallbackModels: [],
    },
  });

  useEffect(() => {
    if (endpoint && endpoint.model_mappings) {
      const primaryMappings = endpoint.model_mappings.filter((m) => m.linkage_type === 'PRIMARY') ?? [];
      const fallbackMappings = endpoint.model_mappings.filter((m) => m.linkage_type === 'FALLBACK') ?? [];
      const totalWeight = primaryMappings.reduce((sum, m) => sum + (m.weight ?? 1.0), 0);

      form.reset({
        name: endpoint.name ?? '',
        trafficSplitModels: primaryMappings.map((m) => ({
          modelDefinitionId: m.model_definition?.model_definition_id,
          modelDefinitionName: m.model_definition?.name ?? '',
          provider: m.model_definition?.provider ?? '',
          modelName: m.model_definition?.model_name ?? '',
          secretMode: 'existing' as const,
          existingSecretId: m.model_definition?.secret_id ?? '',
          newSecret: {
            name: '',
            authMode: '',
            secretFields: {},
            configFields: {},
          },
          weight: totalWeight > 0 ? ((m.weight ?? 1.0) / totalWeight) * 100 : 100 / primaryMappings.length,
        })),
        fallbackModels: fallbackMappings
          .sort((a, b) => (a.fallback_order ?? 0) - (b.fallback_order ?? 0))
          .map((m, idx) => ({
            modelDefinitionId: m.model_definition?.model_definition_id,
            modelDefinitionName: m.model_definition?.name ?? '',
            provider: m.model_definition?.provider ?? '',
            modelName: m.model_definition?.model_name ?? '',
            secretMode: 'existing' as const,
            existingSecretId: m.model_definition?.secret_id ?? '',
            newSecret: {
              name: '',
              authMode: '',
              secretFields: {},
              configFields: {},
            },
            fallbackOrder: idx + 1,
          })),
      });
    }
  }, [endpoint, form]);

  const {
    mutateAsync: updateEndpoint,
    error: updateEndpointError,
    isLoading: isUpdatingEndpoint,
    reset: resetEndpointError,
  } = useUpdateEndpointMutation();

  const { mutateAsync: createSecret } = useCreateSecret();
  const { mutateAsync: createModelDefinition } = useCreateModelDefinitionMutation();
  const { mutateAsync: updateModelDefinition } = useUpdateModelDefinitionMutation();

  const [customError, setCustomError] = useState<Error | null>();

  const resetErrors = useCallback(() => {
    resetEndpointError();
    setCustomError(null);
  }, [resetEndpointError]);

  const isSubmitting = isUpdatingEndpoint;
  const mutationError = (customError || updateEndpointError) as Error | null;

  const handleSubmit = useCallback(
    async (values: EditEndpointFormData) => {
      if (!endpoint) return;

      try {
        const getSecretId = async (model: TrafficSplitModel | FallbackModel): Promise<string> => {
          if (model.secretMode === 'existing') {
            return model.existingSecretId;
          }

          const secretResponse = await createSecret({
            secret_name: model.newSecret.name,
            secret_value: model.newSecret.secretFields,
            provider: model.provider,
            auth_config: model.newSecret.configFields,
          });

          return secretResponse.secret.secret_id;
        };

        const getOrCreateModelDef = async (model: TrafficSplitModel | FallbackModel): Promise<string> => {
          // If model has a modelDefinitionId, check if it needs updating
          if (model.modelDefinitionId) {
            const originalMapping = endpoint.model_mappings?.find(
              (m) => m.model_definition?.model_definition_id === model.modelDefinitionId,
            );

            if (originalMapping?.model_definition) {
              const hasChanges =
                model.provider !== originalMapping.model_definition.provider ||
                model.modelName !== originalMapping.model_definition.model_name ||
                (model.secretMode === 'existing' &&
                  model.existingSecretId !== originalMapping.model_definition.secret_id) ||
                model.secretMode === 'new';

              if (hasChanges) {
                // Update the existing model definition
                const secretId = await getSecretId(model);
                await updateModelDefinition({
                  modelDefinitionId: model.modelDefinitionId,
                  secretId,
                  provider: model.provider,
                  modelName: model.modelName,
                });
              }
            }

            return model.modelDefinitionId;
          }

          // Otherwise, create a new model definition
          const secretId = await getSecretId(model);

          if (!model.provider || !model.modelName) {
            throw new Error('Provider and model name are required to create a model definition');
          }

          const timestamp = Date.now();
          const rawModelDefName = `${values.name || endpoint.name}-${model.provider}-${model.modelName}-${timestamp}`;
          let modelDefName = rawModelDefName
            .toLowerCase()
            .replace(/[^a-z0-9-]/g, '-')
            .replace(/-+/g, '-')
            .replace(/^-|-$/g, '');

          if (!modelDefName) {
            modelDefName = `model-definition-${timestamp}`;
          }

          const modelDefResponse = await createModelDefinition({
            name: modelDefName,
            secret_id: secretId,
            provider: model.provider,
            model_name: model.modelName,
          });

          return modelDefResponse.model_definition.model_definition_id;
        };

        const trafficSplitModelDefIds = await Promise.all(values.trafficSplitModels.map((m) => getOrCreateModelDef(m)));

        const fallbackModelDefIds = await Promise.all(values.fallbackModels.map((m) => getOrCreateModelDef(m)));

        const totalWeightPercent = values.trafficSplitModels.reduce((sum, m) => sum + m.weight, 0);
        const trafficSplitCount = values.trafficSplitModels.length || 1;

        const modelConfigs = [
          ...values.trafficSplitModels.map((m, idx) => ({
            model_definition_id: trafficSplitModelDefIds[idx],
            linkage_type: 'PRIMARY' as const,
            weight: totalWeightPercent > 0 ? m.weight / 100 : 1.0 / trafficSplitCount,
          })),
          ...values.fallbackModels.map((m, idx) => ({
            model_definition_id: fallbackModelDefIds[idx],
            linkage_type: 'FALLBACK' as const,
            fallback_order: m.fallbackOrder,
          })),
        ];

        const routingStrategy = values.trafficSplitModels.length > 1 ? 'REQUEST_BASED_TRAFFIC_SPLIT' : undefined;

        const fallbackConfig =
          values.fallbackModels.length > 0
            ? {
                strategy: 'SEQUENTIAL',
                max_attempts: values.fallbackModels.length + 1,
              }
            : undefined;

        await updateEndpoint({
          endpointId: endpoint.endpoint_id,
          name: values.name || undefined,
          model_configs: modelConfigs,
          routing_strategy: routingStrategy,
          fallback_config: fallbackConfig,
        });

        navigate(GatewayRoutes.getEndpointDetailsRoute(endpoint.endpoint_id));
      } catch {
        // Errors are captured by mutation error state
      }
    },
    [endpoint, updateEndpoint, navigate, createSecret, createModelDefinition, updateModelDefinition],
  );

  const handleCancel = useCallback(() => {
    navigate(GatewayRoutes.getEndpointDetailsRoute(endpointId));
  }, [navigate, endpointId]);

  const trafficSplitModels = form.watch('trafficSplitModels');
  const fallbackModels = form.watch('fallbackModels');

  const isFormComplete = useMemo(() => {
    if (trafficSplitModels.length === 0) return false;

    const totalWeight = trafficSplitModels.reduce((sum, m) => sum + m.weight, 0);
    if (Math.abs(totalWeight - 100) > 0.01) return false;

    const isModelComplete = (m: TrafficSplitModel | FallbackModel) => {
      if (!m.provider || !m.modelName) return false;

      if (m.secretMode === 'existing') {
        return Boolean(m.existingSecretId);
      } else {
        if (!m.newSecret) return false;

        const hasRequiredNewSecretFields = Boolean(m.newSecret.name && m.newSecret.authMode);
        const hasAtLeastOneSecretFieldValue = m.newSecret.secretFields
          ? Object.values(m.newSecret.secretFields).some((value) => value)
          : false;

        return hasRequiredNewSecretFields && hasAtLeastOneSecretFieldValue;
      }
    };

    const allTrafficSplitComplete = trafficSplitModels.every(isModelComplete);
    const allFallbackComplete = fallbackModels.every(isModelComplete);

    return allTrafficSplitComplete && allFallbackComplete;
  }, [trafficSplitModels, fallbackModels]);

  const name = form.watch('name');

  const hasChanges = useMemo(() => {
    if (!endpoint) return false;

    const originalName = endpoint.name ?? '';
    if (name !== originalName) return true;

    const originalPrimaryMappings = endpoint.model_mappings?.filter((m) => m.linkage_type === 'PRIMARY') ?? [];
    const originalFallbackMappings = endpoint.model_mappings?.filter((m) => m.linkage_type === 'FALLBACK') ?? [];

    if (trafficSplitModels.length !== originalPrimaryMappings.length) return true;
    if (fallbackModels.length !== originalFallbackMappings.length) return true;

    const originalTotalWeight = originalPrimaryMappings.reduce((sum, m) => sum + (m.weight ?? 1.0), 0);
    const trafficSplitChanged = trafficSplitModels.some((model, idx) => {
      const original = originalPrimaryMappings[idx];
      if (!original || !original.model_definition) return true;

      const originalWeightPercent =
        originalTotalWeight > 0
          ? ((original.weight ?? 1.0) / originalTotalWeight) * 100
          : 100 / originalPrimaryMappings.length;

      // Check if model definition ID changed (different model selected)
      if (model.modelDefinitionId && model.modelDefinitionId !== original.model_definition.model_definition_id) {
        return true;
      }

      // Check if model definition content changed (provider, model, or secret modified)
      if (model.modelDefinitionId && model.modelDefinitionId === original.model_definition.model_definition_id) {
        const modelDefChanged =
          model.provider !== original.model_definition.provider ||
          model.modelName !== original.model_definition.model_name ||
          (model.secretMode === 'existing' && model.existingSecretId !== original.model_definition.secret_id) ||
          model.secretMode === 'new';

        if (modelDefChanged) return true;
      }

      return (
        model.provider !== original.model_definition.provider ||
        model.modelName !== original.model_definition.model_name ||
        model.secretMode !== 'existing' ||
        model.existingSecretId !== original.model_definition.secret_id ||
        Math.abs(model.weight - originalWeightPercent) > 0.01
      );
    });

    const fallbackChanged = fallbackModels.some((model) => {
      // If no modelDefinitionId, it's a newly added model
      if (!model.modelDefinitionId) return true;

      const original = originalFallbackMappings.find(
        (m) => m.model_definition?.model_definition_id === model.modelDefinitionId,
      );

      if (!original) return true;

      if (original.model_definition) {
        const modelDefChanged =
          model.provider !== original.model_definition.provider ||
          model.modelName !== original.model_definition.model_name ||
          (model.secretMode === 'existing' && model.existingSecretId !== original.model_definition.secret_id) ||
          model.secretMode === 'new';

        if (modelDefChanged) return true;
      }

      return original.fallback_order !== model.fallbackOrder;
    });

    return trafficSplitChanged || fallbackChanged;
  }, [endpoint, name, trafficSplitModels, fallbackModels]);

  const { data: existingEndpoints } = useEndpointsQuery();

  const handleNameBlur = useCallback(() => {
    const name = form.getValues('name');
    if (name) {
      if (!isValidEndpointName(name)) {
        form.setError('name', {
          type: 'manual',
          message:
            'Name can only contain letters, numbers, underscores, hyphens, and dots. Spaces and special characters are not allowed.',
        });
        return;
      }
      const otherEndpoints = existingEndpoints?.filter((e) => e.endpoint_id !== endpointId);
      if (otherEndpoints?.some((e) => e.name === name)) {
        form.setError('name', {
          type: 'manual',
          message: 'An endpoint with this name already exists',
        });
      }
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
    isFormComplete,
    hasChanges,
    handleSubmit,
    handleCancel,
    handleNameBlur,
  };
}
