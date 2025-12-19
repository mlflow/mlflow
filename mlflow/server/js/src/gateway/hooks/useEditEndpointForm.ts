import { useForm } from 'react-hook-form';
import { useNavigate } from '../../common/utils/RoutingUtils';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useEffect, useCallback, useMemo, useState } from 'react';
import { GatewayApi } from '../api';
import { useCreateSecret } from './useCreateSecret';
import { useEndpointsQuery } from './useEndpointsQuery';
import { useEndpointQuery } from './useEndpointQuery';
import { useUpdateEndpointMutation } from './useUpdateEndpointMutation';
import { useUpdateModelDefinitionMutation } from './useUpdateModelDefinitionMutation';
import { getReadableErrorMessage, isSecretNameConflict } from '../utils/errorUtils';
import GatewayRoutes from '../routes';
import type { SecretMode } from '../components/model-configuration';
import type { Endpoint } from '../types';

export { getReadableErrorMessage };

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

  const provider = form.watch('provider');

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
  } = useCreateSecret();

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

  const resolveSecretId = useCallback(
    async (values: EditEndpointFormData): Promise<string | null> => {
      if (values.secretMode !== 'new') {
        return values.existingSecretId;
      }

      const authConfig: Record<string, any> = { ...values.newSecret.configFields };
      if (values.newSecret.authMode) {
        authConfig['auth_mode'] = values.newSecret.authMode;
      }
      const authConfigJson = Object.keys(authConfig).length > 0 ? JSON.stringify(authConfig) : undefined;

      try {
        const secretResponse = await createSecret({
          secret_name: values.newSecret.name,
          secret_value: values.newSecret.secretFields,
          provider: values.provider,
          auth_config_json: authConfigJson,
        });
        return (secretResponse as { secret: { secret_id: string } }).secret.secret_id;
      } catch (secretError) {
        if (!isSecretNameConflict(secretError)) {
          throw secretError;
        }
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
        if (values.name !== endpoint.name) {
          await updateEndpoint({ endpointId: endpoint.endpoint_id, name: values.name || undefined });
        }

        const secretId = await resolveSecretId(values);
        if (secretId === null) {
          return;
        }

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
        // Errors are captured by mutation error state
      }
    },
    [endpoint, primaryModelDef, updateEndpoint, updateModelDefinition, navigate, resolveSecretId],
  );

  const handleCancel = useCallback(() => {
    navigate(GatewayRoutes.getEndpointDetailsRoute(endpointId));
  }, [navigate, endpointId]);

  const modelName = form.watch('modelName');
  const secretMode = form.watch('secretMode');
  const existingSecretId = form.watch('existingSecretId');
  const newSecretName = form.watch('newSecret.name');
  const newSecretFields = form.watch('newSecret.secretFields');

  const isFormComplete = useMemo(() => {
    const hasSecretFieldValues = Object.values(newSecretFields || {}).some((v) => Boolean(v));
    const isSecretConfigured =
      secretMode === 'existing' ? Boolean(existingSecretId) : Boolean(newSecretName) && hasSecretFieldValues;
    return Boolean(provider) && Boolean(modelName) && isSecretConfigured;
  }, [secretMode, existingSecretId, newSecretName, newSecretFields, provider, modelName]);

  const name = form.watch('name');

  const hasChanges = useMemo(() => {
    if (!endpoint || !primaryModelDef) return false;

    const originalName = endpoint.name ?? '';

    if (name !== originalName) return true;

    const originalProvider = primaryModelDef.provider ?? '';
    const originalModelName = primaryModelDef.model_name ?? '';
    const originalSecretId = primaryModelDef.secret_id ?? '';

    if (secretMode === 'new') {
      const hasNewSecretData = Boolean(newSecretName) || Object.values(newSecretFields || {}).some((v) => Boolean(v));
      if (hasNewSecretData) return true;
    }

    return (
      provider !== originalProvider ||
      modelName !== originalModelName ||
      (secretMode === 'existing' && existingSecretId !== originalSecretId)
    );
  }, [
    endpoint,
    primaryModelDef,
    name,
    provider,
    modelName,
    secretMode,
    existingSecretId,
    newSecretName,
    newSecretFields,
  ]);

  const { data: existingEndpoints } = useEndpointsQuery();

  const handleNameBlur = useCallback(() => {
    const name = form.getValues('name');
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
    isFormComplete,
    hasChanges,
    handleSubmit,
    handleCancel,
    handleNameBlur,
  };
}
