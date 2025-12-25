import { useForm } from 'react-hook-form';
import { useCallback } from 'react';
import { useNavigate } from '../../common/utils/RoutingUtils';
import { useEndpointsQuery } from './useEndpointsQuery';
import { useModelsQuery } from './useModelsQuery';
import GatewayRoutes from '../routes';
import type { ProviderModel } from '../types';

export interface CreateEndpointFormData {
  name: string;
  provider: string;
  modelName: string;
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

export function useCreateEndpointForm(): UseCreateEndpointFormResult {
  const navigate = useNavigate();

  const form = useForm<CreateEndpointFormData>({
    defaultValues: {
      name: '',
      provider: '',
      modelName: '',
    },
  });

  const resetErrors = useCallback(() => {
    // Will be expanded when we add actual mutations
  }, []);

  const handleSubmit = async (_values: CreateEndpointFormData) => {
    // TODO: Implement actual endpoint creation in a future PR
    // For now, this is a placeholder - we need model selection and API key config first
  };

  const handleCancel = () => {
    navigate(GatewayRoutes.gatewayPageRoute);
  };

  const provider = form.watch('provider');
  const name = form.watch('name');
  const modelName = form.watch('modelName');

  const { data: existingEndpoints } = useEndpointsQuery();

  const { data: models } = useModelsQuery({ provider: provider || undefined });
  const selectedModel = models?.find((m) => m.model === modelName);

  const handleNameBlur = () => {
    const currentName = form.getValues('name');
    if (currentName && existingEndpoints?.some((e) => e.name === currentName)) {
      form.setError('name', {
        type: 'manual',
        message: 'An endpoint with this name already exists',
      });
    }
  };

  const isFormComplete = !!provider && !!modelName;

  return {
    form,
    isLoading: false,
    error: null,
    resetErrors,
    existingEndpoints,
    selectedModel,
    isFormComplete,
    handleSubmit,
    handleCancel,
    handleNameBlur,
  };
}
