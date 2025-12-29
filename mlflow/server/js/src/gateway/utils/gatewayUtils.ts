import type { Endpoint } from '../types';

export const GATEWAY_MODEL_PREFIX = 'gateway:/';

export enum ModelProvider {
  GATEWAY = 'gateway',
  OTHER = 'other',
}

export const getModelProvider = (model: string | undefined): ModelProvider => {
  if (!model || model.startsWith(GATEWAY_MODEL_PREFIX)) {
    return ModelProvider.GATEWAY;
  }
  return ModelProvider.OTHER;
};

export const getEndpointNameFromModel = (model: string | undefined): string | undefined => {
  if (model?.startsWith(GATEWAY_MODEL_PREFIX)) {
    return model.replace(GATEWAY_MODEL_PREFIX, '');
  }
  return undefined;
};

export const formatModelFromEndpoint = (endpointName: string): string => {
  return `${GATEWAY_MODEL_PREFIX}${endpointName}`;
};

export const isDirectModel = (model: string | undefined): boolean => {
  return Boolean(model && !model.startsWith(GATEWAY_MODEL_PREFIX));
};

export const getEndpointDisplayInfo = (
  endpoint: Endpoint,
): { id: string; provider: string; modelName: string } | undefined => {
  const firstMapping = endpoint.model_mappings?.[0];
  if (firstMapping?.model_definition) {
    return {
      id: firstMapping.endpoint_id,
      provider: firstMapping.model_definition.provider,
      modelName: firstMapping.model_definition.model_name,
    };
  }
  return undefined;
};
