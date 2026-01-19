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

export const getEndpointNameFromGatewayModel = (model: string | undefined): string | undefined => {
  if (model?.startsWith(GATEWAY_MODEL_PREFIX)) {
    return model.replace(GATEWAY_MODEL_PREFIX, '');
  }
  return undefined;
};

export const formatGatewayModelFromEndpoint = (endpointName: string): string => {
  return `${GATEWAY_MODEL_PREFIX}${endpointName}`;
};

/**
 * Checks if the model is a non-gateway model (i.e., openai:/gpt-4, anthropic:/claude-3-5-sonnet, etc.
 * that doesn't use the gateway:/ prefix).
 */
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

/**
 * Validates if an endpoint name is URL-safe.
 * Matches the backend validation pattern: ASCII alphanumeric, underscore, hyphen, and dot only.
 * This allows the name to be used directly in URL paths.
 *
 * @param name - The endpoint name to validate
 * @returns true if the name is valid for use in URLs
 */
export const isValidEndpointName = (name: string): boolean => {
  return /^[a-zA-Z0-9_\-.]+$/.test(name);
};
