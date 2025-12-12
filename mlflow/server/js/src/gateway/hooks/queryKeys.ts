/**
 * Centralized query keys for Gateway React Query hooks.
 * Using constants ensures consistency between queries and invalidations.
 */
export const GatewayQueryKeys = {
  // Base keys for cache invalidation
  endpoints: 'gateway_endpoints' as const,
  modelDefinitions: 'gateway_model_definitions' as const,
  secrets: 'gateway_secrets' as const,
  providers: 'gateway_providers' as const,
  models: 'gateway_models' as const,
  bindings: 'gateway_bindings' as const,
  secretsConfig: 'gateway_secrets_config' as const,

  // Full query key factories
  endpointsList: (params?: { provider?: string }) => [GatewayQueryKeys.endpoints, params ?? {}] as const,

  endpoint: (endpointId: string) => [GatewayQueryKeys.endpoints, { endpointId }] as const,

  modelDefinitionsList: () => [GatewayQueryKeys.modelDefinitions] as const,

  secretsList: (params?: { provider?: string }) => [GatewayQueryKeys.secrets, params ?? {}] as const,

  secret: (secretId: string) => [GatewayQueryKeys.secrets, { secretId }] as const,

  providersList: () => [GatewayQueryKeys.providers] as const,

  providerConfig: (provider: string) => [GatewayQueryKeys.providers, { provider }] as const,

  modelsList: (provider: string) => [GatewayQueryKeys.models, { provider }] as const,

  bindingsList: () => [GatewayQueryKeys.bindings] as const,

  secretsConfigQuery: () => [GatewayQueryKeys.secretsConfig] as const,
} as const;
