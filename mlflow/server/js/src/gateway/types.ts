export interface Provider {
  name: string;
}

export interface ProviderModel {
  model: string;
  provider: string;
  supports_function_calling: boolean;
  supports_vision?: boolean;
  supports_reasoning?: boolean;
  supports_prompt_caching?: boolean;
  supports_response_schema?: boolean;
  max_input_tokens?: number;
  max_output_tokens?: number;
  input_cost_per_token?: number;
  output_cost_per_token?: number;
  deprecation_date?: string;
}

export interface SecretField {
  name: string;
  type: string;
  description?: string;
  required: boolean;
}

export interface ConfigField {
  name: string;
  type: string;
  description?: string;
  required: boolean;
}

export interface AuthMode {
  mode: string;
  display_name: string;
  description?: string;
  secret_fields: SecretField[];
  config_fields: ConfigField[];
}

export interface ProviderConfig {
  auth_modes: AuthMode[];
  default_mode: string;
}

export interface ProvidersResponse {
  providers: string[];
}

export interface ModelsResponse {
  models: ProviderModel[];
}

export interface SecretInfo {
  secret_id: string;
  secret_name: string;
  masked_values: Record<string, string>;
  provider?: string;
  auth_config?: Record<string, string>;
  created_at: number;
  last_updated_at: number;
  created_by?: string;
  last_updated_by?: string;
}

export interface CreateSecretRequest {
  secret_name: string;
  secret_value: Record<string, string>;
  provider?: string;
  auth_config?: Record<string, string>;
  created_by?: string;
}

export interface CreateSecretInfoResponse {
  secret: SecretInfo;
}

export interface GetSecretInfoResponse {
  secret: SecretInfo;
}

export interface UpdateSecretRequest {
  secret_id: string;
  secret_value?: Record<string, string>;
  auth_config?: Record<string, string>;
  updated_by?: string;
}

export interface UpdateSecretInfoResponse {
  secret: SecretInfo;
}

export interface ListSecretInfosResponse {
  secrets: SecretInfo[];
}

export interface ModelDefinition {
  model_definition_id: string;
  name: string;
  secret_id: string;
  secret_name: string;
  provider: string;
  model_name: string;
  created_at: number;
  last_updated_at: number;
  created_by?: string;
  last_updated_by?: string;
  endpoint_count: number;
}

export interface EndpointModelMapping {
  mapping_id: string;
  endpoint_id: string;
  model_definition_id: string;
  model_definition?: ModelDefinition;
  weight: number;
  linkage_type?: string;
  fallback_order?: number;
  created_at: number;
  created_by?: string;
}

export interface GatewayEndpointModelConfig {
  model_definition_id: string;
  linkage_type: string;
  weight?: number;
  fallback_order?: number;
}

export interface Endpoint {
  endpoint_id: string;
  name: string;
  model_mappings: EndpointModelMapping[];
  created_at: number;
  last_updated_at: number;
  created_by?: string;
  last_updated_by?: string;
  routing_strategy?: string;
  fallback_config?: {
    strategy: string;
    max_attempts: number;
  };
  experiment_id?: string;
  usage_tracking?: boolean;
}

export interface CreateEndpointRequest {
  name?: string;
  model_configs: GatewayEndpointModelConfig[];
  created_by?: string;
  routing_strategy?: string;
  fallback_config?: {
    strategy: string;
    max_attempts: number;
  };
  usage_tracking?: boolean;
  experiment_id?: string;
}

export interface CreateEndpointResponse {
  endpoint: Endpoint;
}

export interface GetEndpointResponse {
  endpoint: Endpoint;
}

export interface UpdateEndpointRequest {
  endpoint_id: string;
  name?: string;
  updated_by?: string;
  routing_strategy?: string;
  fallback_config?: {
    strategy: string;
    max_attempts: number;
  };
  model_configs?: GatewayEndpointModelConfig[];
  usage_tracking?: boolean;
  experiment_id?: string;
}

export interface UpdateEndpointResponse {
  endpoint: Endpoint;
}

export interface ListEndpointsResponse {
  endpoints: Endpoint[];
}

// Model Definition CRUD
export interface CreateModelDefinitionRequest {
  name: string;
  secret_id: string;
  provider: string;
  model_name: string;
  created_by?: string;
}

export interface CreateModelDefinitionResponse {
  model_definition: ModelDefinition;
}

export interface GetModelDefinitionResponse {
  model_definition: ModelDefinition;
}

export interface ListModelDefinitionsResponse {
  model_definitions: ModelDefinition[];
}

export interface UpdateModelDefinitionRequest {
  model_definition_id: string;
  name?: string;
  secret_id?: string;
  provider?: string;
  model_name?: string;
  last_updated_by?: string;
}

export interface UpdateModelDefinitionResponse {
  model_definition: ModelDefinition;
}

// Attach/Detach Model to Endpoint
export interface AttachModelToEndpointRequest {
  endpoint_id: string;
  model_config: GatewayEndpointModelConfig;
  created_by?: string;
}

export interface AttachModelToEndpointResponse {
  mapping: EndpointModelMapping;
}

export interface DetachModelFromEndpointRequest {
  endpoint_id: string;
  model_definition_id: string;
}

export type ResourceType = 'scorer';

export interface EndpointBinding {
  endpoint_id: string;
  resource_type: ResourceType;
  resource_id: string;
  created_at: number;
  last_updated_at?: number;
  created_by?: string;
  last_updated_by?: string;
  display_name?: string;
}

export interface CreateEndpointBindingRequest {
  endpoint_id: string;
  resource_type: ResourceType;
  resource_id: string;
  created_by?: string;
}

export interface CreateEndpointBindingResponse {
  binding: EndpointBinding;
}

export interface ListEndpointBindingsResponse {
  bindings: EndpointBinding[];
}

export interface SecretsConfigResponse {
  secrets_available: boolean;
  using_default_passphrase: boolean;
}

export interface UserInfo {
  id: number;
  username: string;
}

export interface ListUsersResponse {
  users: UserInfo[];
}

// Budget Policy types
export type BudgetUnit = 'USD';
export type DurationUnit = 'MINUTES' | 'HOURS' | 'DAYS' | 'WEEKS' | 'MONTHS';
export type TargetScope = 'GLOBAL' | 'WORKSPACE';
export type BudgetAction = 'ALERT' | 'REJECT';

export interface BudgetDuration {
  unit: DurationUnit;
  value: number;
}

export interface BudgetPolicy {
  budget_policy_id: string;
  budget_unit: BudgetUnit;
  budget_amount: number;
  duration: BudgetDuration;
  target_scope: TargetScope;
  budget_action: BudgetAction;
  created_at: number;
  last_updated_at: number;
  created_by?: string | null;
  last_updated_by?: string | null;
  workspace?: string | null;
}

export interface CreateBudgetPolicyRequest {
  budget_unit: BudgetUnit;
  budget_amount: number;
  duration: BudgetDuration;
  target_scope: TargetScope;
  budget_action: BudgetAction;
}

export interface CreateBudgetPolicyResponse {
  budget_policy: BudgetPolicy;
}

export interface GetBudgetPolicyResponse {
  budget_policy: BudgetPolicy;
}

export interface UpdateBudgetPolicyRequest {
  budget_policy_id: string;
  budget_unit?: BudgetUnit;
  budget_amount?: number;
  duration?: BudgetDuration;
  target_scope?: TargetScope;
  budget_action?: BudgetAction;
}

export interface UpdateBudgetPolicyResponse {
  budget_policy: BudgetPolicy;
}

export interface ListBudgetPoliciesResponse {
  budget_policies: BudgetPolicy[];
  next_page_token?: string;
}

export interface BudgetPolicyWindow {
  budget_policy_id: string;
  window_start_ms: number;
  window_end_ms: number;
  current_spend: number;
}

export interface ListBudgetWindowsResponse {
  windows: BudgetPolicyWindow[];
}

// Guardrail types — aligned with backend proto (service.proto)
export type GuardrailStage = 'BEFORE' | 'AFTER';
export type GuardrailAction = 'VALIDATION' | 'SANITIZATION';

/** Matches GatewayGuardrail proto message. */
export interface GatewayGuardrail {
  guardrail_id: string;
  name: string;
  scorer?: {
    scorer_id: string;
    scorer_version: number;
    serialized_scorer?: string;
  };
  stage: GuardrailStage;
  action: GuardrailAction;
  action_endpoint_id?: string;
  created_by?: string;
  created_at: number;
  last_updated_by?: string;
  last_updated_at: number;
}

/** Matches GatewayGuardrailConfig proto message. */
export interface GatewayGuardrailConfig {
  endpoint_id: string;
  guardrail_id: string;
  execution_order?: number;
  created_by?: string;
  created_at: number;
  guardrail?: GatewayGuardrail;
}

/** Request for POST /mlflow/gateway/guardrails/create */
export interface CreateGatewayGuardrailRequest {
  name: string;
  scorer_id: string;
  scorer_version: number;
  stage: GuardrailStage;
  action: GuardrailAction;
  action_endpoint_id?: string;
}

export interface CreateGatewayGuardrailResponse {
  guardrail: GatewayGuardrail;
}

/** Request for GET /mlflow/gateway/guardrails/get */
export interface GetGatewayGuardrailRequest {
  guardrail_id: string;
}

export interface GetGatewayGuardrailResponse {
  guardrail: GatewayGuardrail;
}

/** Request for DELETE /mlflow/gateway/guardrails/delete */
export interface DeleteGatewayGuardrailRequest {
  guardrail_id: string;
}

/** Request for GET /mlflow/gateway/guardrails/list */
export interface ListGatewayGuardrailsRequest {
  max_results?: number;
  page_token?: string;
}

export interface ListGatewayGuardrailsResponse {
  guardrails: GatewayGuardrail[];
  next_page_token?: string;
}

/** Request for POST /mlflow/gateway/guardrails/add-to-endpoint */
export interface AddGuardrailToEndpointRequest {
  endpoint_id: string;
  guardrail_id: string;
  execution_order?: number;
}

export interface AddGuardrailToEndpointResponse {
  config: GatewayGuardrailConfig;
}

/** Request for DELETE /mlflow/gateway/guardrails/remove-from-endpoint */
export interface RemoveGuardrailFromEndpointRequest {
  endpoint_id: string;
  guardrail_id: string;
}

/** Request for GET /mlflow/gateway/guardrails/list-for-endpoint */
export interface ListEndpointGuardrailConfigsRequest {
  endpoint_id: string;
}

export interface ListEndpointGuardrailConfigsResponse {
  configs: GatewayGuardrailConfig[];
}

/** Request for PATCH /mlflow/gateway/guardrails/update-config */
export interface UpdateEndpointGuardrailConfigRequest {
  endpoint_id: string;
  guardrail_id: string;
  execution_order?: number;
}

export interface UpdateEndpointGuardrailConfigResponse {
  config: GatewayGuardrailConfig;
}
