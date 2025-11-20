export enum ResourceType {
  SCORER_JOB = 'SCORER_JOB',
  GLOBAL = 'GLOBAL',
  ROUTE = 'ROUTE',
}

export interface Secret {
  secret_id: string;
  secret_name: string;
  masked_value: string;
  is_shared: boolean;
  created_at: number;
  last_updated_at: number;
  created_by?: string;
  last_updated_by?: string;
  binding_count?: number;
  provider?: string;
}

export interface SecretBinding {
  binding_id: string;
  secret_id: string;
  resource_type: string;
  resource_id: string;
  field_name: string;
  created_at: number;
  last_updated_at: number;
  created_by?: string;
  last_updated_by?: string;
  endpoint_id?: string;
  route_name?: string;
  secret_name?: string;
  provider?: string;
}

export interface CreateSecretRequest {
  secret_name: string;
  secret_value: string;
  provider?: string;
  is_shared?: boolean;
  created_by?: string;
  auth_config?: string;
}

export interface CreateSecretResponse {
  secret: Secret;
}

export interface UpdateSecretRequest {
  secret_id: string;
  secret_value: string;
  updated_by?: string;
}

export interface DeleteSecretRequest {
  secret_id: string;
}

export interface ListSecretsResponse {
  secrets: Secret[];
}

export interface BindSecretRequest {
  secret_id: string;
  resource_type: ResourceType;
  resource_id: string;
}

export interface UnbindSecretRequest {
  binding_id: string;
}

export interface ListBindingsRequest {
  secret_id: string;
}

export interface ListBindingsResponse {
  bindings: SecretBinding[];
}

// Endpoint types for endpoint-centric architecture
// API response format - single route per model
export interface RouteResponse {
  endpoint_id: string;
  secret_id: string;
  secret_name?: string; // Name of the secret/API key being used
  model_name: string;
  name?: string;
  description?: string;
  provider?: string;
  created_at: number;
  last_updated_at: number;
  created_by?: string;
  last_updated_by?: string;
  binding_count?: number;
  tags?: Array<{ key: string; value: string }> | Record<string, string>;
}

export interface CreateEndpointRequest_Legacy {
  // For "Add Route" flow (use existing secret - calls create-route-and-bind)
  secret_id?: string;

  // For "Create Route" flow (create new secret - calls create-and-bind)
  secret_name?: string;
  secret_value?: string;
  provider?: string;
  is_shared?: boolean;
  auth_config?: string; // JSON stringified object

  // Common fields for both flows
  model_name: string;
  route_name?: string;
  route_description?: string;
  route_tags?: string; // JSON stringified array
  resource_type: string;
  resource_id: string;
  field_name: string;
  created_by?: string;
}

export interface CreateEndpointResponse_Legacy {
  secret?: Secret; // Present if new secret was created
  route: RouteResponse;
  binding?: SecretBinding;
}

export interface UpdateEndpointRequest_Legacy {
  endpoint_id: string;

  // Option 1: Update to existing secret
  secret_id?: string;

  // Option 2: Create new secret and update route
  secret_name?: string;
  secret_value?: string;
  provider?: string;
  is_shared?: boolean;
  auth_config?: string;

  // Option 3: Update route metadata
  route_description?: string;
  route_tags?: string;
}

export interface UpdateEndpointResponse_Legacy {
  route: RouteResponse;
  secret: Secret;
}

export interface ListEndpointsResponse_Legacy {
  routes: RouteResponse[];
}

// Endpoint model type (individual model within an endpoint)
export interface EndpointModel {
  model_id: string;
  model_name: string;
  secret_id: string;
  secret_name?: string;
  provider?: string;
  created_at: number;
  last_updated_at: number;
  created_by?: string;
  last_updated_by?: string;
}

export interface Endpoint {
  endpoint_id: string;
  name?: string;
  description?: string;
  models: EndpointModel[];
  tags?: Array<{ key: string; value: string }> | Record<string, string>;
  created_at: number;
  last_updated_at: number;
  created_by?: string;
  last_updated_by?: string;
}

export interface CreateEndpointRequest {
  name?: string;
  description?: string;
  tags?: Array<{ key: string; value: string }>; // Array of tag objects
  models: Array<{
    model_name: string;
    secret_id: string;
  }>;
  created_by?: string;
}

export interface CreateEndpointResponse {
  endpoint: Endpoint;
}

export interface UpdateEndpointRequest {
  endpoint_id: string;
  name?: string;
  description?: string;
  tags?: Array<{ key: string; value: string }>; // Array of tag objects
}

export interface UpdateEndpointResponse {
  endpoint: Endpoint;
}

export interface ListEndpointsResponse {
  routes: Endpoint[];
}

export interface BindEndpointRequest {
  endpoint_id: string;
  resource_type: string;
  resource_id: string;
  field_name: string;
}

export interface BindEndpointResponse {
  binding_id: string;
}

// Model-level CRUD operations
export interface AddEndpointModelRequest {
  endpoint_id: string;
  model_name: string;
  secret_id: string;
  routing_config?: Record<string, any>;
  created_by?: string;
}

export interface AddEndpointModelResponse {
  endpoint: Endpoint;
}

export interface UpdateEndpointModelRequest {
  endpoint_id: string;
  model_id: string;
  model_name?: string;
  secret_id?: string;
  routing_config?: Record<string, any>;
  updated_by?: string;
}

export interface UpdateEndpointModelResponse {
  endpoint: Endpoint;
}

export interface RemoveEndpointModelRequest {
  endpoint_id: string;
  model_id: string;
}

export interface RemoveEndpointModelResponse {
  endpoint: Endpoint;
}
