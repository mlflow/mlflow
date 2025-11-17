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
  route_id?: string;
  route_name?: string;
  secret_name?: string;
  provider?: string;
}

export interface CreateSecretRequest {
  secret_name: string;
  secret_value: string;
  field_name: string;
  resource_type: string;
  resource_id: string;
  is_shared?: boolean;
  created_by?: string;
}

export interface CreateSecretResponse {
  secret: Secret;
  binding: SecretBinding;
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

// Route types for route-centric architecture
export interface Route {
  route_id: string;
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

export interface CreateRouteRequest {
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

export interface CreateRouteResponse {
  secret?: Secret; // Present if new secret was created
  route: Route;
  binding: SecretBinding;
}

export interface UpdateRouteRequest {
  route_id: string;

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

export interface UpdateRouteResponse {
  route: Route;
  secret: Secret;
}

export interface ListRoutesResponse {
  routes: Route[];
}
