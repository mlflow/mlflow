export enum ResourceType {
  SCORER_JOB = 'SCORER_JOB',
  GLOBAL = 'GLOBAL',
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
}

export interface SecretBinding {
  binding_id: string;
  secret_id: string;
  resource_type: string;
  resource_id: string;
  field_name: string;
  created_at: number;
  last_updated_at: number;
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
  resource_type: string;
  resource_id: string;
  field_name: string;
}

export interface ListBindingsRequest {
  secret_id: string;
}

export interface ListBindingsResponse {
  bindings: SecretBinding[];
}
