# AI Gateway: API Keys + Routes Architecture

## Overview

This document describes the architecture for MLflow's AI Gateway feature, which manages API keys and model routes for AI providers. The design separates concerns between authentication (API Keys) and configuration (Routes) to enable key reuse, automatic key rotation propagation, and support for complex provider authentication.

## Motivation

**Problems with coupled design:**
- Duplicate API keys when using multiple models from the same provider
- Key rotation requires updating every model configuration individually
- Difficult to track which models share the same credentials
- No support for complex provider authentication (AWS Bedrock, Azure, Vertex AI)
- No support for provider-specific base URLs

**Goals:**
- One API key → many model routes
- Key rotation automatically updates all routes
- Support complex provider authentication (base URLs, multi-field credentials)
- Clear separation: keys are secrets, routes are configurations
- Model-centric UX with independent key management
- Support clone functionality without exposing API keys

## Architecture

### Database Schema

#### Secrets Table (API Keys - Provider Authentication)
```python
Table: secrets
- secret_id (PK, UUID)                       # Unique identifier
- secret_value (LargeBinary, encrypted)      # Simple API key (OpenAI, Anthropic)
- provider (String)                          # openai, anthropic, bedrock, vertex_ai, azure, databricks
- auth_config (LargeBinary, encrypted)       # JSON for complex authentication
- is_shared (Boolean)                        # Shared vs private key
- created_at (Integer, timestamp_ms)
- updated_at (Integer, timestamp_ms)
- created_by (String, nullable)
- updated_by (String, nullable)

Indexes:
- PRIMARY KEY (secret_id)
- INDEX idx_secrets_provider (provider)
```

#### Routes Table (Model Configurations)
```python
Table: routes
- route_id (PK, UUID)                        # Unique identifier
- secret_id (FK, UUID)                       # References secrets.secret_id
- model_name (String)                        # gpt-4, claude-sonnet-4-5, etc.
- name (String, nullable)                    # User-friendly display name
- model_config (String, encrypted)           # JSON for runtime params & overrides
- created_at (Integer, timestamp_ms)
- updated_at (Integer, timestamp_ms)
- created_by (String, nullable)
- updated_by (String, nullable)

Indexes:
- PRIMARY KEY (route_id)
- FOREIGN KEY (secret_id) REFERENCES secrets(secret_id) ON DELETE CASCADE
- INDEX idx_routes_secret_id (secret_id)
```

#### Bindings Table (Resource Associations)
```python
Table: bindings
- binding_id (PK, UUID)                      # Unique identifier
- route_id (FK, UUID)                        # References routes.route_id (CHANGED from secret_id)
- resource_type (String)                     # SCORER_JOB, GLOBAL
- resource_id (String)                       # Resource identifier
- field_name (String)                        # Environment variable name
- created_at (Integer, timestamp_ms)
- created_by (String, nullable)

Indexes:
- PRIMARY KEY (binding_id)
- FOREIGN KEY (route_id) REFERENCES routes(route_id) ON DELETE CASCADE
- INDEX idx_bindings_route_id (route_id)
- INDEX idx_bindings_resource (resource_type, resource_id)
```

### JSON Field Schemas

#### auth_config (Provider Authentication)

**Simple Providers (use secret_value, auth_config = NULL):**
```python
# OpenAI
secret_value = "sk-proj-..."
auth_config = None

# Anthropic
secret_value = "sk-ant-..."
auth_config = None
```

**Complex Providers (use auth_config, secret_value = NULL):**

```python
# AWS Bedrock
auth_config = {
    "aws_access_key_id": "AKIA...",
    "aws_secret_access_key": "...",
    "aws_region": "us-east-1",
    "aws_session_token": "..."  # Optional, for temporary credentials
}

# Google Vertex AI
auth_config = {
    "project_id": "my-gcp-project",
    "location": "us-central1",
    "service_account_json": "{...}"  # Entire service account key as JSON string
}

# Azure OpenAI
auth_config = {
    "api_key": "...",
    "azure_endpoint": "https://my-resource.openai.azure.com",  # Base URL
    "api_version": "2024-02-15-preview"
}

# Databricks
auth_config = {
    "databricks_host": "https://dbc-abc123.cloud.databricks.com",  # Base URL
    "databricks_token": "dapi..."
}

# Custom Provider
auth_config = {
    "base_url": "https://custom-api.example.com/v1",
    "api_key": "...",
    "headers": {
        "X-Custom-Header": "value"
    }
}
```

#### model_config (Route Runtime Parameters)

```python
# Basic runtime parameters (provider-agnostic)
model_config = {
    "temperature": 0.7,
    "max_tokens": 4096,
    "top_p": 0.9,
    "timeout": 30
}

# Azure-specific deployment override
model_config = {
    "temperature": 0.7,
    "azure_deployment": "gpt-4-turbo-deployment"
}

# Bedrock-specific model ID override
model_config = {
    "temperature": 0.7,
    "bedrock_model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
}

# Advanced with retry logic
model_config = {
    "temperature": 0.7,
    "max_retries": 3,
    "retry_delay": 1.0,
    "fallback_model": "gpt-4o-mini"
}
```

### Protocol Buffer Definitions

```protobuf
// Secret = Provider authentication credentials
message Secret {
  optional string secret_id = 1;
  optional string provider = 2;
  optional string secret_value = 3;     // Encrypted, for simple providers
  optional string auth_config = 4;      // Encrypted JSON, for complex providers
  optional bool is_shared = 5;
  optional int64 created_at = 6;
  optional int64 updated_at = 7;
  optional string created_by = 8;
  optional string updated_by = 9;
}

// Route = Model configuration
message Route {
  optional string route_id = 1;
  optional string secret_id = 2;
  optional string model_name = 3;
  optional string name = 4;             // User-friendly display name
  optional string model_config = 5;     // Encrypted JSON runtime parameters
  optional int64 created_at = 6;
  optional int64 updated_at = 7;
  optional string created_by = 8;
  optional string updated_by = 9;

  // Joined data (not persisted)
  optional Secret secret = 100;         // For list/detail responses
}

// Binding = Resource association
message Binding {
  optional string binding_id = 1;
  optional string route_id = 2;         // Changed from secret_id
  optional string resource_type = 3;
  optional string resource_id = 4;
  optional string field_name = 5;
  optional int64 created_at = 6;
  optional string created_by = 7;
}

// Request/Response messages for Secret APIs
message CreateSecret {
  option (scalapb.message).extends = "com.databricks.rpc.RPC[$this.Response]";

  optional string provider = 1 [(validate_required) = true];
  optional string secret_value = 2;     // For simple providers
  optional string auth_config = 3;      // For complex providers (JSON string)
  optional bool is_shared = 4;

  message Response {
    optional Secret secret = 1;
  }
}

message UpdateSecret {
  option (scalapb.message).extends = "com.databricks.rpc.RPC[$this.Response]";

  optional string secret_id = 1 [(validate_required) = true];
  optional string secret_value = 2;     // New key value
  optional string auth_config = 3;      // New auth config
  optional string updated_by = 4;

  message Response {
    optional Secret secret = 1;
    repeated Route affected_routes = 2; // Routes using this key
  }
}

message DeleteSecret {
  option (scalapb.message).extends = "com.databricks.rpc.RPC[$this.Response]";

  optional string secret_id = 1 [(validate_required) = true];

  message Response {
    optional string deleted_secret_id = 1;
    repeated Route deleted_routes = 2;
    repeated Binding deleted_bindings = 3;
  }
}

message ListSecrets {
  option (scalapb.message).extends = "com.databricks.rpc.RPC[$this.Response]";

  optional string provider = 1;         // Filter by provider

  message Response {
    repeated Secret secrets = 1;
  }
}

message GetSecret {
  option (scalapb.message).extends = "com.databricks.rpc.RPC[$this.Response]";

  optional string secret_id = 1 [(validate_required) = true];

  message Response {
    optional Secret secret = 1;
    repeated Route routes = 2;          // Routes using this key
  }
}

// Request/Response messages for Route APIs
message CreateRoute {
  option (scalapb.message).extends = "com.databricks.rpc.RPC[$this.Response]";

  // Option 1: Use existing key
  optional string secret_id = 1;

  // Option 2: Create new key
  optional string provider = 2;
  optional string secret_value = 3;
  optional string auth_config = 4;      // JSON string
  optional bool is_shared = 5;

  // Route configuration
  optional string model_name = 6 [(validate_required) = true];
  optional string name = 7;
  optional string model_config = 8;     // JSON string

  // Bindings
  repeated CreateBinding bindings = 9;

  message CreateBinding {
    optional string resource_type = 1 [(validate_required) = true];
    optional string resource_id = 2 [(validate_required) = true];
    optional string field_name = 3 [(validate_required) = true];
  }

  message Response {
    optional Route route = 1;
    optional Secret secret = 2;         // If created new
    repeated Binding bindings = 3;
  }
}

message CloneRoute {
  option (scalapb.message).extends = "com.databricks.rpc.RPC[$this.Response]";

  optional string route_id = 1 [(validate_required) = true];
  optional string model_name = 2;       // Override model
  optional string name = 3;             // New name
  optional string model_config = 4;     // Override config

  message Response {
    optional Route route = 1;
  }
}

message UpdateRoute {
  option (scalapb.message).extends = "com.databricks.rpc.RPC[$this.Response]";

  optional string route_id = 1 [(validate_required) = true];
  optional string model_name = 2;
  optional string name = 3;
  optional string model_config = 4;     // JSON string
  optional string updated_by = 5;

  message Response {
    optional Route route = 1;
  }
}

message DeleteRoute {
  option (scalapb.message).extends = "com.databricks.rpc.RPC[$this.Response]";

  optional string route_id = 1 [(validate_required) = true];

  message Response {
    optional string deleted_route_id = 1;
    repeated Binding deleted_bindings = 2;
  }
}

message ListRoutes {
  option (scalapb.message).extends = "com.databricks.rpc.RPC[$this.Response]";

  optional string provider = 1;         // Filter by provider

  message Response {
    repeated Route routes = 1;          // Includes joined secret info
  }
}

message GetRoute {
  option (scalapb.message).extends = "com.databricks.rpc.RPC[$this.Response]";

  optional string route_id = 1 [(validate_required) = true];

  message Response {
    optional Route route = 1;
    repeated Binding bindings = 2;
  }
}

// Request/Response messages for Binding APIs
message CreateBinding {
  option (scalapb.message).extends = "com.databricks.rpc.RPC[$this.Response]";

  optional string route_id = 1 [(validate_required) = true];
  optional string resource_type = 2 [(validate_required) = true];
  optional string resource_id = 3 [(validate_required) = true];
  optional string field_name = 4 [(validate_required) = true];

  message Response {
    optional Binding binding = 1;
  }
}

message DeleteBinding {
  option (scalapb.message).extends = "com.databricks.rpc.RPC[$this.Response]";

  optional string binding_id = 1 [(validate_required) = true];

  message Response {
    optional string deleted_binding_id = 1;
  }
}

message ListBindings {
  option (scalapb.message).extends = "com.databricks.rpc.RPC[$this.Response]";

  optional string route_id = 1 [(validate_required) = true];

  message Response {
    repeated Binding bindings = 1;
  }
}
```

## API Endpoints

### Secret Management APIs

#### Create API Key
```
POST /ajax-api/3.0/mlflow/secrets/create

Request (simple provider):
{
  "provider": "openai",
  "secret_value": "sk-...",
  "is_shared": true
}

Request (complex provider):
{
  "provider": "bedrock",
  "auth_config": {
    "aws_access_key_id": "AKIA...",
    "aws_secret_access_key": "...",
    "aws_region": "us-east-1"
  },
  "is_shared": true
}

Response:
{
  "secret": {
    "secret_id": "uuid-...",
    "provider": "openai",
    "is_shared": true,
    "created_at": 1234567890,
    "updated_at": 1234567890
  }
}
```

#### Update API Key (Rotate Credentials)
```
PUT /ajax-api/3.0/mlflow/secrets/update

Request:
{
  "secret_id": "uuid-...",
  "secret_value": "sk-new-...",     // For simple providers
  "auth_config": {...},             // For complex providers
  "updated_by": "user@example.com"
}

Response:
{
  "secret": {...},
  "affected_routes": [              // Routes that will use new key
    {
      "route_id": "uuid-...",
      "model_name": "gpt-4o",
      "name": "Production GPT-4o"
    },
    ...
  ]
}
```

#### Delete API Key
```
DELETE /ajax-api/3.0/mlflow/secrets/{secret_id}

Response:
{
  "deleted_secret_id": "uuid-...",
  "deleted_routes": [...],          // Cascade delete
  "deleted_bindings": [...]         // Cascade delete
}
```

#### List API Keys
```
GET /ajax-api/3.0/mlflow/secrets/list?provider=openai

Response:
{
  "secrets": [
    {
      "secret_id": "uuid-...",
      "provider": "openai",
      "is_shared": true,
      "created_at": 1234567890
    },
    ...
  ]
}
```

#### Get API Key Details
```
GET /ajax-api/3.0/mlflow/secrets/{secret_id}

Response:
{
  "secret": {...},
  "routes": [                       // Routes using this key
    {
      "route_id": "uuid-...",
      "model_name": "gpt-4o",
      "name": "Production"
    },
    ...
  ]
}
```

### Route Management APIs

#### Create Route with New Key (Atomic)
```
POST /ajax-api/3.0/mlflow/routes/create

Request:
{
  "provider": "openai",
  "secret_value": "sk-...",         // Create new key
  "model_name": "gpt-4o",
  "name": "Production GPT-4o",
  "model_config": {
    "temperature": 0.7
  },
  "bindings": [
    {
      "resource_type": "SCORER_JOB",
      "resource_id": "job-123",
      "field_name": "OPENAI_API_KEY"
    }
  ],
  "is_shared": true
}

Response:
{
  "route": {...},
  "secret": {...},                  // Newly created secret
  "bindings": [...]
}
```

#### Create Route with Existing Key
```
POST /ajax-api/3.0/mlflow/routes/create

Request:
{
  "secret_id": "uuid-...",          // Reuse existing key
  "model_name": "gpt-4o-mini",
  "name": "Dev GPT-4o Mini",
  "bindings": [...]
}

Response:
{
  "route": {...},
  "bindings": [...]
}
```

#### Clone Route (Server-Side - addresses Yuki's request)
```
POST /ajax-api/3.0/mlflow/routes/{route_id}/clone

Request:
{
  "model_name": "gpt-4-turbo",      // Override model
  "name": "Cloned GPT-4 Turbo",
  "model_config": {
    "temperature": 0.9
  }
}

Response:
{
  "route": {
    "route_id": "new-uuid-...",
    "secret_id": "same-secret-id",  // Reuses same key!
    "model_name": "gpt-4-turbo",
    "name": "Cloned GPT-4 Turbo"
  }
}

Note: Clone operation happens server-side, so API key is never
exposed to client. User can clone routes using same key without
knowing the actual key value.
```

#### Update Route
```
PUT /ajax-api/3.0/mlflow/routes/{route_id}/update

Request:
{
  "model_name": "gpt-4-turbo",
  "name": "Updated Name",
  "model_config": {
    "temperature": 0.9
  }
}

Response:
{
  "route": {...}
}
```

#### Delete Route
```
DELETE /ajax-api/3.0/mlflow/routes/{route_id}

Response:
{
  "deleted_route_id": "uuid-...",
  "deleted_bindings": [...]         // Cascade delete
}
```

#### List Routes
```
GET /ajax-api/3.0/mlflow/routes/list?provider=openai

Response:
{
  "routes": [
    {
      "route_id": "uuid-...",
      "secret_id": "uuid-...",
      "model_name": "gpt-4o",
      "name": "Production",
      "secret": {                   // Joined data
        "provider": "openai",
        "is_shared": true
      }
    },
    ...
  ]
}
```

#### Get Route Details
```
GET /ajax-api/3.0/mlflow/routes/{route_id}

Response:
{
  "route": {...},
  "bindings": [...]
}
```

### Binding APIs

#### Create Binding
```
POST /ajax-api/3.0/mlflow/bindings/create

Request:
{
  "route_id": "uuid-...",
  "resource_type": "SCORER_JOB",
  "resource_id": "job-456",
  "field_name": "OPENAI_API_KEY"
}

Response:
{
  "binding": {...}
}
```

#### Delete Binding
```
DELETE /ajax-api/3.0/mlflow/bindings/{binding_id}

Response:
{
  "deleted_binding_id": "uuid-..."
}
```

#### List Bindings for Route
```
GET /ajax-api/3.0/mlflow/bindings/list?route_id=uuid-...

Response:
{
  "bindings": [...]
}
```

## Frontend Implementation

### UI Structure

#### Main Page: Two Tabs

**Routes Tab (Primary View - Model-Centric):**
```
┌─────────────────────────────────────────────────────────┐
│ AI Gateway                                              │
│ Manage AI model routes and API keys                    │
├─────────────────────────────────────────────────────────┤
│ [Routes] [API Keys]                                     │
├─────────────────────────────────────────────────────────┤
│ Routes                                 [+ Create Route] │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Name          Model        Provider    Bindings     │ │
│ ├─────────────────────────────────────────────────────┤ │
│ │ Production    gpt-4o       OpenAI      3 bindings   │ │
│ │ Development   gpt-4o-mini  OpenAI      1 binding    │ │
│ │ Judge         claude-4.5   Anthropic   5 bindings   │ │
│ │ Bedrock Prod  claude-4.5   AWS Bedrock 2 bindings   │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**API Keys Tab (Management View - Independent Key Management):**
```
┌─────────────────────────────────────────────────────────┐
│ API Keys                                [+ Add API Key] │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Provider      Created    Routes Using    Actions    │ │
│ ├─────────────────────────────────────────────────────┤ │
│ │ OpenAI        2 days     5 routes        [⚙️] [🗑️]  │ │
│ │ Anthropic     1 week     2 routes        [⚙️] [🗑️]  │ │
│ │ AWS Bedrock   3 days     8 routes        [⚙️] [🗑️]  │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Provider Configuration Schemas

```typescript
interface ProviderAuthField {
  name: string;
  label: string;
  type: 'text' | 'password' | 'url' | 'select' | 'textarea';
  required: boolean;
  placeholder?: string;
  options?: string[];
  helpText?: string;
  defaultValue?: string;
}

interface ProviderConfig {
  provider: string;
  displayName: string;
  authType: 'simple' | 'complex';
  fields?: ProviderAuthField[];
  models: {
    id: string;
    displayName: string;
  }[];
}

const PROVIDER_CONFIGS: ProviderConfig[] = [
  {
    provider: 'openai',
    displayName: 'OpenAI',
    authType: 'simple',
    models: [
      { id: 'gpt-5-turbo', displayName: 'GPT-5 Turbo' },
      { id: 'o4', displayName: 'o4' },
      { id: 'gpt-4o', displayName: 'GPT-4o' },
      { id: 'gpt-4o-mini', displayName: 'GPT-4o Mini' },
    ],
  },
  {
    provider: 'anthropic',
    displayName: 'Anthropic',
    authType: 'simple',
    models: [
      { id: 'claude-sonnet-4-5-20250929', displayName: 'Claude Sonnet 4.5 (Latest)' },
      { id: 'claude-opus-4-1-20250805', displayName: 'Claude Opus 4.1' },
      { id: 'claude-haiku-4-5-20251001', displayName: 'Claude Haiku 4.5' },
    ],
  },
  {
    provider: 'bedrock',
    displayName: 'AWS Bedrock',
    authType: 'complex',
    fields: [
      {
        name: 'aws_access_key_id',
        label: 'AWS Access Key ID',
        type: 'text',
        required: true,
        placeholder: 'AKIA...',
      },
      {
        name: 'aws_secret_access_key',
        label: 'AWS Secret Access Key',
        type: 'password',
        required: true,
      },
      {
        name: 'aws_region',
        label: 'AWS Region',
        type: 'select',
        required: true,
        options: ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'],
        defaultValue: 'us-east-1',
      },
      {
        name: 'aws_session_token',
        label: 'Session Token (Optional)',
        type: 'password',
        required: false,
        helpText: 'For temporary AWS credentials',
      },
    ],
    models: [
      { id: 'anthropic.claude-sonnet-4-5-20250929-v1:0', displayName: 'Claude Sonnet 4.5' },
      { id: 'anthropic.claude-opus-4-1-20250805-v1:0', displayName: 'Claude Opus 4.1' },
      { id: 'meta.llama3-3-70b-instruct-v1:0', displayName: 'Llama 3.3 70B' },
    ],
  },
  {
    provider: 'azure',
    displayName: 'Azure OpenAI',
    authType: 'complex',
    fields: [
      {
        name: 'api_key',
        label: 'API Key',
        type: 'password',
        required: true,
      },
      {
        name: 'azure_endpoint',
        label: 'Azure Endpoint (Base URL)',
        type: 'url',
        required: true,
        placeholder: 'https://my-resource.openai.azure.com',
        helpText: 'Your Azure OpenAI resource endpoint',
      },
      {
        name: 'api_version',
        label: 'API Version',
        type: 'text',
        required: true,
        placeholder: '2024-02-15-preview',
        defaultValue: '2024-02-15-preview',
      },
    ],
    models: [
      { id: 'gpt-4', displayName: 'GPT-4' },
      { id: 'gpt-4-turbo', displayName: 'GPT-4 Turbo' },
      { id: 'gpt-35-turbo', displayName: 'GPT-3.5 Turbo' },
    ],
  },
  {
    provider: 'vertex_ai',
    displayName: 'Google Vertex AI',
    authType: 'complex',
    fields: [
      {
        name: 'project_id',
        label: 'GCP Project ID',
        type: 'text',
        required: true,
        placeholder: 'my-gcp-project',
      },
      {
        name: 'location',
        label: 'Location',
        type: 'text',
        required: true,
        placeholder: 'us-central1',
        defaultValue: 'us-central1',
      },
      {
        name: 'service_account_json',
        label: 'Service Account Key (JSON)',
        type: 'textarea',
        required: true,
        helpText: 'Paste the entire JSON key file contents',
      },
    ],
    models: [
      { id: 'gemini-2.5-pro', displayName: 'Gemini 2.5 Pro' },
      { id: 'gemini-2.5-flash', displayName: 'Gemini 2.5 Flash' },
      { id: 'gemini-2.0-flash', displayName: 'Gemini 2.0 Flash' },
    ],
  },
  {
    provider: 'databricks',
    displayName: 'Databricks',
    authType: 'complex',
    fields: [
      {
        name: 'databricks_host',
        label: 'Databricks Host (Base URL)',
        type: 'url',
        required: true,
        placeholder: 'https://dbc-abc123.cloud.databricks.com',
        helpText: 'Your Databricks workspace URL',
      },
      {
        name: 'databricks_token',
        label: 'Personal Access Token',
        type: 'password',
        required: true,
      },
    ],
    models: [
      { id: 'databricks-claude-sonnet-4-5', displayName: 'Claude Sonnet 4.5' },
      { id: 'databricks-llama-4-maverick', displayName: 'Llama 4 Maverick' },
      { id: 'databricks-gemini-2.5-pro', displayName: 'Gemini 2.5 Pro' },
    ],
  },
];
```

## Security Considerations

### Encryption

1. **Field-Level Encryption:**
   - `secrets.secret_value` → Encrypted with KEK
   - `secrets.auth_config` → Encrypted with KEK
   - `routes.model_config` → Encrypted with KEK (contains potentially sensitive deployment names)

2. **Encryption Implementation:**
   ```python
   def encrypt_field(plaintext: str, kek: bytes) -> bytes:
       """Encrypt field using AES-256-GCM with KEK."""
       # Implementation uses same encryption as current secrets
   ```

3. **Decryption Access:**
   - Never return decrypted values in list operations
   - Only decrypt during create/update operations when user provides password
   - Audit all decryption operations

### Validation

1. **Provider Schema Validation:**
   ```python
   PROVIDER_AUTH_SCHEMAS = {
       'bedrock': {
           'required': ['aws_access_key_id', 'aws_secret_access_key', 'aws_region'],
           'optional': ['aws_session_token'],
       },
       'azure': {
           'required': ['api_key', 'azure_endpoint', 'api_version'],
       },
       # ...
   }

   def validate_auth_config(provider: str, auth_config: dict) -> None:
       """Validate auth_config matches provider schema."""
       schema = PROVIDER_AUTH_SCHEMAS.get(provider)
       if not schema:
           raise ValueError(f"Unknown provider: {provider}")

       # Check required fields
       for field in schema['required']:
           if field not in auth_config:
               raise ValueError(f"Missing required field: {field}")

       # Check no extra fields
       allowed = set(schema['required'] + schema['optional'])
       extra = set(auth_config.keys()) - allowed
       if extra:
           raise ValueError(f"Unexpected fields: {extra}")
   ```

2. **Input Sanitization:**
   - Sanitize all string inputs
   - Validate URLs (base URLs, endpoints)
   - Validate enum values (regions, API versions)
   - Validate JSON structure for `auth_config` and `model_config`

### Auditing

1. **Audit Log Events:**
   - `secret.created` - Log provider, is_shared
   - `secret.updated` - Log which routes affected
   - `secret.deleted` - Log cascade deletes
   - `secret.accessed` - Log decryption events
   - `route.created` - Log model, bindings
   - `route.cloned` - Log source route_id
   - `route.deleted` - Log bindings removed

2. **Sensitive Field Masking:**
   ```python
   def mask_secret_for_logs(secret: Secret) -> dict:
       """Return loggable representation with masked sensitive fields."""
       return {
           'secret_id': secret.secret_id,
           'provider': secret.provider,
           'is_shared': secret.is_shared,
           'secret_value': '***MASKED***',
           'auth_config': '***MASKED***',
       }
   ```

## Migration Strategy

### Database Migration Script

```python
def migrate_secrets_to_routes():
    """
    Migrate existing coupled secrets to separated architecture.

    Migration steps:
    1. For each existing secret with model field:
       a. Create new route with that model
       b. Update bindings to reference new route_id
    2. Clear model field from secrets table
    3. Add new columns: auth_config to secrets, model_config to routes
    """
    session = _get_sqlalchemy_store().session

    # Get all existing secrets with models
    existing_secrets = session.query(SqlSecret).filter(
        SqlSecret.model.isnot(None)
    ).all()

    print(f"Migrating {len(existing_secrets)} secrets...")

    for secret in existing_secrets:
        # Create route from secret's model
        route = SqlRoute(
            route_id=uuid.uuid4().hex,
            secret_id=secret.secret_id,
            model_name=secret.model,
            name=secret.secret_name,  # Use old secret name as route name
            created_at=secret.created_at,
            updated_at=secret.updated_at,
            created_by=secret.created_by,
        )
        session.add(route)

        # Update all bindings to reference new route
        bindings = session.query(SqlBinding).filter(
            SqlBinding.secret_id == secret.secret_id
        ).all()

        for binding in bindings:
            binding.route_id = route.route_id
            # Note: Don't delete secret_id column yet for rollback safety

        print(f"  Migrated secret {secret.secret_id} -> route {route.route_id} ({len(bindings)} bindings)")

    # Commit migration
    session.commit()
    print(f"✅ Migration complete!")

    # After confirming migration success, can drop model column from secrets
    # and secret_id column from bindings in a follow-up migration
```

## Implementation Phases

### Phase 1: Backend Schema & Data Model (Day 1)
- [ ] Update SQLAlchemy models (SqlSecret, SqlRoute, SqlBinding)
- [ ] Create database migration script
- [ ] Add JSON validation helpers
- [ ] Update proto definitions
- [ ] Regenerate proto files
- [ ] Add encryption/decryption for new JSON fields

### Phase 2: Backend API Implementation (Day 1-2)
- [ ] Implement Secret APIs (create, update, delete, list, get)
- [ ] Implement Route APIs (create, clone, update, delete, list, get)
- [ ] Implement Binding APIs (create, delete, list)
- [ ] Update handlers with provider validation
- [ ] Add cascade delete logic
- [ ] Update REST store client

### Phase 3: Frontend Core (Day 2)
- [ ] Update TypeScript types
- [ ] Create React hooks (useListRoutes, useListSecrets, useCreateRoute, etc.)
- [ ] Implement provider configs schema
- [ ] Create CreateRouteModal with provider-specific forms
- [ ] Update RoutesTable component

### Phase 4: Frontend Management (Day 2-3)
- [ ] Create API Keys tab/page
- [ ] Implement KeyDetailDrawer
- [ ] Implement RouteDetailDrawer
- [ ] Add clone functionality
- [ ] Update delete modals with cascade warnings
- [ ] Update UpdateApiKeyModal with affected routes warning

### Phase 5: Testing & Migration (Day 3)
- [ ] Run database migration on dev database
- [ ] Update test-secrets-data.py script
- [ ] Write unit tests for new store methods
- [ ] Write integration tests for API endpoints
- [ ] Write frontend component tests
- [ ] E2E test for full create/update/delete flows
- [ ] Test complex providers (Bedrock, Azure, Vertex)

### Phase 6: Polish & Documentation (Day 3)
- [ ] Add loading states and error handling
- [ ] Add empty states with helpful guidance
- [ ] Add tooltips and help text
- [ ] Update user documentation
- [ ] Create demo video/screenshots
- [ ] Code review and cleanup

## Success Criteria

- [ ] Users can create one API key and use it for multiple models (addresses "many models per key" requirement)
- [ ] Key rotation updates all routes automatically (addresses "punishingly stupid" key refresh problem)
- [ ] Complex providers (Bedrock, Azure, Vertex) are fully supported (addresses base URL needs)
- [ ] Clone feature works server-side without exposing keys (addresses Yuki's clone request)
- [ ] All cascade deletes show clear warnings
- [ ] Migration from old schema completes successfully
- [ ] Zero downtime deployment possible
- [ ] All tests pass
- [ ] Performance meets requirements (<200ms for list operations)
- [ ] Security review passes

## References

- **Original PR Discussion:** Slack thread with Yuki Watanabe (2025-01-07)
- **Key Requirements:**
  - Many models per key (common use case)
  - Key rotation as feature (not burden)
  - Model-centric UX with independent key management
  - Base URL support for Azure/Databricks
  - Clone functionality without key exposure
  - Separated concerns: "API Keys" and "Routes"

---

**Last Updated:** 2025-01-11
**Status:** Design Phase
**Implementer:** Ben Wilson (@benjamin.wilson)
**Reviewer:** Yuki Watanabe (@yuki.watanabe)

### Edit Functionality & UI

#### EditRouteModal (Inline Editing)

**Purpose:** Allow users to edit route configuration (model, name, runtime params) with instant commit to backend

```
┌────────────────────────────────────────┐
│ Edit Route                        [×]  │
├────────────────────────────────────────┤
│                                        │
│ Route Name                             │
│ ┌────────────────────────────────┐    │
│ │ Production GPT-4o              │    │
│ └────────────────────────────────┘    │
│                                        │
│ Provider: OpenAI (read-only)           │
│ API Key:  OpenAI Key (read-only)       │
│          [View Key Details →]          │
│                                        │
│ Model                                  │
│ ┌────────────────────────────────┐    │
│ │ gpt-4o                      ▼ │    │
│ └────────────────────────────────┘    │
│ • gpt-5-turbo                          │
│ • o4                                   │
│ • gpt-4o ✓                             │
│ • gpt-4o-mini                          │
│ • Custom model...                      │
│                                        │
│ ▼ Model Configuration                  │
│   Temperature: [0.7      ]             │
│   Max Tokens:  [4096     ]             │
│   Top P:       [0.9      ]             │
│   Timeout:     [30       ] seconds     │
│                                        │
│   ▼ Azure-Specific Settings            │
│     Deployment Name:                   │
│     ┌────────────────────────────┐    │
│     │ gpt-4o-prod-deployment     │    │
│     └────────────────────────────┘    │
│                                        │
│ ⓘ Changes will be saved immediately    │
│   and affect all resources using       │
│   this route.                          │
│                                        │
│         [Cancel]  [Save Changes]       │
└────────────────────────────────────────┘
```

**API Flow:**
```typescript
// When user clicks "Save Changes"
const handleSaveRoute = async () => {
  const payload = {
    route_id: currentRoute.route_id,
    model_name: selectedModel,
    name: routeName,
    model_config: {
      temperature: parseFloat(temperature),
      max_tokens: parseInt(maxTokens),
      top_p: parseFloat(topP),
      timeout: parseInt(timeout),
      // Provider-specific fields
      ...azureSettings
    },
    updated_by: currentUser.email
  };

  const response = await fetch(
    `/ajax-api/3.0/mlflow/routes/${currentRoute.route_id}/update`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }
  );

  if (response.ok) {
    // Show success toast
    toast.success('Route updated successfully');
    // Refresh routes list
    refetchRoutes();
    // Close modal
    onClose();
  }
};
```

**Form Validation:**
```typescript
const validate = () => {
  const errors = {};
  
  // Route name validation
  if (!routeName.trim()) {
    errors.routeName = 'Route name is required';
  }
  
  // Model validation
  if (!selectedModel) {
    errors.model = 'Model is required';
  }
  
  // Model config validation
  if (temperature < 0 || temperature > 2) {
    errors.temperature = 'Temperature must be between 0 and 2';
  }
  
  if (maxTokens < 1 || maxTokens > 128000) {
    errors.maxTokens = 'Max tokens must be between 1 and 128000';
  }
  
  return errors;
};
```

#### EditApiKeyModal (Key Rotation)

**Purpose:** Update API key credentials with warnings about affected routes

```
┌────────────────────────────────────────┐
│ Update API Key                    [×]  │
├────────────────────────────────────────┤
│                                        │
│ Provider: OpenAI (read-only)           │
│ Created:  2 days ago                   │
│                                        │
│ ⚠️  Warning: Impact Analysis           │
│                                        │
│ Updating this key will immediately     │
│ affect the following routes:           │
│                                        │
│ ┌────────────────────────────────┐    │
│ │ • Production GPT-4o (3)        │    │
│ │ • Development GPT-4o Mini (1)  │    │
│ │ • Testing GPT-4 Turbo (2)      │    │
│ │ • Staging GPT-4o (1)           │    │
│ │ • Backup GPT-4 (0)             │    │
│ │                                │    │
│ │ Total: 5 routes, 7 bindings    │    │
│ └────────────────────────────────┘    │
│                                        │
│ ⚠️  Running processes using these      │
│    routes will be affected.            │
│                                        │
│ ───────────────────────────────────    │
│                                        │
│ New API Key                            │
│ ┌────────────────────────────────┐    │
│ │ ••••••••••••••••••             │    │
│ └────────────────────────────────┘    │
│ [Show/Hide]                            │
│                                        │
│ ☐ I understand the impact and want    │
│   to proceed with this key rotation    │
│                                        │
│         [Cancel]  [Update Key]         │
│                  (disabled until ☑)    │
└────────────────────────────────────────┘
```

**API Flow:**
```typescript
const handleUpdateKey = async () => {
  if (!confirmationChecked) return;
  
  const payload = {
    secret_id: currentSecret.secret_id,
    secret_value: newApiKey,  // Or auth_config for complex providers
    updated_by: currentUser.email
  };

  const response = await fetch(
    `/ajax-api/3.0/mlflow/secrets/update`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }
  );

  if (response.ok) {
    const data = await response.json();
    // Show success with affected routes count
    toast.success(
      `API key updated successfully. ${data.affected_routes.length} routes updated.`
    );
    refetchSecrets();
    onClose();
  }
};
```

#### EditApiKeyModal - Complex Provider (Bedrock)

```
┌────────────────────────────────────────┐
│ Update API Key: AWS Bedrock       [×]  │
├────────────────────────────────────────┤
│                                        │
│ ⚠️  5 routes will be affected          │
│                                        │
│ AWS Access Key ID                      │
│ ┌────────────────────────────────┐    │
│ │ AKIA...                        │    │
│ └────────────────────────────────┘    │
│                                        │
│ AWS Secret Access Key                  │
│ ┌────────────────────────────────┐    │
│ │ ••••••••••••••••••             │    │
│ └────────────────────────────────┘    │
│ [Show/Hide]                            │
│                                        │
│ AWS Region                             │
│ ┌────────────────────────────────┐    │
│ │ us-east-1                   ▼ │    │
│ └────────────────────────────────┘    │
│                                        │
│ Session Token (Optional)               │
│ ┌────────────────────────────────┐    │
│ │ ••••••••••••••••••             │    │
│ └────────────────────────────────┘    │
│                                        │
│ ☐ I understand the impact              │
│                                        │
│         [Cancel]  [Update Key]         │
└────────────────────────────────────────┘
```

**Complex Provider Update Flow:**
```typescript
const handleUpdateComplexKey = async () => {
  const auth_config = {
    aws_access_key_id: awsAccessKeyId,
    aws_secret_access_key: awsSecretAccessKey,
    aws_region: awsRegion,
    aws_session_token: awsSessionToken || undefined
  };

  // Validate against provider schema
  const validationErrors = validateAuthConfig('bedrock', auth_config);
  if (validationErrors.length > 0) {
    setErrors(validationErrors);
    return;
  }

  const payload = {
    secret_id: currentSecret.secret_id,
    auth_config: JSON.stringify(auth_config),
    updated_by: currentUser.email
  };

  await fetch(`/ajax-api/3.0/mlflow/secrets/update`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
};
```

#### Inline Editing in Tables (Quick Edit)

**Routes Table with Inline Edit:**
```
┌─────────────────────────────────────────────────────────┐
│ Routes                                 [+ Create Route] │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Name          Model        Provider    Actions      │ │
│ ├─────────────────────────────────────────────────────┤ │
│ │ Production    gpt-4o       OpenAI      [📝] [📋] [🗑️]│ │
│ │ Development   gpt-4o-mini  OpenAI      [📝] [📋] [🗑️]│ │
│ │ Judge         claude-4.5   Anthropic   [📝] [📋] [🗑️]│ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘

Actions:
📝 = Edit Route (opens EditRouteModal)
📋 = Clone Route (server-side, opens pre-filled CreateRouteModal)
🗑️ = Delete Route (opens DeleteRouteModal with warnings)
```

**Hover Actions:**
```typescript
<TableRow onMouseEnter={() => setHoveredRow(route.route_id)}>
  <TableCell>{route.name}</TableCell>
  <TableCell>{route.model_name}</TableCell>
  <TableCell>{route.secret.provider}</TableCell>
  <TableCell>
    {hoveredRow === route.route_id && (
      <div css={{ display: 'flex', gap: theme.spacing.sm }}>
        <Tooltip content="Edit route">
          <IconButton
            icon={<EditIcon />}
            onClick={() => openEditModal(route)}
          />
        </Tooltip>
        <Tooltip content="Clone route">
          <IconButton
            icon={<CopyIcon />}
            onClick={() => handleCloneRoute(route.route_id)}
          />
        </Tooltip>
        <Tooltip content="Delete route">
          <IconButton
            icon={<TrashIcon />}
            onClick={() => openDeleteModal(route)}
            danger
          />
        </Tooltip>
      </div>
    )}
  </TableCell>
</TableRow>
```

#### Real-Time Validation & Auto-Save

**Debounced Auto-Save (Optional):**
```typescript
// Auto-save after 2 seconds of no changes
const [debouncedSave] = useDebouncedCallback(
  async (route: Route) => {
    await updateRoute(route);
    toast.success('Changes saved', { duration: 1000 });
  },
  2000
);

// In form onChange handlers
const handleFieldChange = (field: string, value: any) => {
  const updatedRoute = { ...currentRoute, [field]: value };
  setCurrentRoute(updatedRoute);
  debouncedSave(updatedRoute);
};
```

**Form State Management:**
```typescript
const EditRouteForm = ({ route, onSave, onCancel }) => {
  const [formData, setFormData] = useState({
    name: route.name,
    model_name: route.model_name,
    model_config: JSON.parse(route.model_config || '{}')
  });
  
  const [isDirty, setIsDirty] = useState(false);
  const [errors, setErrors] = useState({});
  const [isSaving, setIsSaving] = useState(false);

  // Track changes
  useEffect(() => {
    const hasChanges = JSON.stringify(formData) !== JSON.stringify({
      name: route.name,
      model_name: route.model_name,
      model_config: JSON.parse(route.model_config || '{}')
    });
    setIsDirty(hasChanges);
  }, [formData, route]);

  // Warn before closing with unsaved changes
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (isDirty) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isDirty]);

  const handleSubmit = async () => {
    setIsSaving(true);
    try {
      await onSave(formData);
    } catch (error) {
      setErrors(error.response?.data?.errors || { general: error.message });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Modal
      visible
      onCancel={() => {
        if (isDirty) {
          if (confirm('You have unsaved changes. Discard them?')) {
            onCancel();
          }
        } else {
          onCancel();
        }
      }}
      okButtonProps={{ disabled: !isDirty || isSaving, loading: isSaving }}
      onOk={handleSubmit}
    >
      {/* Form fields */}
    </Modal>
  );
};
```

#### Bulk Edit (Future Enhancement)

**Select Multiple Routes for Batch Update:**
```
┌─────────────────────────────────────────────────────────┐
│ Routes                  2 selected   [Bulk Actions ▼]  │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ ☑ Production    gpt-4o       OpenAI                 │ │
│ │ ☑ Development   gpt-4o-mini  OpenAI                 │ │
│ │ ☐ Judge         claude-4.5   Anthropic              │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘

Bulk Actions:
• Update Model Configuration (temperature, max_tokens, etc.)
• Change API Key (repoint to different key)
• Delete Selected Routes
• Export Configuration
```

#### Edit Workflow Summary

**User Journey:**
1. User clicks "Edit" icon on route row
2. EditRouteModal opens with current values pre-populated
3. User modifies fields (name, model, model_config)
4. Form validates in real-time (shows errors inline)
5. User clicks "Save Changes"
6. PUT request sent to `/ajax-api/3.0/mlflow/routes/{id}/update`
7. Backend validates, encrypts model_config, updates database
8. Success response triggers:
   - Success toast notification
   - Routes list refresh
   - Modal close
9. Updated route appears in table with new values

**Error Handling:**
```typescript
try {
  await updateRoute(payload);
  toast.success('Route updated successfully');
} catch (error) {
  if (error.status === 400) {
    // Validation error
    setErrors(error.data.errors);
  } else if (error.status === 404) {
    // Route not found
    toast.error('Route not found. It may have been deleted.');
    refetchRoutes();
    onClose();
  } else if (error.status === 409) {
    // Conflict (e.g., duplicate name)
    toast.error(error.data.message);
  } else {
    // Generic error
    toast.error('Failed to update route. Please try again.');
  }
}
```

#### Optimistic Updates

**For Better UX:**
```typescript
const { updateRoute } = useUpdateRouteMutation({
  onMutate: async (updatedRoute) => {
    // Cancel outgoing refetches
    await queryClient.cancelQueries(['routes']);
    
    // Snapshot previous value
    const previousRoutes = queryClient.getQueryData(['routes']);
    
    // Optimistically update cache
    queryClient.setQueryData(['routes'], (old) =>
      old.map((r) => r.route_id === updatedRoute.route_id ? updatedRoute : r)
    );
    
    // Return rollback function
    return { previousRoutes };
  },
  onError: (err, updatedRoute, context) => {
    // Rollback on error
    queryClient.setQueryData(['routes'], context.previousRoutes);
    toast.error('Failed to update route');
  },
  onSuccess: () => {
    toast.success('Route updated successfully');
  },
  onSettled: () => {
    // Refetch to ensure consistency
    queryClient.invalidateQueries(['routes']);
  }
});
```

