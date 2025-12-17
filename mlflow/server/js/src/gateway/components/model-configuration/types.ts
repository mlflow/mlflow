/**
 * Types for the unified Model Configuration component.
 *
 * A ModelConfiguration represents the complete setup for connecting to an LLM:
 * Provider -> Model -> API Key (authentication)
 *
 * This structure is designed to:
 * 1. Make the dependency chain clear (API key is tied to the model, not the endpoint)
 * 2. Support multiple configurations per endpoint in the future
 * 3. Separate data concerns from rendering concerns
 */

/**
 * Data for creating a new API key/secret
 */
export interface NewSecretData {
  name: string;
  authMode: string;
  secretFields: Record<string, string>;
  configFields: Record<string, string>;
}

/**
 * API key configuration - either use existing or create new
 */
export interface ApiKeyConfiguration {
  mode: 'new' | 'existing';
  existingSecretId: string;
  newSecret: NewSecretData;
}

/**
 * Complete model configuration combining provider, model, and API key
 */
export interface ModelConfiguration {
  /** Unique identifier for this configuration (for React keys when multiple) */
  id: string;
  /** LLM provider (e.g., 'openai', 'anthropic') */
  provider: string;
  /** Provider-specific model name (e.g., 'gpt-4o', 'claude-3-5-sonnet') */
  modelName: string;
  /** API key/authentication configuration */
  apiKey: ApiKeyConfiguration;
}

/**
 * Validation errors for a model configuration
 */
export interface ModelConfigurationErrors {
  provider?: string;
  modelName?: string;
  apiKey?: {
    mode?: string;
    existingSecretId?: string;
    newSecret?: {
      name?: string;
      secretFields?: Record<string, string>;
      configFields?: Record<string, string>;
    };
  };
}

/**
 * Props for controlled ModelConfiguration components
 */
export interface ModelConfigurationProps {
  /** Current configuration value */
  value: ModelConfiguration;
  /** Callback when configuration changes */
  onChange: (value: ModelConfiguration) => void;
  /** Validation errors to display */
  errors?: ModelConfigurationErrors;
  /** Whether the component is disabled */
  disabled?: boolean;
  /** Component ID prefix for telemetry */
  componentIdPrefix?: string;
}

/**
 * State returned by useModelConfigurationState hook
 */
export interface ModelConfigurationState {
  /** Current configuration value */
  value: ModelConfiguration;
  /** Whether the configuration is complete and valid */
  isComplete: boolean;
  /** Validation errors */
  errors: ModelConfigurationErrors;
}

/**
 * Actions returned by useModelConfigurationState hook
 */
export interface ModelConfigurationActions {
  /** Update the entire configuration */
  setValue: (value: ModelConfiguration) => void;
  /** Update just the provider (resets model and API key) */
  setProvider: (provider: string) => void;
  /** Update just the model name */
  setModelName: (modelName: string) => void;
  /** Update the API key configuration */
  setApiKey: (apiKey: ApiKeyConfiguration) => void;
  /** Reset to empty configuration */
  reset: () => void;
}
