/**
 * Model Configuration Components
 *
 * A unified component system for configuring Provider -> Model -> API Key.
 *
 * Usage:
 *
 * // Simple usage with ModelConfigurationSection (handles data fetching internally)
 * import { ModelConfigurationSection, createEmptyModelConfiguration } from './model-configuration';
 *
 * const [config, setConfig] = useState(createEmptyModelConfiguration());
 * <ModelConfigurationSection value={config} onChange={setConfig} />
 *
 * // Advanced usage with individual components
 * import {
 *   ModelConfigurationRenderer,
 *   useApiKeyConfiguration,
 *   useModelConfigurationState,
 * } from './model-configuration';
 */

// Main container component
export { ModelConfigurationSection } from './ModelConfigurationSection';

// Presentation components
export { ModelConfigurationRenderer } from './components/ModelConfigurationRenderer';
export { ProviderModelSelector, ModelCapabilitiesTags } from './components/ProviderModelSelector';
export { ApiKeyConfigurator } from './components/ApiKeyConfigurator';

// Hooks
export { useApiKeyConfiguration } from './hooks/useApiKeyConfiguration';
export { useModelConfigurationState, createModelConfigurationActions } from './hooks/useModelConfigurationState';

// Utilities
export {
  createEmptyModelConfiguration,
  createEmptyApiKeyConfiguration,
  createEmptyNewSecret,
  resetConfigurationForProvider,
  resetApiKeyConfiguration,
  isModelConfigurationComplete,
  isApiKeyConfigurationComplete,
  validateModelConfiguration,
  hasValidationErrors,
  transformNewSecretForApi,
  generateConfigurationId,
} from './utils/modelConfigurationUtils';

// Types
export type {
  SecretMode,
  ModelConfiguration,
  ApiKeyConfiguration,
  NewSecretData,
  ModelConfigurationErrors,
  ModelConfigurationProps,
  ModelConfigurationState,
  ModelConfigurationActions,
} from './types';
