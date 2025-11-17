/**
 * Public API for secrets and routes UI components
 *
 * These components can be used from any part of the application that needs
 * route or secret management UI (e.g., scorer creation, model deployment, etc.)
 */

// Route modals - for creating and managing routes from external contexts
export { CreateRouteModal } from './CreateRouteModal';
export { AddRouteModal } from './AddRouteModal';
export { UpdateRouteModal } from './UpdateRouteModal';

// Secret modals - use the hooks versions instead for better encapsulation
// These are exported for legacy compatibility
export { CreateSecretModal } from './CreateSecretModal';
export { UpdateSecretModal } from './UpdateSecretModal';
export { DeleteSecretModal } from './DeleteSecretModal';

// Utility components
export { MaskedApiKeyInput } from './MaskedApiKeyInput';
export { AuthConfigFields } from './AuthConfigFields';
export { ProviderBadge } from './ProviderBadge';
export { SecretBindingsList } from './SecretBindingsList';

// Route and provider constants
export { PROVIDERS, type Model, type Provider } from './routeConstants';
