/**
 * Public API for secrets and endpoints management hooks
 *
 * These hooks can be used from any part of the application that needs to
 * interact with secrets or endpoints (e.g., scorer creation, model deployment, etc.)
 */

// Endpoint management hooks
export { useCreateEndpoint } from './useCreateEndpoint';
export { useUpdateEndpoint } from './useUpdateEndpoint';
export { useDeleteEndpointMutation } from './useDeleteEndpointMutation';
export { useListEndpoints } from './useListEndpoints';

// Secret management hooks
export { useCreateSecretMutation } from './useCreateSecretMutation';
export { useUpdateSecretMutation } from './useUpdateSecretMutation';
export { useDeleteSecretMutation } from './useDeleteSecretMutation';
export { useListSecrets } from './useListSecrets';

// Secret binding hooks
export { useBindSecretMutation } from './useBindSecretMutation';
export { useUnbindSecretMutation } from './useUnbindSecretMutation';
export { useListBindings } from './useListBindings';

// Modal hooks (for simple modal patterns)
export { useCreateSecretModal } from './modals/useCreateSecretModal';
export type { UseCreateSecretModalProps } from './modals/useCreateSecretModal';

export { useUpdateSecretModal } from './modals/useUpdateSecretModal';
export type { UseUpdateSecretModalProps } from './modals/useUpdateSecretModal';

export { useDeleteSecretModal } from './modals/useDeleteSecretModal';
export type { UseDeleteSecretModalProps } from './modals/useDeleteSecretModal';

// Utility hooks
export { useRoutesTagsFilter } from './useRoutesTagsFilter';
