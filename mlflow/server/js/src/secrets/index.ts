/**
 * MLflow Secrets and Endpoints Management - Public API
 *
 * This module provides reusable components, hooks, and utilities for managing
 * secrets and endpoints throughout the MLflow application.
 *
 * ## Usage Examples:
 *
 * ### Creating an endpoint from scorer creation:
 * ```typescript
 * import { CreateRouteModal, useCreateEndpoint } from '@mlflow/mlflow/src/secrets';
 *
 * function ScorerCreation() {
 *   const { createEndpointAsync } = useCreateEndpoint();
 *   const [showEndpointModal, setShowEndpointModal] = useState(false);
 *
 *   const handleCreateEndpoint = async (endpointData) => {
 *     const endpoint = await createEndpointAsync(endpointData);
 *     // Use endpoint.endpoint_id for scorer
 *   };
 *
 *   return (
 *     <>
 *       <Button onClick={() => setShowEndpointModal(true)}>Create New Endpoint</Button>
 *       <CreateRouteModal
 *         visible={showEndpointModal}
 *         onCancel={() => setShowEndpointModal(false)}
 *         onCreate={handleCreateEndpoint}
 *       />
 *     </>
 *   );
 * }
 * ```
 *
 * ### Using modal hooks for cleaner code:
 * ```typescript
 * import { useCreateSecretModal } from '@mlflow/mlflow/src/secrets';
 *
 * function MyComponent() {
 *   const { CreateSecretModal, openModal } = useCreateSecretModal({
 *     onSuccess: () => console.log('Secret created!'),
 *   });
 *
 *   return (
 *     <>
 *       <Button onClick={openModal}>Create Secret</Button>
 *       {CreateSecretModal}
 *     </>
 *   );
 * }
 * ```
 */

// Re-export everything from submodules
export * from './hooks';
export * from './components';
export * from './types';
export * from './constants';
