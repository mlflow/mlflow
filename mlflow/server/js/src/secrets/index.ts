/**
 * MLflow Secrets and Routes Management - Public API
 *
 * This module provides reusable components, hooks, and utilities for managing
 * secrets and routes throughout the MLflow application.
 *
 * ## Usage Examples:
 *
 * ### Creating a route from scorer creation:
 * ```typescript
 * import { CreateRouteModal, useCreateRoute } from '@mlflow/mlflow/src/secrets';
 *
 * function ScorerCreation() {
 *   const { createRouteAsync } = useCreateRoute();
 *   const [showRouteModal, setShowRouteModal] = useState(false);
 *
 *   const handleCreateRoute = async (routeData) => {
 *     const route = await createRouteAsync(routeData);
 *     // Use route.route_id for scorer
 *   };
 *
 *   return (
 *     <>
 *       <Button onClick={() => setShowRouteModal(true)}>Create New Route</Button>
 *       <CreateRouteModal
 *         visible={showRouteModal}
 *         onCancel={() => setShowRouteModal(false)}
 *         onCreate={handleCreateRoute}
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
