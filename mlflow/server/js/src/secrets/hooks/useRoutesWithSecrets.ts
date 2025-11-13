import { useMemo } from 'react';
import { useListRoutes } from './useListRoutes';
import { useListSecrets } from './useListSecrets';
import { useListBindings } from './useListBindings';
import type { RouteWithSecret } from '../components/RoutesTable.utils';

export const useRoutesWithSecrets = ({ enabled = true }: { enabled?: boolean } = {}) => {
  const { routes = [], isLoading: isLoadingRoutes, error: routesError } = useListRoutes({ enabled });
  const { secrets = [], isLoading: isLoadingSecrets, error: secretsError } = useListSecrets({ enabled });
  const { bindings = [], isLoading: isLoadingBindings, error: bindingsError } = useListBindings({
    enabled,
  });

  const routesWithSecrets = useMemo<RouteWithSecret[]>(() => {
    if (!routes.length || !secrets.length) return [];

    return routes.map((route) => {
      const secret = secrets.find((s) => s.secret_id === route.secret_id);
      const routeBindings = bindings.filter((b) => b.route_id === route.route_id);

      return {
        ...route,
        secret_name: secret?.secret_name,
        masked_value: secret?.masked_value,
        provider: secret?.provider,
        bindings: routeBindings,
      };
    });
  }, [routes, secrets, bindings]);

  return {
    routes: routesWithSecrets,
    isLoading: isLoadingRoutes || isLoadingSecrets || isLoadingBindings,
    error: routesError || secretsError || bindingsError,
  };
};
