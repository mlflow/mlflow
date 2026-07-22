import { useIsAuthAvailable, useCurrentUserQuery } from '../../account/hooks';
import type { MCPServer } from '../types';
import { getServerPermissions, isServerDimmed } from '../utils';

export const useServerState = (server?: MCPServer) => {
  const isAuthAvailable = useIsAuthAvailable();
  const { isLoading: isAuthLoading } = useCurrentUserQuery();
  const { canUpdate, canDelete, canManage } = getServerPermissions(server);
  const hasLatestEndpoint = server?.latest_version
    ? (server.access_endpoints ?? []).some((e) => e.resolved_version?.version === server.latest_version)
    : (server?.access_endpoints?.length ?? 0) > 0;

  return {
    canUpdate,
    canDelete,
    canManage,
    isDimmed: isAuthAvailable && !isAuthLoading && (server ? isServerDimmed(server) : false),
    isUnavailable: !hasLatestEndpoint,
    showVisibilityControls: isAuthAvailable && canUpdate,
    isAuthAvailable,
  };
};
