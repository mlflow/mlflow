import { useIsAuthAvailable, useCurrentUserQuery } from '../../account/hooks';
import type { MCPServer } from '../types';
import { findLatestEndpoint, getServerPermissions, isServerDimmed } from '../utils';

export const useServerState = (server?: MCPServer) => {
  const isAuthAvailable = useIsAuthAvailable();
  const { isLoading: isAuthLoading } = useCurrentUserQuery();
  const { canUpdate, canDelete, canManage } = getServerPermissions(server);
  const hasLatestEndpoint = server ? findLatestEndpoint(server) !== undefined : false;

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
