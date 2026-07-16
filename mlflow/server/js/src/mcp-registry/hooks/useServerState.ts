import { useIsAuthAvailable } from '../../account/hooks';
import type { MCPServer } from '../types';
import { getServerPermissions, isServerDimmed } from '../utils';

export const useServerState = (server?: MCPServer) => {
  const isAuthAvailable = useIsAuthAvailable();
  const { canUpdate, canDelete, canManage } = getServerPermissions(server);
  const hasBindings = (server?.access_bindings?.length ?? 0) > 0;

  return {
    canUpdate,
    canDelete,
    canManage,
    isDimmed: isAuthAvailable && (server ? isServerDimmed(server) : false),
    isUnavailable: !hasBindings,
    showVisibilityControls: isAuthAvailable && canManage,
    isAuthAvailable,
  };
};
