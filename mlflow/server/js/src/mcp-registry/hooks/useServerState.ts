import { useIsAuthAvailable } from '../../account/hooks';
import type { MCPServer } from '../types';
import { getServerPermissions, isServerDimmed } from '../utils';

export const useServerState = (server?: MCPServer) => {
  const isAuthAvailable = useIsAuthAvailable();
  const { canUpdate, canDelete, canManage } = getServerPermissions(server);

  return {
    canUpdate,
    canDelete,
    canManage,
    isDimmed: !!server && isServerDimmed(server),
    showVisibilityControls: isAuthAvailable && canUpdate,
    isAuthAvailable,
  };
};
