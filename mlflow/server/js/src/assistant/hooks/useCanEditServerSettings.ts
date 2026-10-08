import { useCurrentUserIsAdmin, useIsAuthAvailable } from '../../account/hooks';

/**
 * Whether the caller may change the Assistant's server-wide settings (project paths, skills, API
 * keys, full access). Mirrors the server: only from the server host, and on a server with auth,
 * only as an admin. While the current user is loading it returns false, so server-wide inputs are
 * never shown to a non-admin, even briefly.
 */
export const useCanEditServerSettings = (isLocalServer: boolean): boolean => {
  const isAuthAvailable = useIsAuthAvailable();
  const isAdmin = useCurrentUserIsAdmin();
  return isLocalServer && (!isAuthAvailable || isAdmin);
};
