import { NotFoundError } from '@databricks/web-shared/errors';
import { useQuery } from '@databricks/web-shared/query-client';
import { AccountApi } from '../../account/api';
import { useCurrentUserQuery } from '../../account/hooks';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { useWorkspacesEnabled } from './useServerInfo';

/** A UX gate; the server resolves effective grants and enforces authorization. */
export const useExperimentPermission = (experimentId: string, levels: string[]): boolean => {
  const user = useCurrentUserQuery();
  const username = user.data?.user?.username;
  const isAdmin = Boolean(!user.isError && user.data?.user?.is_admin);
  const workspace = useActiveWorkspace();
  const { workspacesEnabled, loading } = useWorkspacesEnabled();
  const workspaceResolved = (!workspacesEnabled && !loading) || Boolean(workspace);
  const permission = useQuery({
    queryKey: ['experiment-permission', username, workspace, experimentId],
    queryFn: () => AccountApi.getUserPermission(username ?? '', 'experiment', experimentId),
    enabled: Boolean(!isAdmin && username && experimentId && workspaceResolved),
    retry: false,
    refetchOnWindowFocus: false,
  });

  // Only a missing auth endpoint identifies no-auth; lookup failures must never grant access.
  return (
    user.error instanceof NotFoundError ||
    Boolean(
      !user.isError &&
      workspaceResolved &&
      (isAdmin || (permission.isSuccess && levels.includes(permission.data.permission))),
    )
  );
};
