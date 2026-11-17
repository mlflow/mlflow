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

  // A missing auth route returns HTML; an unknown authenticated user can return a JSON 404.
  const authRouteMissing =
    user.error instanceof NotFoundError &&
    user.error.response?.headers.get('content-type')?.split(';')[0].trim().toLowerCase() === 'text/html';
  return (
    workspaceResolved &&
    (authRouteMissing ||
      Boolean(!user.isError && (isAdmin || (permission.isSuccess && levels.includes(permission.data.permission)))))
  );
};
