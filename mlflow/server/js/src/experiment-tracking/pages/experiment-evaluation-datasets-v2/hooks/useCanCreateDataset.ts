import { NotFoundError } from '@databricks/web-shared/errors';
import { AccountApi } from '../../../../account/api';
import { useCurrentUserQuery } from '../../../../account/hooks';
import { useQuery } from '../../../../common/utils/reactQueryHooks';
import { useActiveWorkspace } from '../../../../workspaces/utils/WorkspaceUtils';
import { useWorkspacesEnabled } from '../../../hooks/useServerInfo';

export const useCanCreateDataset = (experimentId: string): boolean => {
  const user = useCurrentUserQuery();
  const username = user.data?.user?.username;
  const isAdmin = Boolean(!user.isError && user.data?.user?.is_admin);
  const workspace = useActiveWorkspace();
  const { workspacesEnabled, loading } = useWorkspacesEnabled();
  const workspaceResolved = (!workspacesEnabled && !loading) || Boolean(workspace);
  const permission = useQuery({
    queryKey: ['dataset-create-permission', username, workspace, experimentId],
    queryFn: () => AccountApi.getUserPermission(username ?? '', 'experiment', experimentId),
    enabled: Boolean(username && experimentId && workspaceResolved && !isAdmin),
    retry: false,
    refetchOnWindowFocus: false,
  });

  // Missing auth routes return HTML; an auth-enabled missing user returns a JSON 404.
  return (
    workspaceResolved &&
    ((user.error instanceof NotFoundError &&
      user.error.response?.headers.get('content-type')?.split(';')[0].trim().toLowerCase() === 'text/html') ||
      (!user.isError && (isAdmin || (permission.isSuccess && ['EDIT', 'MANAGE'].includes(permission.data.permission)))))
  );
};
