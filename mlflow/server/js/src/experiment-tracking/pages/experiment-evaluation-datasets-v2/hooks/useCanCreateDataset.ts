import { NotFoundError } from '@databricks/web-shared/errors';
import { AccountApi } from '../../../../account/api';
import { useCurrentUserQuery } from '../../../../account/hooks';
import { useQuery } from '../../../../common/utils/reactQueryHooks';
import { useActiveWorkspace } from '../../../../workspaces/utils/WorkspaceUtils';

export const useCanCreateDataset = (experimentId: string): boolean => {
  const user = useCurrentUserQuery();
  const username = user.data?.user?.username;
  const workspace = useActiveWorkspace();
  const permission = useQuery({
    queryKey: ['dataset-create-permission', username, workspace, experimentId],
    queryFn: () => AccountApi.getUserPermission(username ?? '', 'experiment', experimentId),
    enabled: Boolean(username && experimentId),
    retry: false,
    refetchOnWindowFocus: false,
  });

  // Only the missing auth endpoint identifies a no-auth server; lookup failures must not grant access.
  return (
    user.error instanceof NotFoundError ||
    (!user.isError && permission.isSuccess && ['EDIT', 'MANAGE'].includes(permission.data.permission))
  );
};
