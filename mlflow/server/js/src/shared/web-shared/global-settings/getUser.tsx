import { NetworkRequestError } from '@databricks/web-shared/errors';
import { useCurrentUserQuery } from '@mlflow/mlflow/src/account/hooks';

// The placeholder is used only when the server has no current-user endpoint.
const DEFAULT_USER = 'User';

export const getUser = () => DEFAULT_USER;

export const useUserIdentity = () => {
  const { data, error } = useCurrentUserQuery();
  const isAuthUnavailable = error instanceof NetworkRequestError && error.status === 404;
  const user = data?.user?.username ?? (isAuthUnavailable ? DEFAULT_USER : undefined);

  return { user: user ?? '', isResolved: Boolean(user) };
};
