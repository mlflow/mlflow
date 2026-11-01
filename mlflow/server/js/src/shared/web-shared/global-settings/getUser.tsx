import { useEffect } from 'react';

import { NetworkRequestError } from '@databricks/web-shared/errors';
import { useCurrentUserQuery } from '@mlflow/mlflow/src/account/hooks';

// The placeholder is used only when the server has no current-user endpoint.
const DEFAULT_USER = 'User';

const CURRENT_USER_RETRY_DELAY_MS = 2000;

export const getUser = () => DEFAULT_USER;

export const useUserIdentity = () => {
  // Credentials can change in another tab, so check the identity again when the window gets focus.
  const { data, error, errorUpdatedAt, isFetching, refetch } = useCurrentUserQuery({ refetchOnWindowFocus: true });
  const isAuthUnavailable = error instanceof NetworkRequestError && error.status === 404;

  useEffect(() => {
    if (!error || isAuthUnavailable) {
      return undefined;
    }
    const timer = setTimeout(() => void refetch(), CURRENT_USER_RETRY_DELAY_MS);
    return () => clearTimeout(timer);
  }, [error, errorUpdatedAt, isAuthUnavailable, refetch]);

  const user = data?.user?.username ?? (isAuthUnavailable ? DEFAULT_USER : '');
  // During a refetch, or after a refetch fails, the cache can still hold the previous user. Writes wait until the
  // identity is confirmed, but `user` stays set so views keyed on it do not reset.
  const isResolved = Boolean(user) && !isFetching && (!error || isAuthUnavailable);

  return { user, isResolved };
};
