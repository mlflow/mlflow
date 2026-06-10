import type { IntlShape } from 'react-intl';

import { useCurrentUserQuery } from '../../../../account/hooks';

/** Reserved fallback reviewer on a bare no-auth server (no authenticated user). */
export const DEFAULT_REVIEWER = 'default';

/**
 * Display label for a user identity. The reserved no-auth `default` user
 * queue is shown as "Default"; real usernames are shown as-is.
 */
export const displayUser = (user: string, intl: IntlShape): string =>
  user === DEFAULT_REVIEWER
    ? intl.formatMessage({
        defaultMessage: 'Default',
        description: 'Display name shown for the reserved no-auth default review user',
      })
    : user;

/**
 * The current reviewer's identity. Authenticated deployments (basic-auth /
 * account plugin) resolve the real username; a bare no-auth server has no
 * `/users/current`, so the query misses and we fall back to the reserved
 * default user. Used to scope "my queues" and to stamp `completed_by` /
 * `created_by`.
 */
export const useReviewer = (): string => {
  const { data: currentUser } = useCurrentUserQuery();
  return currentUser?.user?.username || DEFAULT_REVIEWER;
};
