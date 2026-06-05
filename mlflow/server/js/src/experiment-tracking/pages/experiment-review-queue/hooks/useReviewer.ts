import { useCurrentUserQuery } from '../../../../account/hooks';

/** Reserved fallback reviewer on a bare no-auth server (no authenticated user). */
export const DEFAULT_REVIEWER = 'default';

/**
 * Canonical form of a user identifier, mirroring the server's
 * `normalize_user` (`mlflow/genai/review_queues/validation.py`): the server
 * lowercases + strips assigned users and USER-queue names at write time. Use
 * this for any client-side comparison against a stored queue `name` / user,
 * so a reviewer whose raw username carries casing or whitespace still matches.
 */
export const normalizeUser = (user: string): string => user.trim().toLowerCase();

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
