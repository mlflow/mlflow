import { useIsAuthAvailable, useMyPermissionsQuery } from '../../../../account/hooks';

/**
 * Whether the current user may MANAGE reviews for an experiment — create / edit
 * / delete queues and add / edit / delete questions. Management requires
 * experiment MANAGE; reviewing (answering assigned queues) needs experiment EDIT
 * plus queue membership and is not gated by this.
 *
 * This is a UX gate that hides controls the user can't use; the real
 * authorization is enforced server-side by the basic-auth plugin.
 *
 * On a no-auth server there's no permission model, so everyone is effectively
 * the admin and management is always available.
 */
export const useCanManageReviews = (experimentId: string): boolean => {
  const authAvailable = useIsAuthAvailable();
  const { data } = useMyPermissionsQuery();

  if (!authAvailable) {
    return true;
  }
  if (!data) {
    // Auth is on but permissions haven't loaded — hide management until we know
    // (better to reveal a control late than expose one that 403s server-side).
    return false;
  }
  if (data.is_admin) {
    return true;
  }
  return (data.permissions ?? []).some(
    (p) =>
      p.resource_type === 'experiment' &&
      (p.resource_pattern === experimentId || p.resource_pattern === '*') &&
      p.permission === 'MANAGE',
  );
};
