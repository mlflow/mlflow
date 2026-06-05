import { useIsAuthAvailable, useMyPermissionsQuery } from '../../../../account/hooks';

/**
 * Whether the current user may MANAGE reviews for an experiment — create / edit
 * / delete queues and add / edit / delete questions. Management requires
 * experiment EDIT (or MANAGE); reviewing (answering assigned queues) only needs
 * READ and is not gated by this.
 *
 * This is a UX gate only — it hides controls the user can't use. The real
 * authorization is server-side (see [[project_review_queue_authz_model]],
 * planned as its own stack); until then these endpoints are ungated.
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
      (p.permission === 'EDIT' || p.permission === 'MANAGE'),
  );
};
