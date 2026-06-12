import { useIsAuthAvailable, useMyPermissionsQuery } from '../../../../account/hooks';

/**
 * Whether the current user holds one of `levels` on the experiment.
 *
 * These are UX gates that hide controls the user can't use; the real
 * authorization is enforced server-side by the basic-auth plugin. On a no-auth
 * server there's no permission model, so everyone is effectively the admin.
 * While auth is on but permissions haven't loaded we return false — better to
 * reveal a control late than expose one that 403s server-side.
 */
const useExperimentReviewPermission = (experimentId: string, levels: string[]): boolean => {
  const authAvailable = useIsAuthAvailable();
  const { data } = useMyPermissionsQuery();

  if (!authAvailable) {
    return true;
  }
  if (!data) {
    return false;
  }
  if (data.is_admin) {
    return true;
  }
  return (data.permissions ?? []).some(
    (p) =>
      p.resource_type === 'experiment' &&
      (p.resource_pattern === experimentId || p.resource_pattern === '*') &&
      levels.includes(p.permission),
  );
};

/**
 * Whether the current user may MANAGE reviews for an experiment — create / edit
 * / delete queues and add / edit / delete questions. Requires experiment MANAGE.
 */
export const useCanManageReviews = (experimentId: string): boolean =>
  useExperimentReviewPermission(experimentId, ['MANAGE']);

/**
 * Whether the current user may make EDIT-level review changes — create and own
 * queues, route traces, and (combined with queue ownership) manage the queues
 * they created. Requires experiment EDIT or MANAGE.
 */
export const useCanEditReviews = (experimentId: string): boolean =>
  useExperimentReviewPermission(experimentId, ['EDIT', 'MANAGE']);
