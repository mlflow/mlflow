import { useIsAuthAvailable, useMyPermissionsQuery } from '../../../../account/hooks';

/**
 * Whether the current user holds one of `levels` on the experiment. A UX gate
 * only — the server enforces the real authorization. No-auth: everyone passes;
 * while permissions load: deny (reveal late rather than expose a control that 403s).
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

/** MANAGE reviews: create/edit/delete queues and questions. Requires experiment MANAGE. */
export const useCanManageReviews = (experimentId: string): boolean =>
  useExperimentReviewPermission(experimentId, ['MANAGE']);

/** EDIT reviews: create/own queues, route traces, review. Requires experiment EDIT or MANAGE. */
export const useCanEditReviews = (experimentId: string): boolean =>
  useExperimentReviewPermission(experimentId, ['EDIT', 'MANAGE']);
