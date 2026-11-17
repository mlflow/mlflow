import { useExperimentPermission } from '../../../hooks/useExperimentPermission';

/** MANAGE reviews: create/edit/delete queues and questions. Requires experiment MANAGE. */
export const useCanManageReviews = (experimentId: string): boolean => useExperimentPermission(experimentId, ['MANAGE']);

/** EDIT reviews: create/own queues, route traces, review. Requires experiment EDIT or MANAGE. */
export const useCanEditReviews = (experimentId: string): boolean =>
  useExperimentPermission(experimentId, ['EDIT', 'MANAGE']);
