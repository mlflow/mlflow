/**
 * Whether the current user can persist custom views on an experiment (save /
 * update via `setExperimentTag`).
 *
 * Databricks gates this on an experiment ACL check (`authorize/CAN_EDIT`). OSS
 * MLflow is single-tenant and has no per-experiment write-permission system for
 * tags today, so every editor can persist custom views. Kept as a hook with the
 * same shape as the Databricks version so it stays a one-line swap if OSS ever
 * adds workspace/permission-aware editing.
 */
export const useCanEditExperimentCustomViews = (
  // eslint-disable-next-line @typescript-eslint/no-unused-vars -- kept for interface parity with the Databricks-backed hook.
  experimentId?: string,
): { canEdit: boolean; isLoading: boolean } => ({ canEdit: true, isLoading: false });
