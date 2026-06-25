import { useIsAuthAvailable, useMyPermissionsQuery } from '../../../../account/hooks';
import { useActiveWorkspace } from '../../../../workspaces/utils/WorkspaceUtils';
import { useWorkspacesEnabled } from '../../../hooks/useServerInfo';

/**
 * Whether the current user holds one of `levels` on the experiment. A UX gate
 * only — the server enforces the real authorization. No-auth: everyone passes;
 * while permissions load: deny (reveal late rather than expose a control that 403s).
 */
const useExperimentReviewPermission = (experimentId: string, levels: string[]): boolean => {
  const authAvailable = useIsAuthAvailable();
  const { data } = useMyPermissionsQuery();
  const { workspacesEnabled, loading: workspacesLoading } = useWorkspacesEnabled();
  const activeWorkspace = useActiveWorkspace();

  if (!authAvailable) {
    return true;
  }
  if (!data) {
    return false;
  }
  if (data.is_admin) {
    return true;
  }
  // The permissions endpoint returns grants across every workspace, and a `*`
  // grant is workspace-relative, so only count grants in the workspace being
  // viewed. Skip that filter only when workspaces are *definitively* disabled
  // (single-workspace deployment, nothing to leak). While the flag is loading or
  // the workspace hasn't resolved yet, require the match and deny rather than
  // over-reveal a control that would 403 server-side.
  const workspacesDefinitelyDisabled = !workspacesEnabled && !workspacesLoading;
  return (data.permissions ?? []).some(
    (p) =>
      p.resource_type === 'experiment' &&
      (p.resource_pattern === experimentId || p.resource_pattern === '*') &&
      levels.includes(p.permission) &&
      (workspacesDefinitelyDisabled || p.workspace === activeWorkspace),
  );
};

/** MANAGE reviews: create/edit/delete queues and questions. Requires experiment MANAGE. */
export const useCanManageReviews = (experimentId: string): boolean =>
  useExperimentReviewPermission(experimentId, ['MANAGE']);

/** EDIT reviews: create/own queues, route traces, review. Requires experiment EDIT or MANAGE. */
export const useCanEditReviews = (experimentId: string): boolean =>
  useExperimentReviewPermission(experimentId, ['EDIT', 'MANAGE']);
