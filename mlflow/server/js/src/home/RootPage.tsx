import React from 'react';
import { useSearchParams } from '../common/utils/RoutingUtils';
import { shouldEnableWorkspaces } from '../common/utils/FeatureUtils';
import { extractWorkspaceFromSearchParams } from '../workspaces/utils/WorkspaceUtils';

const WorkspaceLandingPage = React.lazy(() => import('./WorkspaceLandingPage'));
const HomePage = React.lazy(() => import('./HomePage'));

/**
 * Root page component that conditionally renders based on workspace mode and query param.
 *
 * When workspaces are enabled:
 * - /?workspace=<name> → HomePage (workspace home)
 * - / (no workspace param) → WorkspaceLandingPage (workspace selector)
 *
 * When workspaces are disabled:
 * - / → HomePage
 */
export const RootPage = () => {
  const [searchParams] = useSearchParams();
  const workspacesEnabled = shouldEnableWorkspaces();

  // If not in workspace mode, always show HomePage
  if (!workspacesEnabled) {
    return <HomePage />;
  }

  // Check if we're inside a workspace context via query param
  const workspaceFromQuery = extractWorkspaceFromSearchParams(searchParams);

  // If inside a workspace, show the workspace's home page
  if (workspaceFromQuery) {
    return <HomePage />;
  }

  // Otherwise (no workspace param), show the workspace landing page (selector)
  return <WorkspaceLandingPage />;
};

export default RootPage;
