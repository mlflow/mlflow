import React from 'react';
import { useLocation } from '../common/utils/RoutingUtils';
import { shouldEnableWorkspaces } from '../common/utils/FeatureUtils';
import { extractWorkspaceFromPathname } from '../common/utils/WorkspaceUtils';

const WorkspaceLandingPage = React.lazy(() => import('./WorkspaceLandingPage'));
const HomePage = React.lazy(() => import('./HomePage'));

/**
 * Root page component that conditionally renders based on workspace mode and path.
 *
 * When workspaces are enabled:
 * - / (root) → WorkspaceLandingPage (workspace selector)
 * - /workspaces/:workspaceName → HomePage (workspace home)
 *
 * When workspaces are disabled:
 * - / → HomePage
 */
export const RootPage = () => {
  const location = useLocation();
  const workspacesEnabled = shouldEnableWorkspaces();

  // If not in workspace mode, always show HomePage
  if (!workspacesEnabled) {
    return <HomePage />;
  }

  // Check if we're inside a workspace context (e.g., /workspaces/default)
  const workspaceFromPath = extractWorkspaceFromPathname(location.pathname);

  // If inside a workspace, show the workspace's home page
  if (workspaceFromPath) {
    return <HomePage />;
  }

  // Otherwise (at root /), show the workspace landing page (selector)
  return <WorkspaceLandingPage />;
};

export default RootPage;
