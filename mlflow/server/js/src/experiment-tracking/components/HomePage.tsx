import { useState, useEffect } from 'react';
import { Navigate, useLocation } from '../../common/utils/RoutingUtils';
import Routes from '../routes';
import { shouldEnableWorkspaces } from '../../common/utils/FeatureUtils';
import { extractWorkspaceFromPathname } from '../../common/utils/WorkspaceUtils';
import { WorkspaceSelectionPrompt } from '../../common/components/WorkspaceSelectionPrompt';

const HomePage = () => {
  const [showWorkspacePrompt, setShowWorkspacePrompt] = useState(false);
  const location = useLocation();
  const workspacesEnabled = shouldEnableWorkspaces();
  const workspace = extractWorkspaceFromPathname(location.pathname);

  useEffect(() => {
    // Show prompt only if:
    // 1. Workspaces are enabled
    // 2. User is on root path (/)
    // 3. No workspace in URL
    if (workspacesEnabled && location.pathname === '/' && !workspace) {
      setShowWorkspacePrompt(true);
    } else {
      setShowWorkspacePrompt(false);
    }
  }, [workspacesEnabled, location.pathname, workspace]);

  // If showing workspace prompt, render it without navigation
  if (showWorkspacePrompt) {
    return (
      <WorkspaceSelectionPrompt
        isOpen={showWorkspacePrompt}
        onClose={() => setShowWorkspacePrompt(false)}
      />
    );
  }

  // Otherwise, redirect to experiments as normal
  return <Navigate to={Routes.experimentsObservatoryRoute} />;
};

export default HomePage;
