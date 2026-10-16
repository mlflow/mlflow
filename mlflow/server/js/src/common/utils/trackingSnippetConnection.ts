import { getServerInfoUrl, SERVER_INFO_PATH, useServerInfo } from '../../experiment-tracking/hooks/useServerInfo';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';

export interface TrackingSnippetConnection {
  trackingUri?: string;
  trackingAuth?: string;
  workspace?: string | null;
}

export const getTrackingServerUri = (): string => {
  const url = new URL(getServerInfoUrl(), document.baseURI);
  const serverInfoPath = `/${SERVER_INFO_PATH}`;
  return `${url.origin}${url.pathname.slice(0, -serverInfoPath.length)}`;
};

export const useTrackingSnippetConnection = (): TrackingSnippetConnection => {
  const { data } = useServerInfo();
  const workspace = useActiveWorkspace();
  return {
    trackingUri: getTrackingServerUri(),
    trackingAuth: data?.tracking_auth,
    workspace,
  };
};
