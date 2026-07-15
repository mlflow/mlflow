import type { DocumentTitleHandle } from '../common/utils/RoutingUtils';
import { createLazyRouteElement } from '../common/utils/RoutingUtils';
import { MCPRegistryPageId, MCPRegistryRoutePaths } from './routes';

export const getMCPRegistryRouteDefs = () => {
  return [
    {
      path: MCPRegistryRoutePaths.mcpRegistryPage,
      element: createLazyRouteElement(() => import('./pages/MCPRegistryPage')),
      pageId: MCPRegistryPageId.mcpRegistryPage,
      handle: { getPageTitle: () => 'MCP Registry' } satisfies DocumentTitleHandle,
    },
    {
      path: MCPRegistryRoutePaths.mcpServerDetailPage,
      element: createLazyRouteElement(() => import('./pages/MCPServerDetailPage')),
      pageId: MCPRegistryPageId.mcpServerDetailPage,
      handle: {
        getPageTitle: (params) => `MCP Server: ${decodeURIComponent(params['serverName'] || '')}`,
      } satisfies DocumentTitleHandle,
    },
  ];
};
