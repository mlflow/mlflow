import { createMLflowRoutePath, generatePath } from '../common/utils/RoutingUtils';

export enum MCPRegistryPageId {
  mcpRegistryPage = 'mlflow.mcp-registry',
  mcpServerDetailPage = 'mlflow.mcp-registry.server-detail',
}

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
export class MCPRegistryRoutePaths {
  static get mcpRegistryPage() {
    return createMLflowRoutePath('/mcp-registry');
  }

  static get mcpServerDetailPage() {
    return createMLflowRoutePath('/mcp-registry/:serverName');
  }
}

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
class MCPRegistryRoutes {
  static get mcpRegistryPageRoute() {
    return MCPRegistryRoutePaths.mcpRegistryPage;
  }

  static getMCPServerDetailRoute(serverName: string, version?: string) {
    const path = generatePath(MCPRegistryRoutePaths.mcpServerDetailPage, {
      serverName: encodeURIComponent(serverName),
    });
    if (version) {
      return `${path}?version=${encodeURIComponent(version)}`;
    }
    return path;
  }
}

export default MCPRegistryRoutes;
