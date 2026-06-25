import { createMLflowRoutePath } from '../common/utils/RoutingUtils';

export enum MCPRegistryPageId {
  mcpRegistryPage = 'mlflow.mcp-registry',
}

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
export class MCPRegistryRoutePaths {
  static get mcpRegistryPage() {
    return createMLflowRoutePath('/mcp-registry');
  }
}

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
class MCPRegistryRoutes {
  static get mcpRegistryPageRoute() {
    return MCPRegistryRoutePaths.mcpRegistryPage;
  }
}

export default MCPRegistryRoutes;
