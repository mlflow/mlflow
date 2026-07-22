import { rest } from 'msw';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import type { MCPAccessEndpoint, MCPServer, MCPServerVersion } from './types';
import { MCPStatus, TransportType } from './types';

const BASE_URL = 'ajax-api/3.0/mlflow/mcp-servers';

export const createMockMCPServer = (overrides: Partial<MCPServer> = {}): MCPServer => ({
  name: 'io.github.test/server',
  aliases: [],
  tags: {},
  ...overrides,
});

export const createMockMCPServerVersion = (overrides: Partial<MCPServerVersion> = {}): MCPServerVersion => ({
  name: 'io.github.test/server',
  version: '1',
  server_json: {
    name: 'io.github.test/server',
    version: '1.0.0',
    title: 'Test Server',
    description: 'A test MCP server',
  },
  status: MCPStatus.ACTIVE,
  aliases: [],
  tags: {},
  creation_timestamp: 1717520552000,
  ...overrides,
});

export const getMockedSearchMCPServersResponse = (
  servers: MCPServer[] = [],
  { userHasManage }: { userHasManage?: boolean } = {},
) =>
  rest.get(getAjaxUrl(BASE_URL), (_req, res, ctx) =>
    res(ctx.json({ mcp_servers: servers, next_page_token: undefined, user_has_manage: userHasManage })),
  );

export const getMockedSearchMCPServersErrorResponse = (status = 500, message = 'Internal error') =>
  rest.get(getAjaxUrl(BASE_URL), (_req, res, ctx) => res(ctx.status(status), ctx.json({ message })));

export const getMockedGetMCPServerResponse = (server: MCPServer) =>
  rest.get(getAjaxUrl(`${BASE_URL}/:name`), (_req, res, ctx) => res(ctx.json(server)));

export const getMockedGetMCPServerErrorResponse = (status = 404, message = 'Not found') =>
  rest.get(getAjaxUrl(`${BASE_URL}/:name`), (_req, res, ctx) => res(ctx.status(status), ctx.json({ message })));

export const getMockedSearchMCPServerVersionsResponse = (versions: MCPServerVersion[] = []) =>
  rest.get(getAjaxUrl(`${BASE_URL}/:name/versions`), (_req, res, ctx) =>
    res(ctx.json({ mcp_server_versions: versions, next_page_token: undefined })),
  );

export const getMockedUpdateMCPServerVersionResponse = (version: MCPServerVersion) =>
  rest.patch(getAjaxUrl(`${BASE_URL}/:name/versions/:version`), (_req, res, ctx) => res(ctx.json(version)));

export const getMockedDeleteMCPServerVersionResponse = () =>
  rest.delete(getAjaxUrl(`${BASE_URL}/:name/versions/:version`), (_req, res, ctx) => res(ctx.json({})));

export const getMockedDeleteMCPServerResponse = () =>
  rest.delete(getAjaxUrl(`${BASE_URL}/:name`), (_req, res, ctx) => res(ctx.json({})));

export const getMockedCreateMCPServerVersionResponse = (version?: MCPServerVersion) =>
  rest.post(getAjaxUrl(`${BASE_URL}/:name/versions`), (_req, res, ctx) =>
    res(ctx.json(version ?? createMockMCPServerVersion())),
  );

export const getMockedCreateMCPServerVersionErrorResponse = (status = 400, message = 'Bad request') =>
  rest.post(getAjaxUrl(`${BASE_URL}/:name/versions`), (_req, res, ctx) =>
    res(ctx.status(status), ctx.json({ message })),
  );

export const getMockedGetLatestMCPServerVersionResponse = (version?: MCPServerVersion) =>
  rest.get(getAjaxUrl(`${BASE_URL}/:name/aliases/latest`), (_req, res, ctx) =>
    version ? res(ctx.json(version)) : res(ctx.status(404), ctx.json({ message: 'No eligible latest version' })),
  );

export const getMockedUpdateMCPServerResponse = (server?: MCPServer) =>
  rest.patch(getAjaxUrl(`${BASE_URL}/:name`), (_req, res, ctx) => res(ctx.json(server ?? createMockMCPServer())));

export const getMockedUpdateMCPServerErrorResponse = (status = 400, message = 'Bad request') =>
  rest.patch(getAjaxUrl(`${BASE_URL}/:name`), (_req, res, ctx) => res(ctx.status(status), ctx.json({ message })));

export const getMockedSetMCPServerTagResponse = () =>
  rest.post(getAjaxUrl(`${BASE_URL}/:name/tags`), (_req, res, ctx) => res(ctx.json({})));

export const getMockedDeleteMCPServerTagResponse = () =>
  rest.delete(getAjaxUrl(`${BASE_URL}/:name/tags/:key`), (_req, res, ctx) => res(ctx.json({})));

export const getMockedCurrentUserResponse = ({ isAdmin = false }: { isAdmin?: boolean } = {}) =>
  rest.get(getAjaxUrl('ajax-api/2.0/mlflow/users/current'), (_req, res, ctx) =>
    res(ctx.json({ user: { username: 'testuser', is_admin: isAdmin } })),
  );

// Access endpoint mocks

export const createMockAccessEndpoint = (overrides: Partial<MCPAccessEndpoint> = {}): MCPAccessEndpoint => ({
  id: 'ae-mock-1',
  server_name: 'io.github.test/server',
  url: 'https://example.com/mcp',
  transport_type: TransportType.STREAMABLE_HTTP,
  ...overrides,
});

export const getMockedSearchAccessEndpointsResponse = (endpoints: MCPAccessEndpoint[] = []) =>
  rest.get(getAjaxUrl(`${BASE_URL}/:name/endpoints`), (_req, res, ctx) =>
    res(ctx.json({ mcp_access_endpoints: endpoints, next_page_token: undefined })),
  );

export const getMockedCreateAccessEndpointResponse = (endpoint?: MCPAccessEndpoint) =>
  rest.post(getAjaxUrl(`${BASE_URL}/:name/endpoints`), (_req, res, ctx) =>
    res(ctx.json(endpoint ?? createMockAccessEndpoint())),
  );

export const getMockedUpdateAccessEndpointResponse = (endpoint?: MCPAccessEndpoint) =>
  rest.patch(getAjaxUrl(`${BASE_URL}/:name/endpoints/:endpointId`), (_req, res, ctx) =>
    res(ctx.json(endpoint ?? createMockAccessEndpoint())),
  );

export const getMockedDeleteAccessEndpointResponse = () =>
  rest.delete(getAjaxUrl(`${BASE_URL}/:name/endpoints/:endpointId`), (_req, res, ctx) => res(ctx.json({})));

export const getMockedAccessEndpointErrorResponse = (
  method: 'post' | 'patch' = 'post',
  status = 400,
  message = 'Bad request',
) => {
  const handler =
    method === 'post'
      ? rest.post(getAjaxUrl(`${BASE_URL}/:name/endpoints`), (_req, res, ctx) =>
          res(ctx.status(status), ctx.json({ message })),
        )
      : rest.patch(getAjaxUrl(`${BASE_URL}/:name/endpoints/:endpointId`), (_req, res, ctx) =>
          res(ctx.status(status), ctx.json({ message })),
        );
  return handler;
};
