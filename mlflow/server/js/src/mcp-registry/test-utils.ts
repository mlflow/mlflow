import { rest } from 'msw';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import type { MCPServer } from './types';

const BASE_URL = 'ajax-api/3.0/mlflow/mcp-servers';

export const createMockMCPServer = (overrides: Partial<MCPServer> = {}): MCPServer => ({
  name: 'io.github.test/server',
  access_bindings: [],
  aliases: [],
  tags: {},
  ...overrides,
});

export const getMockedSearchMCPServersResponse = (servers: MCPServer[] = []) =>
  rest.get(getAjaxUrl(BASE_URL), (_req, res, ctx) =>
    res(ctx.json({ mcp_servers: servers, next_page_token: undefined })),
  );

export const getMockedSearchMCPServersErrorResponse = (status = 500, message = 'Internal error') =>
  rest.get(getAjaxUrl(BASE_URL), (_req, res, ctx) => res(ctx.status(status), ctx.json({ message })));
