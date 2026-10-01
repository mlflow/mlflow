import { rest } from 'msw';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { SkillAction, SkillStatus, type Skill } from './types';

const BASE_URL = 'ajax-api/3.0/mlflow/skills';

export const createMockSkill = (overrides: Partial<Skill> = {}): Skill => ({
  name: 'code-review',
  organization: 'acme',
  description: 'Reviews pull requests',
  icons: null,
  status: SkillStatus.ACTIVE,
  latest_version: 2,
  aliases: [],
  tags: {},
  source_type: 'git',
  created_by: 'alice@example.com',
  last_updated_by: 'bob@example.com',
  creation_timestamp: 1772442000000,
  last_updated_timestamp: 1775034000000,
  ...overrides,
});

export const getMockedSearchSkillsResponse = (skills: Skill[] = [], nextPageToken: string | null = null) =>
  rest.get(getAjaxUrl(BASE_URL), (_req, res, ctx) => res(ctx.json({ skills, next_page_token: nextPageToken })));

export const getMockedSearchSkillsErrorResponse = (status = 500, message = 'Internal error', errorCode?: string) =>
  rest.get(getAjaxUrl(BASE_URL), (_req, res, ctx) =>
    res(ctx.status(status), ctx.json({ error_code: errorCode, message })),
  );

export const getMockedSearchSkillsPermissionDeniedResponse = () =>
  getMockedSearchSkillsErrorResponse(403, 'Not allowed to search skills', 'PERMISSION_DENIED');

export { SkillAction, SkillStatus };
