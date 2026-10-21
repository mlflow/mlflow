import { rest } from 'msw';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { SkillAction, SkillStatus, type Skill, type SkillVersion } from './types';

const BASE_URL = 'ajax-api/3.0/mlflow/skills';
const skillGetPattern = /ajax-api\/3\.0\/mlflow\/skills\/(?:@[^/]+\/)?[^/]+$/;
const skillVersionsPattern = /ajax-api\/3\.0\/mlflow\/skills\/(?:@[^/]+\/)?[^/]+\/versions$/;
const skillVersionPattern = /ajax-api\/3\.0\/mlflow\/skills\/(?:@[^/]+\/)?[^/]+\/versions\/\d+$/;

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

export const createMockSkillVersion = (overrides: Partial<SkillVersion> = {}): SkillVersion => ({
  name: 'code-review',
  version: 2,
  organization: 'acme',
  source_type: 'git',
  source: 'https://github.com/acme/skills',
  ref: 'main',
  subpath: 'code-review',
  digest: 'sha256:abc123',
  status: SkillStatus.ACTIVE,
  aliases: [],
  tags: {},
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

export const getMockedGetSkillResponse = (skill: Skill) =>
  rest.get(skillGetPattern, (_req, res, ctx) => res(ctx.json(skill)));

export const getMockedGetSkillErrorResponse = (status = 404, message = 'Skill not found', errorCode?: string) =>
  rest.get(skillGetPattern, (_req, res, ctx) => res(ctx.status(status), ctx.json({ error_code: errorCode, message })));

export const getMockedGetSkillPermissionDeniedResponse = () =>
  getMockedGetSkillErrorResponse(403, 'Not allowed to view this skill', 'PERMISSION_DENIED');

export const getMockedSearchSkillVersionsResponse = (
  versions: SkillVersion[] = [],
  nextPageToken: string | null = null,
) =>
  rest.get(skillVersionsPattern, (_req, res, ctx) =>
    res(ctx.json({ skill_versions: versions, next_page_token: nextPageToken })),
  );

export const getMockedSearchSkillVersionsErrorResponse = (status = 500, message = 'Failed to load versions') =>
  rest.get(skillVersionsPattern, (_req, res, ctx) => res(ctx.status(status), ctx.json({ message })));

export const getMockedGetSkillVersionResponse = (versions: SkillVersion | SkillVersion[]) => {
  const list = Array.isArray(versions) ? versions : [versions];
  return rest.get(skillVersionPattern, (req, res, ctx) => {
    const versionNumber = Number(req.url.pathname.split('/').pop());
    const match = list.find((version) => version.version === versionNumber);
    if (!match) {
      return res(ctx.status(404), ctx.json({ message: 'Version not found' }));
    }
    return res(ctx.json(match));
  });
};

export const getMockedSkillDetailHandlers = (skill: Skill, versions: SkillVersion[]) => [
  getMockedGetSkillResponse(skill),
  getMockedSearchSkillVersionsResponse(versions),
  getMockedGetSkillVersionResponse(versions),
];

export { SkillAction, SkillStatus };
