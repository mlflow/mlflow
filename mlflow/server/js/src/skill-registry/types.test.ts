import { describe, expect, it } from '@jest/globals';
import {
  SkillAction,
  SkillStatus,
  type RegisterSkillRequest,
  type SearchSkillsResponse,
  type SearchSkillVersionsResponse,
  type Skill,
  type SkillVersion,
} from './types';

const skill: Skill = {
  name: 'code-review',
  organization: '',
  description: null,
  icons: null,
  status: SkillStatus.ACTIVE,
  latest_version: 1,
  aliases: [{ alias: 'production', version: 1 }],
  tags: {},
  created_by: null,
  last_updated_by: null,
  creation_timestamp: null,
  last_updated_timestamp: null,
  allowed_actions: [SkillAction.USE],
};

const skillVersion: SkillVersion = {
  name: 'code-review',
  organization: '',
  version: 1,
  source_type: 'mlflow',
  source: 'skills/code-review/1',
  ref: null,
  subpath: null,
  digest: null,
  status: SkillStatus.ACTIVE,
  aliases: ['production'],
  tags: {},
  created_by: null,
  last_updated_by: null,
  creation_timestamp: null,
  last_updated_timestamp: null,
};

describe('Skill Registry wire contracts', () => {
  it('keeps versions numeric and permissions on the parent response', () => {
    expect(skill.latest_version).toBe(1);
    expect(skillVersion.version).toBe(1);
    expect(skill.allowed_actions).toEqual([SkillAction.USE]);
  });

  it('allows mlflow only as a stored response source type', () => {
    expect(skillVersion.source_type).toBe('mlflow');
  });

  it('models the terminal page token as an explicit null', () => {
    const skillsResponse: SearchSkillsResponse = { skills: [], next_page_token: null };
    const versionsResponse: SearchSkillVersionsResponse = { skill_versions: [], next_page_token: null };

    expect(skillsResponse.next_page_token).toBeNull();
    expect(versionsResponse.next_page_token).toBeNull();
  });
});

const invalidStringVersion: SkillVersion = {
  ...skillVersion,
  // @ts-expect-error Skill versions are server-assigned integers.
  version: '1',
};

const invalidVersionPermission: SkillVersion = {
  ...skillVersion,
  // @ts-expect-error Permissions are inherited from the parent and are not duplicated on versions.
  allowed_actions: [SkillAction.USE],
};

const invalidMlflowRequest: RegisterSkillRequest = {
  name: 'code-review',
  source: 'skills/code-review/1',
  // @ts-expect-error mlflow is assigned by the server during the upload flow.
  source_type: 'mlflow',
};

const invalidRegistrationMetadata: RegisterSkillRequest = {
  name: 'code-review',
  source: 'https://github.com/acme/skills.git',
  // @ts-expect-error Parent presentation metadata is updated through the parent endpoint.
  description: 'Code review instructions',
};

const invalidRegistrationTags: RegisterSkillRequest = {
  name: 'code-review',
  source: 'https://github.com/acme/skills.git',
  // @ts-expect-error Parent tags are managed through their dedicated endpoint.
  tags: { team: 'platform' },
};

const invalidClientSelectedVersion: RegisterSkillRequest = {
  name: 'code-review',
  source: 'https://github.com/acme/skills.git',
  // @ts-expect-error Skill version numbers are assigned by the server.
  version: 3,
};

void invalidStringVersion;
void invalidVersionPermission;
void invalidMlflowRequest;
void invalidRegistrationMetadata;
void invalidRegistrationTags;
void invalidClientSelectedVersion;
