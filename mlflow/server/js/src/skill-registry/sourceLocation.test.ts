import { describe, expect, it } from '@jest/globals';
import { SkillStatus } from './types';
import {
  buildExternalSkillVersionRequest,
  buildUploadedSkillVersionRequest,
  formatSkillImportCli,
  formatSkillImportPython,
  formatSkillRegisterCli,
  formatSkillRegisterPython,
  parseSkillIdentityInput,
  parseSkillLocation,
  toRegisterSkillRequest,
} from './sourceLocation';

const fields = {
  location: '',
  identity: '@acme/code-review',
  sourceTypeOverride: '' as const,
  ref: '',
  subpath: '',
  digest: '',
  status: SkillStatus.ACTIVE,
};

describe('parseSkillLocation', () => {
  it('derives a repository, ref, and folder from GitHub tree and blob links', () => {
    expect(
      parseSkillLocation(
        'https://github.com/RHEcosystemAppEng/agentic-plugins/tree/main/ocp-admin/skills/network-policy-architect',
      ),
    ).toMatchObject({
      sourceType: 'git',
      source: 'https://github.com/RHEcosystemAppEng/agentic-plugins',
      ref: 'main',
      subpath: 'ocp-admin/skills/network-policy-architect',
      suggestedName: 'network-policy-architect',
      suggestedOrganization: 'rhecosystemappeng',
      wholeRepository: false,
    });

    expect(parseSkillLocation('https://github.com/acme/skills/blob/main/skills/code-review/SKILL.md')).toMatchObject({
      ref: 'main',
      subpath: 'skills/code-review',
      suggestedName: 'code-review',
      wholeRepository: false,
    });
  });

  it('treats a repository clone URL as a whole repository', () => {
    expect(parseSkillLocation('https://github.com/redhat-ai/skills-developer.git')).toMatchObject({
      sourceType: 'git',
      source: 'https://github.com/redhat-ai/skills-developer.git',
      ref: null,
      subpath: null,
      suggestedName: 'skills-developer',
      suggestedOrganization: 'redhat-ai',
      wholeRepository: true,
      repositoryUrl: 'https://github.com/redhat-ai/skills-developer',
    });
    expect(formatSkillImportCli({ source: 'https://github.com/redhat-ai/skills-developer' })).toBe(
      "mlflow skills import --source 'https://github.com/redhat-ai/skills-developer'",
    );
  });

  it('infers OCI and ZIP pointers and leaves ambiguous URLs unresolved', () => {
    expect(parseSkillLocation('oci://ghcr.io/acme/skills:v1')).toMatchObject({
      sourceType: 'oci',
      source: 'ghcr.io/acme/skills:v1',
      suggestedName: 'skills',
      wholeRepository: false,
    });
    expect(parseSkillLocation('oci://localhost:5000/acme/reviewer:v1')?.suggestedName).toBe('reviewer');
    expect(parseSkillLocation('https://example.com/skills.zip')).toMatchObject({
      sourceType: 'zip',
      source: 'https://example.com/skills.zip',
      suggestedName: 'skills',
    });
    expect(parseSkillLocation('https://example.com/skills')).toBeUndefined();
    expect(parseSkillLocation('git@github.com:acme/skills.git')).toMatchObject({
      sourceType: 'git',
      source: 'git@github.com:acme/skills.git',
      wholeRepository: true,
    });
  });
});

describe('buildExternalSkillVersionRequest', () => {
  it('builds an organization-qualified git registration from a parsed GitHub link', () => {
    const built = buildExternalSkillVersionRequest({
      ...fields,
      location: 'https://github.com/acme/skills/tree/main/skills/code-review',
      identity: '',
      ref: 'main',
      subpath: 'skills/code-review',
    });
    expect(built.ok).toBe(false);

    const named = buildExternalSkillVersionRequest({
      ...fields,
      location: 'https://github.com/acme/skills/tree/main/skills/code-review',
      ref: 'main',
      subpath: 'skills/code-review',
      digest: 'sha256:' + 'a'.repeat(64),
    });
    expect(named.ok).toBe(true);
    if (!named.ok) return;
    expect(toRegisterSkillRequest(named.request, named.identity)).toEqual({
      name: 'code-review',
      organization: 'acme',
      source: 'https://github.com/acme/skills',
      source_type: 'git',
      ref: 'main',
      subpath: 'skills/code-review',
      digest: 'a'.repeat(64),
      status: 'active',
    });
  });

  it('uses an explicit source type for an otherwise ambiguous URL and rejects contradictions', () => {
    expect(
      buildExternalSkillVersionRequest({
        ...fields,
        location: 'https://example.com/skills',
        identity: 'prompt-style-guide',
      }),
    ).toEqual({ ok: false, error: 'source_type_required' });

    const zip = buildExternalSkillVersionRequest({
      ...fields,
      location: 'https://example.com/skills',
      identity: 'prompt-style-guide',
      sourceTypeOverride: 'zip',
    });
    expect(zip.ok).toBe(true);
    if (!zip.ok) return;
    expect(zip.request).toMatchObject({ source_type: 'zip', source: 'https://example.com/skills' });
    expect(zip.request).not.toHaveProperty('ref');
    expect(toRegisterSkillRequest(zip.request, zip.identity)).not.toHaveProperty('organization');

    const conflict = buildExternalSkillVersionRequest({
      ...fields,
      location: 'oci://ghcr.io/acme/skills:v1',
      sourceTypeOverride: 'git',
    });
    expect(conflict).toEqual({ ok: false, error: 'source_type_conflict' });
  });

  it('rejects credentials, refs on non-git sources, and malformed digests', () => {
    expect(
      buildExternalSkillVersionRequest({
        ...fields,
        location: 'https://user:token@github.com/acme/skills.git',
      }).ok,
    ).toBe(false);
    expect(
      buildExternalSkillVersionRequest({
        ...fields,
        location: 'https://example.com/skills.zip',
        ref: 'main',
      }),
    ).toEqual({ ok: false, error: 'ref_not_git' });
    expect(
      buildExternalSkillVersionRequest({
        ...fields,
        location: 'https://github.com/acme/skills.git',
        digest: 'abc',
      }),
    ).toEqual({ ok: false, error: 'digest_invalid' });
    expect(
      buildExternalSkillVersionRequest({
        ...fields,
        location: 'https://github.com/acme/skills.git',
        status: SkillStatus.DEPRECATED,
      }),
    ).toEqual({ ok: false, error: 'status' });
  });
});

describe('parseSkillIdentityInput', () => {
  it('splits an organization-qualified name and accepts a default-organization name', () => {
    expect(parseSkillIdentityInput('@acme/code-review')).toEqual({ name: 'code-review', organization: 'acme' });
    expect(parseSkillIdentityInput('prompt-style-guide')).toEqual({ name: 'prompt-style-guide', organization: '' });
    expect(parseSkillIdentityInput('@Acme/code-review')).toEqual({ error: 'organization_invalid' });
  });
});

describe('local skill registration', () => {
  it('builds an upload request without a remote source and formats the API example', () => {
    const built = buildUploadedSkillVersionRequest({ ...fields, identity: 'prompt-style-guide' });
    expect(built.ok).toBe(true);
    if (!built.ok) return;
    expect(built.request).toEqual({ source: null, status: 'active' });
    expect(toRegisterSkillRequest(built.request, built.identity)).toEqual({
      source: null,
      status: 'active',
      name: 'prompt-style-guide',
    });
    expect(formatSkillRegisterCli({ location: '', local: false })).toBe(
      "mlflow skills register git \\\n  --url '<location>'",
    );
    expect(
      formatSkillRegisterCli({
        location: 'https://example.com/skill.zip?token=a&next=b',
        local: false,
        sourceType: 'zip',
        name: "o'brien",
      }),
    ).toBe(
      "mlflow skills register zip \\\n  --name 'o'\\''brien' \\\n  --url 'https://example.com/skill.zip?token=a&next=b'",
    );
  });

  it('carries the ref, subpath, image and non-default status into the API examples', () => {
    const options = {
      location: 'https://github.com/acme/skills',
      local: false,
      sourceType: 'git' as const,
      name: 'code-review',
      organization: 'acme',
      ref: 'main',
      subpath: 'skills/code-review',
      status: SkillStatus.DRAFT,
    };
    expect(formatSkillRegisterCli(options)).toBe(
      [
        'mlflow skills register git',
        "  --name 'code-review'",
        "  --organization 'acme'",
        "  --url 'https://github.com/acme/skills'",
        "  --ref 'main'",
        "  --subpath 'skills/code-review'",
        '  --status draft',
      ].join(' \\\n'),
    );
    expect(formatSkillRegisterPython(options)).toBe(
      [
        'import mlflow',
        'from mlflow.genai import GitSource',
        '',
        'mlflow.genai.register_skill(',
        '    name="code-review",',
        '    organization="acme",',
        '    source=GitSource(url="https://github.com/acme/skills", ref="main", subpath="skills/code-review"),',
        '    status="draft",',
        ')',
      ].join('\n'),
    );
    expect(
      formatSkillRegisterCli({ location: 'quay.io/acme/skill:1.0', local: false, sourceType: 'oci', ref: 'main' }),
    ).toBe("mlflow skills register oci \\\n  --image 'quay.io/acme/skill:1.0'");
  });

  it('formats the repository import pointer', () => {
    expect(formatSkillImportCli({ source: 'https://github.com/acme/skills', ref: 'v1', organization: 'acme' })).toBe(
      "mlflow skills import --source 'https://github.com/acme/skills' \\\n  --ref 'v1' \\\n  --organization 'acme'",
    );
    expect(formatSkillImportPython({ source: 'https://github.com/acme/skills', organization: 'acme' })).toBe(
      'import mlflow\n\nmlflow.genai.import_skills(\n    source="https://github.com/acme/skills",\n    organization="acme",\n)',
    );
    expect(formatSkillImportPython({ source: 'https://github.com/acme/skills', ref: 'v1' })).toContain(
      'source=GitSource(url="https://github.com/acme/skills", ref="v1")',
    );
  });

  it('escapes user values in the Python registration snippet', () => {
    const snippet = formatSkillRegisterPython({
      location: 'https://example.com/x")\nimport os  # \\',
      local: false,
      sourceType: 'zip',
      name: 'code"review',
    });
    expect(snippet).toContain('ZipSource(url="https://example.com/x\\")\\nimport os  # \\\\")');
    expect(snippet).toContain('name="code\\"review"');
    expect(snippet.split('\n')).toHaveLength(7);
  });
});
