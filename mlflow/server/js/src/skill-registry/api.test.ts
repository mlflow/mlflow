import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { buildSkillIdentityPath, buildSkillMultipartBody, buildSkillSearchParams, SkillRegistryApi } from './api';
import {
  SkillStatus,
  type ExternalSkillVersionRequest,
  type RegisterExternalSkillRequest,
  type RegisterUploadedSkillRequest,
  type UploadedSkillVersionRequest,
} from './types';

const jsonResponse = (body: unknown = {}) =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve(body),
  } as Response);

describe('Skill Registry API', () => {
  let fetchMock: jest.SpiedFunction<typeof global.fetch>;

  beforeEach(() => {
    fetchMock = jest.spyOn(global, 'fetch').mockImplementation(() => jsonResponse());
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('route construction', () => {
    it('builds and encodes default and explicit organization identities one segment at a time', () => {
      expect(buildSkillIdentityPath('name/with space')).toBe('name%2Fwith%20space');
      expect(buildSkillIdentityPath('name/with space', 'org/with space')).toBe(
        '@org%2Fwith%20space/name%2Fwith%20space',
      );
      expect(buildSkillIdentityPath('skill', '')).toBe('skill');
    });

    const routeFamilies = [
      {
        name: 'parent',
        defaultPath: 'ajax-api/3.0/mlflow/skills/my%2Fskill',
        organizationPath: 'ajax-api/3.0/mlflow/skills/@my%2Forg/my%2Fskill',
        invoke: (organization?: string) => SkillRegistryApi.getSkill('my/skill', organization),
      },
      {
        name: 'versions collection',
        defaultPath: 'ajax-api/3.0/mlflow/skills/my%2Fskill/versions',
        organizationPath: 'ajax-api/3.0/mlflow/skills/@my%2Forg/my%2Fskill/versions',
        invoke: (organization?: string) =>
          SkillRegistryApi.createSkillVersion('my/skill', { source: 'git://example/repo' }, organization),
      },
      {
        name: 'version',
        defaultPath: 'ajax-api/3.0/mlflow/skills/my%2Fskill/versions/12',
        organizationPath: 'ajax-api/3.0/mlflow/skills/@my%2Forg/my%2Fskill/versions/12',
        invoke: (organization?: string) => SkillRegistryApi.getSkillVersion('my/skill', 12, organization),
      },
      {
        name: 'parent tags',
        defaultPath: 'ajax-api/3.0/mlflow/skills/my%2Fskill/tags/key%2Fwith%20space',
        organizationPath: 'ajax-api/3.0/mlflow/skills/@my%2Forg/my%2Fskill/tags/key%2Fwith%20space',
        invoke: (organization?: string) => SkillRegistryApi.deleteSkillTag('my/skill', 'key/with space', organization),
      },
      {
        name: 'version tags',
        defaultPath: 'ajax-api/3.0/mlflow/skills/my%2Fskill/versions/12/tags/key%2Fwith%20space',
        organizationPath: 'ajax-api/3.0/mlflow/skills/@my%2Forg/my%2Fskill/versions/12/tags/key%2Fwith%20space',
        invoke: (organization?: string) =>
          SkillRegistryApi.deleteSkillVersionTag('my/skill', 12, 'key/with space', organization),
      },
      {
        name: 'aliases',
        defaultPath: 'ajax-api/3.0/mlflow/skills/my%2Fskill/aliases/alias%2Fwith%20space',
        organizationPath: 'ajax-api/3.0/mlflow/skills/@my%2Forg/my%2Fskill/aliases/alias%2Fwith%20space',
        invoke: (organization?: string) =>
          SkillRegistryApi.resolveSkillAlias('my/skill', 'alias/with space', organization),
      },
    ];

    it.each(routeFamilies)('supports both organization forms for the $name route family', async (route) => {
      await route.invoke();
      expect(fetchMock).toHaveBeenLastCalledWith(route.defaultPath, expect.any(Object));

      await route.invoke('my/org');
      expect(fetchMock).toHaveBeenLastCalledWith(route.organizationPath, expect.any(Object));
    });
  });

  describe('query serialization', () => {
    it('serializes RFC search parameters without reinterpreting them', async () => {
      const params = {
        filter_string: "search_text LIKE '%review%' AND status = 'active'",
        max_results: 25,
        order_by: ['name ASC', 'latest_version DESC'],
        page_token: 'page/+ token',
      };

      expect(buildSkillSearchParams(params)).toBe(
        '?filter_string=search_text+LIKE+%27%25review%25%27+AND+status+%3D+%27active%27&max_results=25&order_by=name+ASC&order_by=latest_version+DESC&page_token=page%2F%2B+token',
      );

      await SkillRegistryApi.searchSkills(params);
      expect(fetchMock).toHaveBeenLastCalledWith(
        'ajax-api/3.0/mlflow/skills?filter_string=search_text+LIKE+%27%25review%25%27+AND+status+%3D+%27active%27&max_results=25&order_by=name+ASC&order_by=latest_version+DESC&page_token=page%2F%2B+token',
        expect.any(Object),
      );
    });
  });

  describe('endpoint request construction', () => {
    const endpointCases = [
      {
        name: 'update a parent',
        invoke: () => SkillRegistryApi.updateSkill('code-review', { description: 'Updated' }, 'acme'),
        url: 'ajax-api/3.0/mlflow/skills/@acme/code-review',
        method: 'PATCH',
        body: { description: 'Updated' },
      },
      {
        name: 'search versions',
        invoke: () =>
          SkillRegistryApi.searchSkillVersions('code-review', { filter_string: "status = 'active'" }, 'acme'),
        url: 'ajax-api/3.0/mlflow/skills/@acme/code-review/versions?filter_string=status+%3D+%27active%27',
        method: 'GET',
      },
      {
        name: 'update version status',
        invoke: () =>
          SkillRegistryApi.updateSkillVersionStatus('code-review', 3, { status: SkillStatus.DEPRECATED }, 'acme'),
        url: 'ajax-api/3.0/mlflow/skills/@acme/code-review/versions/3',
        method: 'PATCH',
        body: { status: SkillStatus.DEPRECATED },
      },
      {
        name: 'soft-delete a version',
        invoke: () => SkillRegistryApi.deleteSkillVersion('code-review', 3, 'acme'),
        url: 'ajax-api/3.0/mlflow/skills/@acme/code-review/versions/3',
        method: 'DELETE',
      },
      {
        name: 'set a parent tag',
        invoke: () => SkillRegistryApi.setSkillTag('code-review', { key: 'team', value: 'platform' }, 'acme'),
        url: 'ajax-api/3.0/mlflow/skills/@acme/code-review/tags',
        method: 'POST',
        body: { key: 'team', value: 'platform' },
      },
      {
        name: 'set a version tag',
        invoke: () => SkillRegistryApi.setSkillVersionTag('code-review', 3, { key: 'approved', value: 'true' }, 'acme'),
        url: 'ajax-api/3.0/mlflow/skills/@acme/code-review/versions/3/tags',
        method: 'POST',
        body: { key: 'approved', value: 'true' },
      },
      {
        name: 'set an alias',
        invoke: () => SkillRegistryApi.setSkillAlias('code-review', { alias: 'production', version: 3 }, 'acme'),
        url: 'ajax-api/3.0/mlflow/skills/@acme/code-review/aliases',
        method: 'POST',
        body: { alias: 'production', version: 3 },
      },
      {
        name: 'delete an alias',
        invoke: () => SkillRegistryApi.deleteSkillAlias('code-review', 'production', 'acme'),
        url: 'ajax-api/3.0/mlflow/skills/@acme/code-review/aliases/production',
        method: 'DELETE',
      },
    ];

    it.each(endpointCases)('uses the RFC request for $name', async ({ invoke, url, method, body }) => {
      await invoke();

      const [actualUrl, options] = fetchMock.mock.calls[0];
      expect(actualUrl).toBe(url);
      expect(options?.method).toBe(method);
      if (body) {
        expect(options?.body).toBe(JSON.stringify(body));
      } else {
        expect(options?.body).toBeUndefined();
      }
    });
  });

  describe('JSON registration', () => {
    it.each([
      {
        name: 'code-review',
        source_type: 'git',
        source: 'https://github.com/acme/skills.git',
        ref: 'main',
        subpath: 'skills/code-review',
        digest: null,
        status: SkillStatus.DRAFT,
      },
      {
        name: 'code-review',
        source_type: 'oci',
        source: 'ghcr.io/acme/skills:v1',
        subpath: 'skills/code-review',
        digest: null,
        status: SkillStatus.DRAFT,
      },
      {
        name: 'code-review',
        source_type: 'zip',
        source: 'https://example.com/skills.zip',
        subpath: 'skills/code-review',
        digest: null,
        status: SkillStatus.DRAFT,
      },
    ] satisfies RegisterExternalSkillRequest[])('registers an external source as JSON', async (request) => {
      await SkillRegistryApi.registerSkill(request);

      expect(fetchMock).toHaveBeenLastCalledWith(
        'ajax-api/3.0/mlflow/skills/register',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
          body: JSON.stringify(request),
        }),
      );
    });

    it.each([
      { source_type: 'git', source: 'https://github.com/acme/skills.git', status: SkillStatus.ACTIVE },
      { source_type: 'oci', source: 'ghcr.io/acme/skills:v1', status: SkillStatus.ACTIVE },
      { source_type: 'zip', source: 'https://example.com/skills.zip', status: SkillStatus.ACTIVE },
    ] satisfies ExternalSkillVersionRequest[])('creates a version for an external source as JSON', async (request) => {
      await SkillRegistryApi.createSkillVersion('code-review', request);

      expect(fetchMock).toHaveBeenLastCalledWith(
        'ajax-api/3.0/mlflow/skills/code-review/versions',
        expect.objectContaining({ body: JSON.stringify(request) }),
      );
    });
  });

  describe('multipart registration', () => {
    const readBlob = (blob: Blob) => new Response(blob).text();

    it('constructs application/json metadata and application/gzip content parts', async () => {
      const metadata: UploadedSkillVersionRequest = { status: SkillStatus.ACTIVE };
      const body = buildSkillMultipartBody(metadata, new Blob(['archive']));

      const metadataPart = body.get('metadata');
      const contentPart = body.get('content');
      expect(metadataPart).toBeInstanceOf(Blob);
      expect((metadataPart as Blob).type).toBe('application/json');
      expect((metadataPart as File).name).toBe('metadata.json');
      expect(await readBlob(metadataPart as Blob)).toBe(JSON.stringify(metadata));
      expect(contentPart).toBeInstanceOf(Blob);
      expect((contentPart as Blob).type).toBe('application/gzip');
    });

    it('registers local content atomically and leaves the multipart boundary to the browser', async () => {
      const metadata: RegisterUploadedSkillRequest = { name: 'code-review', status: SkillStatus.ACTIVE };

      await SkillRegistryApi.registerSkill(metadata, new Blob(['archive'], { type: 'application/gzip' }));

      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [url, options] = fetchMock.mock.calls[0];
      expect(url).toBe('ajax-api/3.0/mlflow/skills/register');
      expect(options?.body).toBeInstanceOf(FormData);
      expect(new Headers(options?.headers).has('Content-Type')).toBe(false);
    });

    it('creates a local-content version atomically with no upload or finalize request', async () => {
      await SkillRegistryApi.createSkillVersion(
        'code-review',
        { status: SkillStatus.DRAFT },
        'acme',
        new Blob(['archive'], { type: 'application/gzip' }),
      );

      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [url, options] = fetchMock.mock.calls[0];
      expect(url).toBe('ajax-api/3.0/mlflow/skills/@acme/code-review/versions');
      expect(options?.body).toBeInstanceOf(FormData);
      expect(new Headers(options?.headers).has('Content-Type')).toBe(false);
    });

    it.each([400, 409])('preserves the backend error message for a %s multipart failure', async (status) => {
      fetchMock.mockResolvedValueOnce(
        new Response(JSON.stringify({ message: 'Skill registration failed' }), {
          status,
          headers: { 'Content-Type': 'application/json' },
        }),
      );

      await expect(
        SkillRegistryApi.registerSkill(
          { name: 'code-review', status: SkillStatus.ACTIVE },
          new Blob(['archive'], { type: 'application/gzip' }),
        ),
      ).rejects.toThrow('Skill registration failed');
    });
  });
});
