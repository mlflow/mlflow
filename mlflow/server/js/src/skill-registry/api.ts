import { getArtifactChunkedText } from '../common/utils/ArtifactUtils';
import { fetchAPI, fetchOrFail, getAjaxUrl, HTTPMethods } from '../common/utils/FetchUtils';
import { buildSearchParams } from '../common/utils/SearchUtils';
import type {
  CreateSkillVersionRequest,
  ExternalSkillVersionRequest,
  GetSkillResponse,
  GetSkillVersionResponse,
  RegisterExternalSkillRequest,
  RegisterSkillRequest,
  RegisterSkillResponse,
  RegisterUploadedSkillRequest,
  ResolveSkillAliasResponse,
  SearchSkillsParams,
  SearchSkillsResponse,
  SearchSkillVersionsParams,
  SearchSkillVersionsResponse,
  SetSkillAliasRequest,
  SetSkillTagRequest,
  SkillMutationResponse,
  UpdateSkillRequest,
  UpdateSkillResponse,
  UpdateSkillVersionStatusRequest,
  UpdateSkillVersionStatusResponse,
  UploadedSkillVersionRequest,
  CreateSkillRequest,
  Skill,
} from './types';

const BASE_URL = 'ajax-api/3.0/mlflow/skills';
// Uploaded skill content lives in MLflow artifact storage and is read through the artifact proxy.
const ARTIFACTS_URL = 'ajax-api/2.0/mlflow-artifacts/artifacts';

export interface SkillArtifactFileInfo {
  path: string;
  is_dir?: boolean;
  file_size?: number;
}

const encodeArtifactPath = (path: string) => path.split('/').map(encodeURIComponent).join('/');

export const buildSkillIdentityPath = (name: string, organization = ''): string => {
  const encodedName = encodeURIComponent(name);
  return organization ? `@${encodeURIComponent(organization)}/${encodedName}` : encodedName;
};

export const buildSkillSearchParams = (params: SearchSkillsParams = {}): string => {
  return buildSearchParams({
    filter_string: params.filter_string,
    max_results: params.max_results,
    order_by: params.order_by,
    page_token: params.page_token,
  });
};

export const buildSkillMultipartBody = (
  metadata: RegisterUploadedSkillRequest | UploadedSkillVersionRequest,
  content: Blob,
) => {
  const body = new FormData();
  // A Blob keeps the metadata part's application/json type. The backend therefore
  // receives metadata as an UploadFile rather than a string Form field.
  body.append('metadata', new Blob([JSON.stringify(metadata)], { type: 'application/json' }), 'metadata.json');
  const gzipContent =
    content.type === 'application/gzip' ? content : content.slice(0, content.size, 'application/gzip');
  const filename = typeof File !== 'undefined' && content instanceof File ? content.name : 'skill.tar.gz';
  body.append('content', gzipContent, filename);
  return body;
};

const skillUrl = (name: string, organization = '') => `${BASE_URL}/${buildSkillIdentityPath(name, organization)}`;

// fetchAPI always sets Content-Type: application/json, which would break the
// multipart boundary. fetchOrFail is used instead and leaves .response unread,
// so parse the backend message here rather than changing shared error text.
async function fetchSkillMultipartJson<T>(url: string, body: FormData): Promise<T> {
  try {
    const response = await fetchOrFail(url, {
      method: HTTPMethods.POST,
      body,
    });
    return response.json() as Promise<T>;
  } catch (error) {
    const response = (error as { response?: Response }).response;
    if (response && !response.bodyUsed) {
      try {
        const message = (await response.json()).message;
        if (typeof message === 'string' && error instanceof Error) {
          error.message = message;
        }
      } catch {
        // Keep the predefined fetchOrFail message when the body is not JSON.
      }
    }
    throw error;
  }
}

function registerSkill(request: RegisterExternalSkillRequest): Promise<RegisterSkillResponse>;
function registerSkill(request: RegisterUploadedSkillRequest, content: Blob): Promise<RegisterSkillResponse>;
function registerSkill(request: RegisterSkillRequest, content?: Blob): Promise<RegisterSkillResponse> {
  const url = getAjaxUrl(`${skillUrl(request.name, request.organization)}/versions`);
  if (content) {
    return fetchSkillMultipartJson<RegisterSkillResponse>(
      url,
      buildSkillMultipartBody(request as RegisterUploadedSkillRequest, content),
    );
  }
  return fetchAPI(url, {
    method: HTTPMethods.POST,
    body: request,
  }) as Promise<RegisterSkillResponse>;
}

function createSkillVersion(
  name: string,
  request: ExternalSkillVersionRequest,
  organization?: string,
): Promise<GetSkillVersionResponse>;
function createSkillVersion(
  name: string,
  request: UploadedSkillVersionRequest,
  organization: string | undefined,
  content: Blob,
): Promise<GetSkillVersionResponse>;
function createSkillVersion(
  name: string,
  request: CreateSkillVersionRequest,
  organization = '',
  content?: Blob,
): Promise<GetSkillVersionResponse> {
  if (content) {
    return fetchSkillMultipartJson<GetSkillVersionResponse>(
      getAjaxUrl(`${skillUrl(name, organization)}/versions`),
      buildSkillMultipartBody(request as UploadedSkillVersionRequest, content),
    );
  }
  return fetchAPI(getAjaxUrl(`${skillUrl(name, organization)}/versions`), {
    method: HTTPMethods.POST,
    body: request,
  }) as Promise<GetSkillVersionResponse>;
}

export const SkillRegistryApi = {
  searchSkills: (params: SearchSkillsParams = {}): Promise<SearchSkillsResponse> => {
    return fetchAPI(getAjaxUrl(`${BASE_URL}${buildSkillSearchParams(params)}`)) as Promise<SearchSkillsResponse>;
  },

  registerSkill,

  getSkill: (name: string, organization = ''): Promise<GetSkillResponse> => {
    return fetchAPI(getAjaxUrl(skillUrl(name, organization))) as Promise<GetSkillResponse>;
  },

  /** Creates a skill without versions; fails with RESOURCE_ALREADY_EXISTS when the name is taken. */
  createSkill: (request: CreateSkillRequest): Promise<Skill> => {
    return fetchAPI(getAjaxUrl(BASE_URL), {
      method: HTTPMethods.POST,
      body: request,
    }) as Promise<Skill>;
  },

  updateSkill: (name: string, request: UpdateSkillRequest, organization = ''): Promise<UpdateSkillResponse> => {
    return fetchAPI(getAjaxUrl(skillUrl(name, organization)), {
      method: HTTPMethods.PATCH,
      body: request,
    }) as Promise<UpdateSkillResponse>;
  },

  deleteSkill: (name: string, organization = ''): Promise<SkillMutationResponse> => {
    return fetchAPI(getAjaxUrl(skillUrl(name, organization)), {
      method: HTTPMethods.DELETE,
    }) as Promise<SkillMutationResponse>;
  },

  createSkillVersion,

  searchSkillVersions: (
    name: string,
    params: SearchSkillVersionsParams = {},
    organization = '',
  ): Promise<SearchSkillVersionsResponse> => {
    return fetchAPI(
      getAjaxUrl(`${skillUrl(name, organization)}/versions${buildSkillSearchParams(params)}`),
    ) as Promise<SearchSkillVersionsResponse>;
  },

  getSkillVersion: (name: string, version: number, organization = ''): Promise<GetSkillVersionResponse> => {
    return fetchAPI(
      getAjaxUrl(`${skillUrl(name, organization)}/versions/${encodeURIComponent(String(version))}`),
    ) as Promise<GetSkillVersionResponse>;
  },

  updateSkillVersionStatus: (
    name: string,
    version: number,
    request: UpdateSkillVersionStatusRequest,
    organization = '',
  ): Promise<UpdateSkillVersionStatusResponse> => {
    return fetchAPI(getAjaxUrl(`${skillUrl(name, organization)}/versions/${encodeURIComponent(String(version))}`), {
      method: HTTPMethods.PATCH,
      body: request,
    }) as Promise<UpdateSkillVersionStatusResponse>;
  },

  deleteSkillVersion: (name: string, version: number, organization = ''): Promise<SkillMutationResponse> => {
    return fetchAPI(getAjaxUrl(`${skillUrl(name, organization)}/versions/${encodeURIComponent(String(version))}`), {
      method: HTTPMethods.DELETE,
    }) as Promise<SkillMutationResponse>;
  },

  setSkillTag: (name: string, request: SetSkillTagRequest, organization = ''): Promise<SkillMutationResponse> => {
    return fetchAPI(getAjaxUrl(`${skillUrl(name, organization)}/tags`), {
      method: HTTPMethods.POST,
      body: request,
    }) as Promise<SkillMutationResponse>;
  },

  deleteSkillTag: (name: string, key: string, organization = ''): Promise<SkillMutationResponse> => {
    return fetchAPI(getAjaxUrl(`${skillUrl(name, organization)}/tags/${encodeURIComponent(key)}`), {
      method: HTTPMethods.DELETE,
    }) as Promise<SkillMutationResponse>;
  },

  setSkillVersionTag: (
    name: string,
    version: number,
    request: SetSkillTagRequest,
    organization = '',
  ): Promise<SkillMutationResponse> => {
    return fetchAPI(getAjaxUrl(`${skillUrl(name, organization)}/versions/${version}/tags`), {
      method: HTTPMethods.POST,
      body: request,
    }) as Promise<SkillMutationResponse>;
  },

  deleteSkillVersionTag: (
    name: string,
    version: number,
    key: string,
    organization = '',
  ): Promise<SkillMutationResponse> => {
    return fetchAPI(getAjaxUrl(`${skillUrl(name, organization)}/versions/${version}/tags/${encodeURIComponent(key)}`), {
      method: HTTPMethods.DELETE,
    }) as Promise<SkillMutationResponse>;
  },

  setSkillAlias: (name: string, request: SetSkillAliasRequest, organization = ''): Promise<SkillMutationResponse> => {
    return fetchAPI(getAjaxUrl(`${skillUrl(name, organization)}/aliases`), {
      method: HTTPMethods.POST,
      body: request,
    }) as Promise<SkillMutationResponse>;
  },

  resolveSkillAlias: (name: string, alias: string, organization = ''): Promise<ResolveSkillAliasResponse> => {
    return fetchAPI(
      getAjaxUrl(`${skillUrl(name, organization)}/aliases/${encodeURIComponent(alias)}`),
    ) as Promise<ResolveSkillAliasResponse>;
  },

  deleteSkillAlias: (name: string, alias: string, organization = ''): Promise<SkillMutationResponse> => {
    return fetchAPI(getAjaxUrl(`${skillUrl(name, organization)}/aliases/${encodeURIComponent(alias)}`), {
      method: HTTPMethods.DELETE,
    }) as Promise<SkillMutationResponse>;
  },

  /** Lists one directory; `path` is relative to the artifact root, as stored in a version's source. */
  listArtifacts: (path: string): Promise<{ files?: SkillArtifactFileInfo[] }> => {
    return fetchAPI(getAjaxUrl(`${ARTIFACTS_URL}?path=${encodeURIComponent(path)}`)) as Promise<{
      files?: SkillArtifactFileInfo[];
    }>;
  },

  getArtifactText: (path: string): Promise<string> => {
    return getArtifactChunkedText(getAjaxUrl(`${ARTIFACTS_URL}/${encodeArtifactPath(path)}`));
  },
};
