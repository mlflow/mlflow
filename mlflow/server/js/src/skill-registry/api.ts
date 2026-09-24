import { fetchAPI, fetchOrFail, getAjaxUrl, HTTPMethods } from '../common/utils/FetchUtils';
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
} from './types';

const BASE_URL = 'ajax-api/3.0/mlflow/skills';

export const buildSkillIdentityPath = (name: string, organization = ''): string => {
  const encodedName = encodeURIComponent(name);
  return organization ? `@${encodeURIComponent(organization)}/${encodedName}` : encodedName;
};

export const buildSkillSearchParams = (params: SearchSkillsParams = {}): string => {
  const searchParams = new URLSearchParams();
  if (params.filter_string !== undefined) {
    searchParams.append('filter_string', params.filter_string);
  }
  if (params.max_results !== undefined) {
    searchParams.append('max_results', String(params.max_results));
  }
  for (const orderBy of params.order_by ?? []) {
    searchParams.append('order_by', orderBy);
  }
  if (params.page_token !== undefined) {
    searchParams.append('page_token', params.page_token);
  }
  const queryString = searchParams.toString();
  return queryString ? `?${queryString}` : '';
};

export const buildSkillMultipartBody = (
  metadata: RegisterUploadedSkillRequest | UploadedSkillVersionRequest,
  content: Blob,
) => {
  const body = new FormData();
  body.append('metadata', new Blob([JSON.stringify(metadata)], { type: 'application/json' }));
  const gzipContent =
    content.type === 'application/gzip' ? content : content.slice(0, content.size, 'application/gzip');
  const filename = typeof File !== 'undefined' && content instanceof File ? content.name : 'skill.tar.gz';
  body.append('content', gzipContent, filename);
  return body;
};

const skillUrl = (name: string, organization = '') => `${BASE_URL}/${buildSkillIdentityPath(name, organization)}`;

function registerSkill(request: RegisterExternalSkillRequest): Promise<RegisterSkillResponse>;
function registerSkill(request: RegisterUploadedSkillRequest, content: Blob): Promise<RegisterSkillResponse>;
function registerSkill(request: RegisterSkillRequest, content?: Blob): Promise<RegisterSkillResponse> {
  if (content) {
    return fetchOrFail(getAjaxUrl(`${BASE_URL}/register`), {
      method: HTTPMethods.POST,
      body: buildSkillMultipartBody(request as RegisterUploadedSkillRequest, content),
    }).then((response) => response.json() as Promise<RegisterSkillResponse>);
  }
  return fetchAPI(getAjaxUrl(`${BASE_URL}/register`), {
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
    return fetchOrFail(getAjaxUrl(`${skillUrl(name, organization)}/versions`), {
      method: HTTPMethods.POST,
      body: buildSkillMultipartBody(request as UploadedSkillVersionRequest, content),
    }).then((response) => response.json() as Promise<GetSkillVersionResponse>);
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

  updateSkill: (name: string, request: UpdateSkillRequest, organization = ''): Promise<UpdateSkillResponse> => {
    return fetchAPI(getAjaxUrl(skillUrl(name, organization)), {
      method: HTTPMethods.PATCH,
      body: request,
    }) as Promise<UpdateSkillResponse>;
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
};
