import {
  SkillStatus,
  type CreateSkillVersionRequest,
  type CreateSkillVersionStatus,
  type ExternalSkillVersionRequest,
  type RegisterExternalSkillRequest,
  type RegisterUploadedSkillRequest,
  type UploadedSkillVersionRequest,
} from './types';

export type SkillRegistrationSourceType = 'git' | 'oci' | 'zip';

const SKILL_NAME_PATTERN = /^[a-z0-9]+(-[a-z0-9]+)*$/;
const ORGANIZATION_NAME_PATTERN = /^[a-z0-9]+([-.][a-z0-9]+)*$/;
const DIGEST_PATTERN = /^[0-9a-f]{64}$/;
const MAX_FIELD_LENGTH = 2048;
const MAX_NAME_LENGTH = 64;

export interface ParsedSkillLocation {
  sourceType: SkillRegistrationSourceType;
  /** Pointer submitted to the registry. GitHub tree and blob links become the repository URL. */
  source: string;
  ref: string | null;
  subpath: string | null;
  suggestedName: string;
  suggestedOrganization: string;
  wholeRepository: boolean;
  /** Clone or repository URL used by the bulk-import hint. */
  repositoryUrl: string | null;
}

export type SkillRegistrationErrorCode =
  | 'location_required'
  | 'name_required'
  | 'name_invalid'
  | 'organization_invalid'
  | 'source_type_required'
  | 'source_type_conflict'
  | 'credentials'
  | 'zip_scheme'
  | 'ref_not_git'
  | 'digest_invalid'
  | 'source_invalid'
  | 'source_too_long'
  | 'status'
  | 'skill_md_required';

export interface SkillRegistrationFields {
  location: string;
  identity: string;
  sourceTypeOverride: '' | SkillRegistrationSourceType;
  ref: string;
  subpath: string;
  digest: string;
  status: SkillStatus;
}

export type BuiltSkillRegistration<Request extends CreateSkillVersionRequest = CreateSkillVersionRequest> =
  | { ok: true; request: Request; identity: { name: string; organization: string } }
  | { ok: false; error: SkillRegistrationErrorCode };

const normalizeToken = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/-{2,}/g, '-');

const decodeSegment = (value: string) => {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
};

const hasHttpCredentials = (value: string) => {
  try {
    const url = new URL(value);
    if (url.password) return true;
    return Boolean(url.username) && (url.protocol === 'http:' || url.protocol === 'https:');
  } catch {
    return false;
  }
};

const imageName = (image: string) => {
  const withoutDigest = image.split('@')[0] ?? image;
  const withoutTag = withoutDigest.split(':')[0] ?? withoutDigest;
  return withoutTag.split('/').filter(Boolean).pop() ?? '';
};

const parseGitHubLocation = (value: string): ParsedSkillLocation | undefined => {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    return undefined;
  }
  if (url.protocol !== 'https:' || url.hostname !== 'github.com') return undefined;
  const segments = url.pathname.split('/').filter(Boolean).map(decodeSegment);
  if (segments.length < 2) return undefined;
  const [owner, repoRaw, kind, ...rest] = segments;
  const repo = repoRaw.replace(/\.git$/i, '');
  const repositoryUrl = `https://github.com/${owner}/${repo}`;
  const suggestedOrganization = normalizeToken(owner);
  const suggestedRepoName = normalizeToken(repo);
  if (!kind) {
    return {
      sourceType: 'git',
      source: repoRaw.toLowerCase().endsWith('.git') ? `${repositoryUrl}.git` : repositoryUrl,
      ref: null,
      subpath: null,
      suggestedName: suggestedRepoName,
      suggestedOrganization,
      wholeRepository: true,
      repositoryUrl,
    };
  }
  if (kind !== 'tree' && kind !== 'blob') return undefined;
  const ref = rest[0] ?? '';
  const pathSegments = kind === 'blob' ? rest.slice(1, -1) : rest.slice(1);
  const subpath = pathSegments.join('/') || null;
  const folderName = pathSegments.at(-1) ?? repo;
  return {
    sourceType: 'git',
    source: repositoryUrl,
    ref: ref || null,
    subpath,
    suggestedName: normalizeToken(folderName),
    suggestedOrganization,
    wholeRepository: !subpath,
    repositoryUrl,
  };
};

const parseScpGitLocation = (value: string): ParsedSkillLocation | undefined => {
  const match = value.match(/^git@([^:]+):(.+)$/);
  if (!match) return undefined;
  const path = match[2].replace(/\.git$/i, '');
  const [owner, repo] = path.split('/');
  return {
    sourceType: 'git',
    source: value,
    ref: null,
    subpath: null,
    suggestedName: normalizeToken(repo || owner || ''),
    suggestedOrganization: repo ? normalizeToken(owner) : '',
    wholeRepository: true,
    repositoryUrl: value,
  };
};

export const parseSkillLocation = (location: string): ParsedSkillLocation | undefined => {
  const value = location.trim();
  if (!value) return undefined;

  if (value.toLowerCase().startsWith('oci://')) {
    const image = value.slice('oci://'.length).trim();
    if (!image) return undefined;
    return {
      sourceType: 'oci',
      source: image,
      ref: null,
      subpath: null,
      suggestedName: normalizeToken(imageName(image)),
      suggestedOrganization: '',
      wholeRepository: false,
      repositoryUrl: null,
    };
  }

  const github = parseGitHubLocation(value);
  if (github) return github;

  if (value.toLowerCase().startsWith('git://') || /\.git\/?$/i.test(value)) {
    const repoName =
      value
        .split('/')
        .pop()
        ?.replace(/\.git\/?$/i, '') ?? '';
    return {
      sourceType: 'git',
      source: value.replace(/\/$/, ''),
      ref: null,
      subpath: null,
      suggestedName: normalizeToken(repoName),
      suggestedOrganization: '',
      wholeRepository: true,
      repositoryUrl: value.replace(/\/$/, ''),
    };
  }

  const scp = parseScpGitLocation(value);
  if (scp) return scp;

  if (/\.zip$/i.test(value.split('?')[0] ?? value)) {
    let zipUrl: URL | undefined;
    try {
      zipUrl = new URL(value);
    } catch {
      zipUrl = undefined;
    }
    if (zipUrl && (zipUrl.protocol === 'http:' || zipUrl.protocol === 'https:')) {
      const fileName =
        zipUrl.pathname
          .split('/')
          .pop()
          ?.replace(/\.zip$/i, '') ?? '';
      return {
        sourceType: 'zip',
        source: value,
        ref: null,
        subpath: null,
        suggestedName: normalizeToken(fileName),
        suggestedOrganization: '',
        wholeRepository: false,
        repositoryUrl: null,
      };
    }
  }

  return undefined;
};

export const formatSkillImportCommand = (repositoryUrl: string) => `mlflow skills import --source ${repositoryUrl}`;

export const parseSkillIdentityInput = (
  identity: string,
): { name: string; organization: string } | { error: 'name_required' | 'name_invalid' | 'organization_invalid' } => {
  const trimmed = identity.trim();
  if (!trimmed) return { error: 'name_required' };
  let name = trimmed;
  let organization = '';
  if (trimmed.startsWith('@')) {
    const slash = trimmed.indexOf('/');
    if (slash <= 1 || slash === trimmed.length - 1 || trimmed.slice(slash + 1).includes('/')) {
      return { error: 'name_invalid' };
    }
    organization = trimmed.slice(1, slash);
    name = trimmed.slice(slash + 1);
  }
  if (!SKILL_NAME_PATTERN.test(name) || name.length > MAX_NAME_LENGTH) return { error: 'name_invalid' };
  if (organization && (!ORGANIZATION_NAME_PATTERN.test(organization) || organization.length > MAX_NAME_LENGTH)) {
    return { error: 'organization_invalid' };
  }
  return { name, organization };
};

const normalizeDigest = (digest: string) =>
  digest
    .trim()
    .toLowerCase()
    .replace(/^sha256:/, '');

const creatableStatus = (status: SkillStatus): CreateSkillVersionStatus | undefined =>
  status === SkillStatus.ACTIVE || status === SkillStatus.DRAFT ? status : undefined;

const unsafeSource = (value: string) =>
  value.startsWith('-') ||
  [...value].some((character) => {
    const code = character.charCodeAt(0);
    return code <= 32 || code === 127;
  });

export const buildExternalSkillVersionRequest = (
  fields: SkillRegistrationFields,
): BuiltSkillRegistration<ExternalSkillVersionRequest> => {
  const location = fields.location.trim();
  if (!location) return { ok: false, error: 'location_required' };
  if (location.length > MAX_FIELD_LENGTH) return { ok: false, error: 'source_too_long' };
  if (unsafeSource(location) || hasHttpCredentials(location)) return { ok: false, error: 'credentials' };

  const identity = parseSkillIdentityInput(fields.identity);
  if ('error' in identity) return { ok: false, error: identity.error };

  const parsed = parseSkillLocation(location);
  const sourceType = fields.sourceTypeOverride || parsed?.sourceType;
  if (!sourceType) return { ok: false, error: 'source_type_required' };
  if (parsed && fields.sourceTypeOverride && fields.sourceTypeOverride !== parsed.sourceType) {
    return { ok: false, error: 'source_type_conflict' };
  }

  const source = parsed?.source ?? location;
  const ref = fields.ref.trim();
  const subpath = fields.subpath.trim();
  if (ref && sourceType !== 'git') return { ok: false, error: 'ref_not_git' };
  if (sourceType === 'zip') {
    try {
      const url = new URL(source);
      if (url.protocol !== 'http:' && url.protocol !== 'https:') return { ok: false, error: 'zip_scheme' };
    } catch {
      return { ok: false, error: 'zip_scheme' };
    }
  }
  if (sourceType === 'oci' && !source) return { ok: false, error: 'source_invalid' };
  if ([source, ref, subpath].some((value) => value.length > MAX_FIELD_LENGTH)) {
    return { ok: false, error: 'source_too_long' };
  }

  const digest = normalizeDigest(fields.digest);
  if (digest && !DIGEST_PATTERN.test(digest)) return { ok: false, error: 'digest_invalid' };

  const status = creatableStatus(fields.status);
  if (!status) return { ok: false, error: 'status' };

  const request: ExternalSkillVersionRequest =
    sourceType === 'git'
      ? {
          source,
          source_type: 'git',
          ...(ref ? { ref } : {}),
          ...(subpath ? { subpath } : {}),
          ...(digest ? { digest } : {}),
          status,
        }
      : sourceType === 'oci'
        ? {
            source,
            source_type: 'oci',
            ...(subpath ? { subpath } : {}),
            ...(digest ? { digest } : {}),
            status,
          }
        : {
            source,
            source_type: 'zip',
            ...(subpath ? { subpath } : {}),
            ...(digest ? { digest } : {}),
            status,
          };

  return { ok: true, request, identity };
};

export function toRegisterSkillRequest(
  request: ExternalSkillVersionRequest,
  identity: { name: string; organization: string },
): RegisterExternalSkillRequest;
export function toRegisterSkillRequest(
  request: UploadedSkillVersionRequest,
  identity: { name: string; organization: string },
): RegisterUploadedSkillRequest;
export function toRegisterSkillRequest(
  request: CreateSkillVersionRequest,
  identity: { name: string; organization: string },
): RegisterExternalSkillRequest | RegisterUploadedSkillRequest {
  return {
    ...request,
    name: identity.name,
    ...(identity.organization ? { organization: identity.organization } : {}),
  };
}

export const buildUploadedSkillVersionRequest = (
  fields: SkillRegistrationFields,
): BuiltSkillRegistration<UploadedSkillVersionRequest> => {
  const identity = parseSkillIdentityInput(fields.identity);
  if ('error' in identity) return { ok: false, error: identity.error };
  const status = creatableStatus(fields.status);
  if (!status) return { ok: false, error: 'status' };
  return { ok: true, request: { source: null, status }, identity };
};

export const formatSkillRegisterCli = ({
  sourceType,
  location,
  local,
  name,
  organization,
}: {
  sourceType?: SkillRegistrationSourceType;
  location: string;
  local: boolean;
  name?: string;
  organization?: string;
}) => {
  const command = sourceType === 'oci' ? 'oci' : sourceType === 'zip' ? 'zip' : 'git';
  const url = location.trim() || '<location>';
  const lines = local
    ? ['mlflow skills register', '  <directory>']
    : [`mlflow skills register ${command}`, `  --url ${url}`];
  if (name) lines.push(`  --name ${name}`);
  if (organization) lines.push(`  --organization ${organization}`);
  return lines.map((line, index) => (index < lines.length - 1 ? `${line} \\` : line)).join('\n');
};

export const formatSkillRegisterPython = ({
  sourceType,
  location,
  local,
  name,
  organization,
}: {
  sourceType?: SkillRegistrationSourceType;
  location: string;
  local: boolean;
  name?: string;
  organization?: string;
}) => {
  const url = location.trim() || '<location>';
  const identity = [name ? `name="${name}"` : '', organization ? `organization="${organization}"` : ''].filter(Boolean);
  const identityArgs = identity.length ? `, ${identity.join(', ')}` : '';
  if (local) {
    return `import mlflow\n\nmlflow.genai.register_skill(source="<directory>"${identityArgs})`;
  }
  if (sourceType === 'oci') {
    return `import mlflow\nfrom mlflow.entities.skill_source import OCISource\n\nmlflow.genai.register_skill(source=OCISource(image="${url}")${identityArgs})`;
  }
  if (sourceType === 'zip') {
    return `import mlflow\nfrom mlflow.entities.skill_source import ZipSource\n\nmlflow.genai.register_skill(source=ZipSource(url="${url}")${identityArgs})`;
  }
  return `import mlflow\nfrom mlflow.entities.skill_source import GitSource\n\nmlflow.genai.register_skill(source=GitSource(url="${url}")${identityArgs})`;
};
