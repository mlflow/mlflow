export enum SkillStatus {
  DRAFT = 'draft',
  ACTIVE = 'active',
  DEPRECATED = 'deprecated',
  DELETED = 'deleted',
}

export enum SkillAction {
  USE = 'USE',
  UPDATE = 'UPDATE',
  DELETE = 'DELETE',
  MANAGE = 'MANAGE',
}

export type SkillRequestSourceType = 'git' | 'oci' | 'zip';
export type SkillSourceType = SkillRequestSourceType | 'mlflow';

export interface RegistryIcon {
  src: string;
  sizes?: string[];
  mimeType?: string;
  theme?: string;
}

export interface SkillAlias {
  alias: string;
  version: number;
}

export type SkillTags = Record<string, string>;

export interface SkillAuditFields {
  created_by: string | null;
  last_updated_by: string | null;
  creation_timestamp: number | null;
  last_updated_timestamp: number | null;
}

export interface Skill extends SkillAuditFields {
  name: string;
  organization: string;
  description: string | null;
  icons: RegistryIcon[] | null;
  status: SkillStatus | null;
  latest_version: number | null;
  aliases: SkillAlias[];
  tags: SkillTags;
  allowed_actions?: SkillAction[];
}

export interface SkillVersion extends SkillAuditFields {
  name: string;
  version: number;
  organization: string;
  source_type: SkillSourceType | null;
  source: string | null;
  ref: string | null;
  subpath: string | null;
  digest: string | null;
  status: SkillStatus;
  aliases: string[];
  tags: SkillTags;
}

interface SkillVersionRequestBase {
  ref?: string | null;
  subpath?: string | null;
  digest?: string | null;
  status?: SkillStatus;
}

export interface ExternalSkillVersionRequest extends SkillVersionRequestBase {
  source: string;
  source_type?: SkillRequestSourceType | null;
}

export interface UploadedSkillVersionRequest extends SkillVersionRequestBase {
  source?: null;
  source_type?: never;
}

export type CreateSkillVersionRequest = ExternalSkillVersionRequest | UploadedSkillVersionRequest;

interface RegisterSkillIdentity {
  name: string;
  organization?: string;
}

export type RegisterExternalSkillRequest = ExternalSkillVersionRequest & RegisterSkillIdentity;
export type RegisterUploadedSkillRequest = UploadedSkillVersionRequest & RegisterSkillIdentity;
export type RegisterSkillRequest = RegisterExternalSkillRequest | RegisterUploadedSkillRequest;

export interface UpdateSkillRequest {
  description?: string | null;
  icons?: RegistryIcon[] | null;
}

export interface UpdateSkillVersionStatusRequest {
  status?: SkillStatus | null;
}

export interface SetSkillTagRequest {
  key: string;
  value: string;
}

export interface SetSkillAliasRequest {
  alias: string;
  version: number;
}

export interface SearchSkillsParams {
  filter_string?: string;
  max_results?: number;
  order_by?: string[];
  page_token?: string;
}

export type SearchSkillVersionsParams = SearchSkillsParams;

export interface SearchSkillsResponse {
  skills: Skill[];
  next_page_token: string | null;
}

export interface SearchSkillVersionsResponse {
  skill_versions: SkillVersion[];
  next_page_token: string | null;
}

export type GetSkillResponse = Skill;
export type GetSkillVersionResponse = SkillVersion;
export type RegisterSkillResponse = SkillVersion;
export type ResolveSkillAliasResponse = SkillVersion;
export type UpdateSkillResponse = Skill;
export type UpdateSkillVersionStatusResponse = SkillVersion;
export type SkillMutationResponse = Record<string, never>;
