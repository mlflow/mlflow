import { useCallback } from 'react';
import {
  type ArtifactRequestOptions,
  type getArtifactBytesContent,
  getArtifactContent,
  getArtifactLocationUrl,
  getLoggedModelArtifactLocationUrl,
} from '../../../../common/utils/ArtifactUtils';
import type { KeyValueEntity } from '../../../../common/types';
import { fetchArtifactWithPresignedUrl } from '../../../utils/PresignedArtifactUtils';

export type FetchArtifactParams = {
  experimentId?: string;
  runUuid: string;
  path: string;
  isLoggedModelsMode?: boolean;
  loggedModelId?: string;
  entityTags?: Partial<KeyValueEntity>[];
  artifactRootUri?: string;
};

type GetArtifactContentFn = (
  artifactLocation: string,
  options?: ArtifactRequestOptions,
) => ReturnType<typeof getArtifactContent | typeof getArtifactBytesContent>;

// Internal util, strips leading slash from the path if it exists
const normalizeArtifactPath = (path: string) => (path.startsWith('/') ? path.substring(1) : path);

// Internal util that generates the artifact location URL for the workspace API
const getWorkspaceArtifactLocationUrl = (params: FetchArtifactParams) => {
  const { runUuid, path, isLoggedModelsMode, loggedModelId } = params;
  if (isLoggedModelsMode && loggedModelId) {
    return getLoggedModelArtifactLocationUrl(path, loggedModelId);
  }
  return getArtifactLocationUrl(path, runUuid);
};

/**
 * A function that provides a unified function for fetching artifacts, either from the workspace API or SPN API.
 */
export const fetchArtifactUnified = (
  params: FetchArtifactParams,
  getArtifactDataFn: GetArtifactContentFn = getArtifactContent,
) => {
  const workspaceAPIArtifactLocation = getWorkspaceArtifactLocationUrl(params);

  return fetchArtifactWithPresignedUrl(params, workspaceAPIArtifactLocation, getArtifactDataFn);
};

export type FetchArtifactUnifiedFn<T = string> = (
  params: FetchArtifactParams,
  getArtifactDataFn: GetArtifactContentFn,
) => Promise<T>;
