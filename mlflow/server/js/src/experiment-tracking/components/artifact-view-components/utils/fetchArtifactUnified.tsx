import { useCallback } from 'react';
import {
  type getArtifactBytesContent,
  getArtifactContent,
  getArtifactLocationUrl,
  getLoggedModelArtifactLocationUrl,
} from '../../../../common/utils/ArtifactUtils';
import { getArtifactProxyDownloadUrl, isEligibleArtifactProxyUri } from '../../../../common/utils/artifactProxy';
import type { KeyValueEntity } from '../../../../common/types';

type FetchArtifactParams = {
  experimentId?: string;
  runUuid: string;
  path: string;
  isLoggedModelsMode?: boolean;
  loggedModelId?: string;
  entityTags?: Partial<KeyValueEntity>[];
  /**
   * The entity's stored artifact URI. When it points at an MLflow artifact
   * proxy the UI can reach, artifacts are fetched from there directly instead
   * of through the tracking server.
   */
  artifactUri?: string;
  /**
   * Whether `artifactUri` is the artifact root of the entity being fetched.
   * In logged models mode the surrounding page usually only knows the *run's*
   * artifact root, which is a different root, so routing to it would read from
   * the wrong location. Set this only when the URI belongs to the logged model.
   */
  isArtifactUriForEntity?: boolean;
};

type GetArtifactContentFn = typeof getArtifactContent | typeof getArtifactBytesContent;

// Internal util, strips leading slash from the path if it exists
const normalizeArtifactPath = (path: string) => (path.startsWith('/') ? path.substring(1) : path);

// Internal util that generates the artifact location URL for the workspace API
const getWorkspaceArtifactLocationUrl = (params: FetchArtifactParams) => {
  const { runUuid, path, isLoggedModelsMode, loggedModelId, artifactUri, isArtifactUriForEntity } = params;
  const usingLoggedModel = Boolean(isLoggedModelsMode && loggedModelId);
  // The URI describes the requested entity's own artifact root unless we are
  // reading a logged model, where the caller must confirm it.
  const describesRequestedEntity = usingLoggedModel ? Boolean(isArtifactUriForEntity) : true;

  if (describesRequestedEntity && isEligibleArtifactProxyUri(artifactUri)) {
    return getArtifactProxyDownloadUrl(artifactUri, path);
  }
  if (usingLoggedModel && loggedModelId) {
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

  return getArtifactDataFn(workspaceAPIArtifactLocation);
};

export type FetchArtifactUnifiedFn<T = string> = (
  params: FetchArtifactParams,
  getArtifactDataFn: GetArtifactContentFn,
) => Promise<T>;
