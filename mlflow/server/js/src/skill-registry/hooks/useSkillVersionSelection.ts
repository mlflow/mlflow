import { useEffect } from 'react';

import { SkillStatus, type Skill } from '../types';
import { isNotFoundError, resolveDefaultSkillVersion } from '../utils';
import { useSelectedSkillVersion } from './useSelectedSkillVersion';
import { useSkillVersionQuery, useSkillVersionsQuery } from './useSkillVersionsQuery';

/**
 * Loads the version list and resolves the version selected in the URL. A selected version outside the loaded
 * page is fetched on its own; a deleted or missing one is reported as missing rather than as an error.
 */
export const useSkillVersionSelection = (name: string, organization: string, skill: Skill | undefined) => {
  const [selectedVersion, setSelectedVersion] = useSelectedSkillVersion();
  const versionsQuery = useSkillVersionsQuery(name, organization);
  const { data: versions, isLoading: versionsLoading } = versionsQuery;

  const fromList = versions?.find((version) => version.version === selectedVersion);
  const shouldFetch = selectedVersion != null && !fromList && !versionsLoading;
  const fetched = useSkillVersionQuery(name, organization, selectedVersion, shouldFetch);
  const fetchSettled = shouldFetch && !fetched.isLoading;

  const isMissing = fetchSettled && (isNotFoundError(fetched.error) || fetched.data?.status === SkillStatus.DELETED);
  const currentVersion = isMissing ? undefined : (fromList ?? fetched.data);
  const error = fetchSettled && !isMissing ? (fetched.error ?? undefined) : undefined;
  const isLoading =
    selectedVersion != null && !currentVersion && !isMissing && !error && (versionsLoading || fetched.isLoading);

  // Pick a default version once the skill (and, without latest_version, the list) is known.
  useEffect(() => {
    if (selectedVersion != null || !skill) return;
    if (skill.latest_version == null && versionsLoading) return;
    const next = resolveDefaultSkillVersion(skill, versions);
    if (next != null) setSelectedVersion(next);
  }, [selectedVersion, skill, versions, versionsLoading, setSelectedVersion]);

  return {
    versionsQuery,
    selectedVersion,
    setSelectedVersion,
    currentVersion,
    isLoading,
    isMissing,
    error,
  };
};
