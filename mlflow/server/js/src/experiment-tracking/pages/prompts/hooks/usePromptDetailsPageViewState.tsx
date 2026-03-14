import { useCallback, useEffect } from 'react';
import {
  PromptVersionsTableMode,
  PROMPT_COMPARED_VERSION_QUERY_PARAM,
  PROMPT_VERSION_QUERY_PARAM,
  PROMPT_VIEW_MODE_QUERY_PARAM,
} from '../utils';
import { first } from 'lodash';
import type { RegisteredPromptDetailsResponse } from '../types';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';

const isValidMode = (value: string | null): value is PromptVersionsTableMode =>
  value !== null && Object.values<string>(PromptVersionsTableMode).includes(value);

export const usePromptDetailsPageViewState = (promptDetailsData?: RegisteredPromptDetailsResponse) => {
  const [searchParams, setSearchParams] = useSearchParams();

  const modeParam = searchParams.get(PROMPT_VIEW_MODE_QUERY_PARAM);
  const mode = isValidMode(modeParam) ? modeParam : PromptVersionsTableMode.PREVIEW;
  const comparedVersion = searchParams.get(PROMPT_COMPARED_VERSION_QUERY_PARAM) ?? undefined;
  const selectedVersion = searchParams.get(PROMPT_VERSION_QUERY_PARAM) ?? undefined;

  const viewState = { mode, comparedVersion };

  const setSelectedVersion = useCallback(
    (version: string | undefined) => {
      setSearchParams(
        (params) => {
          if (version === undefined) {
            params.delete(PROMPT_VERSION_QUERY_PARAM);
          } else {
            params.set(PROMPT_VERSION_QUERY_PARAM, version);
          }
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const setPreviewMode = useCallback(
    (versionEntity?: { version: string }) => {
      const firstVersion = (versionEntity ?? first(promptDetailsData?.versions))?.version;
      setSearchParams(
        (params) => {
          params.set(PROMPT_VIEW_MODE_QUERY_PARAM, PromptVersionsTableMode.PREVIEW);
          params.delete(PROMPT_COMPARED_VERSION_QUERY_PARAM);
          if (firstVersion) {
            params.set(PROMPT_VERSION_QUERY_PARAM, firstVersion);
          }
          return params;
        },
        { replace: true },
      );
    },
    [promptDetailsData, setSearchParams],
  );

  const setComparedVersion = useCallback(
    (comparedVersion: string) => {
      setSearchParams(
        (params) => {
          params.set(PROMPT_COMPARED_VERSION_QUERY_PARAM, comparedVersion);
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const setCompareMode = useCallback(() => {
    const latestVersion = first(promptDetailsData?.versions)?.version;
    // Use the currently selected version as baseline (left side), or fall back to second version
    const baselineVersion = selectedVersion ?? promptDetailsData?.versions[1]?.version;
    // If baseline is already the latest, compare with the second version; otherwise compare with latest
    const newComparedVersion =
      baselineVersion === latestVersion ? promptDetailsData?.versions[1]?.version : latestVersion;

    setSearchParams(
      (params) => {
        params.set(PROMPT_VIEW_MODE_QUERY_PARAM, PromptVersionsTableMode.COMPARE);
        if (baselineVersion) {
          params.set(PROMPT_VERSION_QUERY_PARAM, baselineVersion);
        }
        if (newComparedVersion) {
          params.set(PROMPT_COMPARED_VERSION_QUERY_PARAM, newComparedVersion);
        }
        return params;
      },
      { replace: true },
    );
  }, [promptDetailsData, selectedVersion, setSearchParams]);

  const setTracesMode = useCallback(
    (versionEntity?: { version: string }) => {
      const firstVersion = (versionEntity ?? first(promptDetailsData?.versions))?.version;
      setSearchParams(
        (params) => {
          params.set(PROMPT_VIEW_MODE_QUERY_PARAM, PromptVersionsTableMode.TRACES);
          params.delete(PROMPT_COMPARED_VERSION_QUERY_PARAM);
          if (firstVersion) {
            params.set(PROMPT_VERSION_QUERY_PARAM, firstVersion);
          }
          return params;
        },
        { replace: true },
      );
    },
    [promptDetailsData, setSearchParams],
  );

  const switchSides = useCallback(() => {
    if (!selectedVersion || !comparedVersion) {
      return;
    }

    setSearchParams(
      (params) => {
        params.set(PROMPT_VERSION_QUERY_PARAM, comparedVersion);
        params.set(PROMPT_COMPARED_VERSION_QUERY_PARAM, selectedVersion);
        return params;
      },
      { replace: true },
    );
  }, [selectedVersion, comparedVersion, setSearchParams]);

  useEffect(() => {
    if (first(promptDetailsData?.versions) && mode === PromptVersionsTableMode.PREVIEW && !selectedVersion) {
      setPreviewMode(first(promptDetailsData?.versions));
    }
  }, [promptDetailsData, mode, selectedVersion, setPreviewMode]);

  return {
    viewState,
    selectedVersion,
    setPreviewMode,
    setCompareMode,
    setTracesMode,
    switchSides,
    setSelectedVersion,
    setComparedVersion,
  };
};
