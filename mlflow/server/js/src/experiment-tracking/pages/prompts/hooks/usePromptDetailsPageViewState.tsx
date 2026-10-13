import { useCallback, useReducer } from 'react';
import { PromptVersionsTableMode } from '../utils';
import { first } from 'lodash';
import type { RegisteredPromptDetailsResponse } from '../types';
import { useSelectedPromptVersion } from './useSelectedPromptVersion';

const promptDetailsViewStateReducer = (
  state: {
    mode: PromptVersionsTableMode;
    comparedVersion?: string;
  },
  action:
    | { type: 'setPreviewMode' }
    | { type: 'setCompareMode'; comparedVersion?: string }
    | { type: 'setTracesMode' }
    | { type: 'setComparedVersion'; comparedVersion?: string },
) => {
  if (action.type === 'setPreviewMode') {
    return { ...state, mode: PromptVersionsTableMode.PREVIEW };
  }
  if (action.type === 'setCompareMode') {
    return {
      ...state,
      mode: PromptVersionsTableMode.COMPARE,
      comparedVersion: action.comparedVersion,
    };
  }
  if (action.type === 'setTracesMode') {
    return { ...state, mode: PromptVersionsTableMode.TRACES };
  }
  if (action.type === 'setComparedVersion') {
    return { ...state, comparedVersion: action.comparedVersion };
  }
  return state;
};

export const usePromptDetailsPageViewState = (promptDetailsData?: RegisteredPromptDetailsResponse) => {
  const [selectedVersion, setSelectedVersion] = useSelectedPromptVersion();
  const [viewState, dispatchViewMode] = useReducer(promptDetailsViewStateReducer, {
    mode: PromptVersionsTableMode.PREVIEW,
  });

  const setPreviewMode = useCallback(
    (versionEntity?: { version: string }) => {
      const firstVersion = (versionEntity ?? first(promptDetailsData?.versions))?.version;
      setSelectedVersion(firstVersion);
      dispatchViewMode({ type: 'setPreviewMode' });
    },
    [promptDetailsData, setSelectedVersion],
  );
  const setComparedVersion = useCallback((comparedVersion: string) => {
    dispatchViewMode({ type: 'setComparedVersion', comparedVersion });
  }, []);

  const setCompareMode = useCallback(() => {
    const latestVersion = first(promptDetailsData?.versions)?.version;
    // Use the currently selected version as baseline (left side), or fall back to second version
    const baselineVersion = selectedVersion ?? promptDetailsData?.versions[1]?.version;
    // If baseline is already the latest, compare with the second version; otherwise compare with latest
    const comparedVersion = baselineVersion === latestVersion ? promptDetailsData?.versions[1]?.version : latestVersion;

    setSelectedVersion(baselineVersion);
    dispatchViewMode({ type: 'setCompareMode', comparedVersion });
  }, [promptDetailsData, selectedVersion, setSelectedVersion]);

  const setTracesMode = useCallback(
    (versionEntity?: { version: string }) => {
      const firstVersion = (versionEntity ?? first(promptDetailsData?.versions))?.version;
      setSelectedVersion(firstVersion);
      dispatchViewMode({ type: 'setTracesMode' });
    },
    [promptDetailsData, setSelectedVersion],
  );

  const switchSides = useCallback(() => {
    if (!selectedVersion || !viewState.comparedVersion) {
      return;
    }

    const comparedVersion = viewState.comparedVersion;
    const tempSelectedVersion = selectedVersion;
    setSelectedVersion(comparedVersion);
    setComparedVersion(tempSelectedVersion);
  }, [selectedVersion, setComparedVersion, setSelectedVersion, viewState.comparedVersion]);

  if (first(promptDetailsData?.versions) && viewState.mode === PromptVersionsTableMode.PREVIEW && !selectedVersion) {
    setPreviewMode(first(promptDetailsData?.versions));
  }

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
