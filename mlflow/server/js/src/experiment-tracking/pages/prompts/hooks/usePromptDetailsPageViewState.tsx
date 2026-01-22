import { useCallback, useReducer } from 'react';
import { PromptVersionsTableMode } from '../utils';
import { first } from 'lodash';
import type { RegisteredPromptDetailsResponse } from '../types';

const promptDetailsViewStateReducer = (
  state: {
    mode: PromptVersionsTableMode;
    selectedVersion?: string;
    comparedVersion?: string;
  },
  action:
    | { type: 'switchSides' }
    | { type: 'setPreviewMode'; selectedVersion?: string }
    | { type: 'setCompareMode'; selectedVersion?: string; comparedVersion?: string }
    | { type: 'setTracesMode'; selectedVersion?: string }
    | { type: 'setSelectedVersion'; selectedVersion: string }
    | { type: 'setComparedVersion'; comparedVersion: string },
) => {
  if (action.type === 'switchSides') {
    return { ...state, selectedVersion: state.comparedVersion, comparedVersion: state.selectedVersion };
  }
  if (action.type === 'setPreviewMode') {
    return { ...state, mode: PromptVersionsTableMode.PREVIEW, selectedVersion: action.selectedVersion };
  }
  if (action.type === 'setCompareMode') {
    return {
      ...state,
      mode: PromptVersionsTableMode.COMPARE,
      selectedVersion: action.selectedVersion,
      comparedVersion: action.comparedVersion,
    };
  }
  if (action.type === 'setTracesMode') {
    return { ...state, mode: PromptVersionsTableMode.TRACES, selectedVersion: action.selectedVersion };
  }
  if (action.type === 'setSelectedVersion') {
    return { ...state, selectedVersion: action.selectedVersion };
  }
  if (action.type === 'setComparedVersion') {
    return { ...state, comparedVersion: action.comparedVersion };
  }
  return state;
};

export const usePromptDetailsPageViewState = (promptDetailsData?: RegisteredPromptDetailsResponse) => {
  const [viewState, dispatchViewMode] = useReducer(promptDetailsViewStateReducer, {
    mode: PromptVersionsTableMode.PREVIEW,
  });

  const setPreviewMode = useCallback(
    (versionEntity?: { version: string }) => {
      const firstVersion = (versionEntity ?? first(promptDetailsData?.versions))?.version;
      dispatchViewMode({ type: 'setPreviewMode', selectedVersion: firstVersion });
    },
    [promptDetailsData],
  );
  const setSelectedVersion = useCallback((selectedVersion: string) => {
    dispatchViewMode({ type: 'setSelectedVersion', selectedVersion });
  }, []);
  const setComparedVersion = useCallback((comparedVersion: string) => {
    dispatchViewMode({ type: 'setComparedVersion', comparedVersion });
  }, []);
  const setCompareMode = useCallback(() => {
    const latestVersion = first(promptDetailsData?.versions)?.version;
    // Use the currently selected version as baseline (left side), or fall back to second version
    const baselineVersion = viewState.selectedVersion ?? promptDetailsData?.versions[1]?.version;
    // If baseline is already the latest, compare with the second version; otherwise compare with latest
    const comparedVersion = baselineVersion === latestVersion ? promptDetailsData?.versions[1]?.version : latestVersion;
    dispatchViewMode({ type: 'setCompareMode', selectedVersion: baselineVersion, comparedVersion });
  }, [promptDetailsData, viewState.selectedVersion]);

  const setTracesMode = useCallback(
    (versionEntity?: { version: string }) => {
      const firstVersion = (versionEntity ?? first(promptDetailsData?.versions))?.version;
      dispatchViewMode({ type: 'setTracesMode', selectedVersion: firstVersion });
    },
    [promptDetailsData],
  );

  const switchSides = useCallback(() => dispatchViewMode({ type: 'switchSides' }), []);

  if (
    first(promptDetailsData?.versions) &&
    viewState.mode === PromptVersionsTableMode.PREVIEW &&
    !viewState.selectedVersion
  ) {
    setPreviewMode(first(promptDetailsData?.versions));
  }

  return {
    viewState,
    setPreviewMode,
    setCompareMode,
    setTracesMode,
    switchSides,
    setSelectedVersion,
    setComparedVersion,
  };
};
