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
    | { type: 'setTableMode' }
    | { type: 'switchSides' }
    | { type: 'setPreviewMode'; selectedVersion?: string }
    | { type: 'setCompareMode'; selectedVersion?: string; comparedVersion?: string }
    | { type: 'setSelectedVersion'; selectedVersion: string }
    | { type: 'setComparedVersion'; comparedVersion: string },
) => {
  if (action.type === 'setTableMode') {
    return { ...state, mode: PromptVersionsTableMode.TABLE };
  }
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

  const setTableMode = useCallback(() => {
    dispatchViewMode({ type: 'setTableMode' });
  }, []);
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
    // Last (highest) version will be the compared version
    const comparedVersion = first(promptDetailsData?.versions)?.version;
    // The one immediately before the last version will be the baseline version
    const baselineVersion = promptDetailsData?.versions[1]?.version;
    dispatchViewMode({ type: 'setCompareMode', selectedVersion: baselineVersion, comparedVersion });
  }, [promptDetailsData]);

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
    setTableMode,
    setPreviewMode,
    setCompareMode,
    switchSides,
    setSelectedVersion,
    setComparedVersion,
  };
};
