import { useCallback, useReducer } from 'react';
import { first } from 'lodash';
import type { MCPServerVersion } from '../types';

export enum MCPServerDetailViewMode {
  PREVIEW = 'preview',
  COMPARE = 'compare',
}

interface State {
  mode: MCPServerDetailViewMode;
  comparedVersion?: string;
}

type ViewAction =
  | { type: 'setPreviewMode' }
  | { type: 'setCompareMode'; comparedVersion?: string }
  | { type: 'setComparedVersion'; comparedVersion?: string };

const viewStateReducer = (state: State, action: ViewAction): State => {
  switch (action.type) {
    case 'setPreviewMode':
      return { ...state, mode: MCPServerDetailViewMode.PREVIEW, comparedVersion: undefined };
    case 'setCompareMode':
      return {
        ...state,
        mode: MCPServerDetailViewMode.COMPARE,
        comparedVersion: action.comparedVersion,
      };
    case 'setComparedVersion':
      return { ...state, comparedVersion: action.comparedVersion };
    default:
      return state;
  }
};

export const useMCPServerDetailViewState = (
  versions: MCPServerVersion[] | undefined,
  selectedVersion: string | undefined,
  setSelectedVersion: (version: string | undefined) => void,
) => {
  const [state, dispatch] = useReducer(viewStateReducer, {
    mode: MCPServerDetailViewMode.PREVIEW,
    comparedVersion: undefined,
  });

  const setPreviewMode = useCallback(() => {
    dispatch({ type: 'setPreviewMode' });
  }, []);

  const setCompareMode = useCallback(() => {
    const latestVersion = first(versions)?.version;
    const baselineVersion = selectedVersion ?? versions?.[1]?.version;
    const comparedVersion = baselineVersion === latestVersion ? versions?.[1]?.version : latestVersion;

    setSelectedVersion(baselineVersion);
    dispatch({ type: 'setCompareMode', comparedVersion });
  }, [versions, selectedVersion, setSelectedVersion]);

  const setComparedVersion = useCallback((comparedVersion: string) => {
    dispatch({ type: 'setComparedVersion', comparedVersion });
  }, []);

  const switchSides = useCallback(() => {
    if (!selectedVersion || !state.comparedVersion) {
      return;
    }
    const tempSelected = selectedVersion;
    setSelectedVersion(state.comparedVersion);
    setComparedVersion(tempSelected);
  }, [selectedVersion, state.comparedVersion, setSelectedVersion, setComparedVersion]);

  if (versions?.length && !selectedVersion) {
    setSelectedVersion(first(versions)?.version);
  }
  if (state.mode === MCPServerDetailViewMode.COMPARE && versions && versions.length < 2) {
    dispatch({ type: 'setPreviewMode' });
  }
  if (state.comparedVersion && versions?.length && !versions.some((v) => v.version === state.comparedVersion)) {
    const fallback =
      first(versions)?.version === selectedVersion ? (versions[1]?.version ?? '') : (first(versions)?.version ?? '');
    dispatch({ type: 'setComparedVersion', comparedVersion: fallback });
  }

  return {
    viewState: { mode: state.mode, comparedVersion: state.comparedVersion },
    setPreviewMode,
    setCompareMode,
    setComparedVersion,
    switchSides,
  };
};
