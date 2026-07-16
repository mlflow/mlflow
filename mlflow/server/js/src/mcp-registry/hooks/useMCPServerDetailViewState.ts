import { useCallback, useReducer } from 'react';
import { first } from 'lodash';
import type { MCPServerVersion } from '../types';

export enum MCPServerDetailViewMode {
  PREVIEW = 'preview',
  COMPARE = 'compare',
}

interface State {
  mode: MCPServerDetailViewMode;
  selectedVersion?: string;
  comparedVersion?: string;
}

type ViewAction =
  | { type: 'setPreviewMode' }
  | { type: 'setCompareMode'; selectedVersion?: string; comparedVersion?: string }
  | { type: 'setComparedVersion'; comparedVersion?: string }
  | { type: 'setSelectedVersion'; version?: string }
  | { type: 'switchSides' };

const viewStateReducer = (state: State, action: ViewAction): State => {
  switch (action.type) {
    case 'setPreviewMode':
      return { ...state, mode: MCPServerDetailViewMode.PREVIEW, comparedVersion: undefined };
    case 'setCompareMode':
      return {
        ...state,
        mode: MCPServerDetailViewMode.COMPARE,
        selectedVersion: action.selectedVersion,
        comparedVersion: action.comparedVersion,
      };
    case 'setComparedVersion':
      if (action.comparedVersion === state.selectedVersion) {
        return { ...state, comparedVersion: action.comparedVersion, selectedVersion: state.comparedVersion };
      }
      return { ...state, comparedVersion: action.comparedVersion };
    case 'setSelectedVersion':
      if (action.version === state.comparedVersion) {
        return { ...state, selectedVersion: action.version, comparedVersion: state.selectedVersion };
      }
      return { ...state, selectedVersion: action.version };
    case 'switchSides':
      return { ...state, selectedVersion: state.comparedVersion, comparedVersion: state.selectedVersion };
    default:
      return state;
  }
};

export const useMCPServerDetailViewState = (versions?: MCPServerVersion[]) => {
  const [state, dispatch] = useReducer(viewStateReducer, {
    mode: MCPServerDetailViewMode.PREVIEW,
    selectedVersion: undefined,
    comparedVersion: undefined,
  });

  const setSelectedVersion = useCallback((version?: string) => {
    dispatch({ type: 'setSelectedVersion', version });
  }, []);

  const setPreviewMode = useCallback(() => {
    dispatch({ type: 'setPreviewMode' });
  }, []);

  const setCompareMode = useCallback(() => {
    const latestVersion = first(versions)?.version;
    const baselineVersion = state.selectedVersion ?? versions?.[1]?.version;
    const comparedVersion = baselineVersion === latestVersion ? versions?.[1]?.version : latestVersion;

    dispatch({ type: 'setCompareMode', selectedVersion: baselineVersion, comparedVersion });
  }, [versions, state.selectedVersion]);

  const setComparedVersion = useCallback((comparedVersion: string) => {
    dispatch({ type: 'setComparedVersion', comparedVersion });
  }, []);

  const switchSides = useCallback(() => {
    dispatch({ type: 'switchSides' });
  }, []);

  return {
    viewState: { mode: state.mode, comparedVersion: state.comparedVersion },
    selectedVersion: state.selectedVersion,
    setSelectedVersion,
    setPreviewMode,
    setCompareMode,
    setComparedVersion,
    switchSides,
  };
};
