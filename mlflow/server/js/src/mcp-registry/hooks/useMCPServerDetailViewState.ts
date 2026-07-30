import { useCallback, useEffect, useReducer } from 'react';
import { first } from 'lodash';
import type { MCPServerDetailViewState, MCPServerVersion } from '../types';
import { MCPServerDetailViewMode } from '../types';

type ViewAction =
  | { type: 'setPreviewMode' }
  | { type: 'setCompareMode'; comparedVersion?: string }
  | { type: 'setComparedVersion'; comparedVersion?: string }
  | { type: 'syncVersions'; versions: string[]; selectedVersion?: string };

const viewStateReducer = (state: MCPServerDetailViewState, action: ViewAction): MCPServerDetailViewState => {
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
    case 'syncVersions': {
      let next = state;
      if (next.mode === MCPServerDetailViewMode.COMPARE && action.versions.length < 2) {
        next = { ...next, mode: MCPServerDetailViewMode.PREVIEW, comparedVersion: undefined };
      }
      if (next.comparedVersion && !action.versions.includes(next.comparedVersion)) {
        const fallback = action.versions.find((v) => v !== action.selectedVersion);
        next = { ...next, comparedVersion: fallback };
      }
      return next;
    }
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

  useEffect(() => {
    if (!versions?.length) return;
    const resolved = selectedVersion ?? first(versions)?.version;
    if (!selectedVersion) {
      setSelectedVersion(resolved);
    }
    const versionStrings = versions.map((v) => v.version);
    dispatch({ type: 'syncVersions', versions: versionStrings, selectedVersion: resolved });
  }, [versions, selectedVersion, setSelectedVersion]);

  return {
    viewState: { mode: state.mode, comparedVersion: state.comparedVersion },
    setPreviewMode,
    setCompareMode,
    setComparedVersion,
    switchSides,
  };
};
