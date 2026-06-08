import { createContext, useCallback, useContext, useMemo, useState } from 'react';

import type { ModelTraceExplorerActiveView, ModelTraceExplorerRenderMode } from './ModelTrace.types';

export type ModelTraceExplorerPreferences = {
  renderMode: ModelTraceExplorerRenderMode;
  setRenderMode: (mode: ModelTraceExplorerRenderMode) => void;
  assessmentsPaneExpanded: boolean | undefined;
  setAssessmentsPaneExpanded: (expanded: boolean) => void;
  activeView: ModelTraceExplorerActiveView | undefined;
  setActiveView: (view: ModelTraceExplorerActiveView) => void;
};

export const ModelTraceExplorerPreferencesContext = createContext<ModelTraceExplorerPreferences>({
  renderMode: 'default',
  setRenderMode: () => {},
  assessmentsPaneExpanded: undefined,
  setAssessmentsPaneExpanded: () => {},
  activeView: undefined,
  setActiveView: () => {},
});

export const useModelTraceExplorerPreferences = () => {
  return useContext(ModelTraceExplorerPreferencesContext);
};

export const ModelTraceExplorerPreferencesProvider = ({
  children,
  initialRenderMode = 'default',
}: {
  children: React.ReactNode;
  initialRenderMode?: ModelTraceExplorerRenderMode;
}) => {
  const [renderMode, setRenderMode] = useState<ModelTraceExplorerRenderMode>(initialRenderMode);
  const [assessmentsPaneExpanded, setAssessmentsPaneExpandedState] = useState<boolean | undefined>(undefined);
  const [activeView, setActiveViewState] = useState<ModelTraceExplorerActiveView | undefined>(undefined);

  const setAssessmentsPaneExpanded = useCallback((expanded: boolean) => {
    setAssessmentsPaneExpandedState(expanded);
  }, []);

  const setActiveView = useCallback((view: ModelTraceExplorerActiveView) => {
    setActiveViewState(view);
  }, []);

  const value = useMemo(
    () => ({
      renderMode,
      setRenderMode,
      assessmentsPaneExpanded,
      setAssessmentsPaneExpanded,
      activeView,
      setActiveView,
    }),
    [renderMode, assessmentsPaneExpanded, setAssessmentsPaneExpanded, activeView, setActiveView],
  );

  return (
    <ModelTraceExplorerPreferencesContext.Provider value={value}>
      {children}
    </ModelTraceExplorerPreferencesContext.Provider>
  );
};
