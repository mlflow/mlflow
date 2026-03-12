import { createContext, useCallback, useContext, useMemo, useState } from 'react';

import type { ModelTraceExplorerRenderMode } from './ModelTrace.types';

export type ModelTraceExplorerPreferences = {
  renderMode: ModelTraceExplorerRenderMode;
  setRenderMode: (mode: ModelTraceExplorerRenderMode) => void;
  assessmentsPaneExpanded: boolean | undefined;
  setAssessmentsPaneExpanded: (expanded: boolean) => void;
  activeView: 'summary' | 'detail' | undefined;
  setActiveView: (view: 'summary' | 'detail') => void;
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

export const ModelTraceExplorerPreferencesProvider = ({ children }: { children: React.ReactNode }) => {
  const [renderMode, setRenderMode] = useState<ModelTraceExplorerRenderMode>('default');
  const [assessmentsPaneExpanded, setAssessmentsPaneExpandedState] = useState<boolean | undefined>(undefined);
  const [activeView, setActiveViewState] = useState<'summary' | 'detail' | undefined>(undefined);

  const setAssessmentsPaneExpanded = useCallback((expanded: boolean) => {
    setAssessmentsPaneExpandedState(expanded);
  }, []);

  const setActiveView = useCallback((view: 'summary' | 'detail') => {
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
