import { createContext, useCallback, useContext, useMemo, useState } from 'react';

import type { ModelTraceExplorerRenderMode } from './ModelTrace.types';

export type TimelineTreeMetric = 'duration' | 'tokens' | 'cost';

export const DEFAULT_TIMELINE_TREE_METRICS: TimelineTreeMetric[] = ['duration', 'tokens', 'cost'];

export type ModelTraceExplorerPreferences = {
  renderMode: ModelTraceExplorerRenderMode;
  setRenderMode: (mode: ModelTraceExplorerRenderMode) => void;
  assessmentsPaneExpanded: boolean | undefined;
  setAssessmentsPaneExpanded: (expanded: boolean) => void;
  timelineTreeMetrics: TimelineTreeMetric[];
  setTimelineTreeMetrics: (metrics: TimelineTreeMetric[]) => void;
};

export const ModelTraceExplorerPreferencesContext: React.Context<ModelTraceExplorerPreferences> =
  createContext<ModelTraceExplorerPreferences>({
    renderMode: 'default',
    setRenderMode: () => {},
    assessmentsPaneExpanded: undefined,
    setAssessmentsPaneExpanded: () => {},
    timelineTreeMetrics: DEFAULT_TIMELINE_TREE_METRICS,
    setTimelineTreeMetrics: () => {},
  });

export const useModelTraceExplorerPreferences = (): ModelTraceExplorerPreferences => {
  return useContext(ModelTraceExplorerPreferencesContext);
};

export const ModelTraceExplorerPreferencesProvider = ({
  children,
  initialRenderMode = 'default',
}: {
  children: React.ReactNode;
  initialRenderMode?: ModelTraceExplorerRenderMode;
}): JSX.Element => {
  const [renderMode, setRenderMode] = useState<ModelTraceExplorerRenderMode>(initialRenderMode);
  const [assessmentsPaneExpanded, setAssessmentsPaneExpandedState] = useState<boolean | undefined>(undefined);
  const [timelineTreeMetrics, setTimelineTreeMetrics] = useState<TimelineTreeMetric[]>(DEFAULT_TIMELINE_TREE_METRICS);

  const setAssessmentsPaneExpanded = useCallback((expanded: boolean) => {
    setAssessmentsPaneExpandedState(expanded);
  }, []);

  const value = useMemo(
    () => ({
      renderMode,
      setRenderMode,
      assessmentsPaneExpanded,
      setAssessmentsPaneExpanded,
      timelineTreeMetrics,
      setTimelineTreeMetrics,
    }),
    [renderMode, assessmentsPaneExpanded, setAssessmentsPaneExpanded, timelineTreeMetrics],
  );

  return (
    <ModelTraceExplorerPreferencesContext.Provider value={value}>
      {children}
    </ModelTraceExplorerPreferencesContext.Provider>
  );
};
