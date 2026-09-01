import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

import type { ModelTrace, ModelTraceExplorerTab, ModelTraceExplorerView, ModelTraceSpanNode } from './ModelTrace.types';
import {
  getDefaultActiveTab,
  parseModelTraceToTreeWithMultipleRoots,
  searchTreeBySpanId,
} from './ModelTraceExplorer.utils';
import { getTimelineTreeNodesMap } from './timeline-tree/TimelineTree.utils';
import { useModelTraceExplorerPreferences } from './ModelTraceExplorerPreferencesContext';

type PaneSizeRatios = {
  detailsSidebar: number;
  detailsPane: number;
  graphPane: number;
};

// Default ratios of pane sizes in the model trace explorer.
const getDefaultPaneSizeRatios = (): PaneSizeRatios => ({
  // Details sidebar
  detailsSidebar: 0.7,
  // Details pane (based on the window width)
  detailsPane: window.innerWidth <= 768 ? 0.33 : 0.25,
  // Graph view pane — balanced with details pane
  graphPane: 0.5,
});

export type ModelTraceExplorerViewState = {
  rootNode: ModelTraceSpanNode | null;
  nodeMap: Record<string, ModelTraceSpanNode>;
  activeView: ModelTraceExplorerView;
  setActiveView: (view: ModelTraceExplorerView) => void;
  selectedNode: ModelTraceSpanNode | undefined;
  setSelectedNode: (node: ModelTraceSpanNode | undefined) => void;
  activeTab: ModelTraceExplorerTab;
  setActiveTab: (tab: ModelTraceExplorerTab) => void;
  showGraph: boolean;
  setShowGraph: (show: boolean) => void;
  showTimelineTreeGantt: boolean;
  setShowTimelineTreeGantt: (show: boolean) => void;
  assessmentsPaneExpanded: boolean;
  setAssessmentsPaneExpanded: (expanded: boolean) => void;
  isTraceInitialLoading?: boolean;
  assessmentsPaneEnabled: boolean;
  updatePaneSizeRatios: (sizes: Partial<PaneSizeRatios>) => void;
  getPaneSizeRatios: () => PaneSizeRatios;
  readOnly?: boolean;
  // NB: There can be multiple top-level spans in the trace when it is in-progress. They are not
  // root spans, but used as a tentative roots until the trace is complete.
  topLevelNodes: ModelTraceSpanNode[];
  subscribeToHighlightEvent: (assessmentId: string, callback: () => void) => () => void;
  highlightAssessment: (assessmentId: string) => void;
  refreshTrace?: () => Promise<void>;
  isRefreshingTrace?: boolean;
};

export const ModelTraceExplorerViewStateContext: React.Context<ModelTraceExplorerViewState> =
  createContext<ModelTraceExplorerViewState>({
    rootNode: null,
    nodeMap: {},
    activeView: 'detail',
    setActiveView: () => {},
    selectedNode: undefined,
    setSelectedNode: () => {},
    activeTab: 'content',
    setActiveTab: () => {},
    showGraph: false,
    setShowGraph: () => {},
    showTimelineTreeGantt: false,
    setShowTimelineTreeGantt: () => {},
    assessmentsPaneExpanded: false,
    setAssessmentsPaneExpanded: () => {},
    isTraceInitialLoading: false,
    assessmentsPaneEnabled: true,
    updatePaneSizeRatios: () => {},
    getPaneSizeRatios: () => getDefaultPaneSizeRatios(),
    readOnly: false,
    topLevelNodes: [],
    subscribeToHighlightEvent: () => () => {},
    highlightAssessment: () => {},
    refreshTrace: undefined,
    isRefreshingTrace: false,
  });

export const useModelTraceExplorerViewState = (): ModelTraceExplorerViewState => {
  return useContext(ModelTraceExplorerViewStateContext);
};

export const ModelTraceExplorerViewStateProvider = ({
  modelTrace,
  selectedSpanIdOnRender,
  // assessments pane is disabled if
  // the trace doesn't exist in the backend
  // (i.e. if the traceinfo fetch fails)
  assessmentsPaneEnabled,
  initialAssessmentsPaneCollapsed,
  isTraceInitialLoading = false,
  children,
  readOnly = false,
  // Seeds the initial tree-vs-timeline toggle; the AI Gateway opts into timeline (true).
  // All other callers omit this prop and keep the default tree view (false).
  initialShowTimelineTreeGantt = false,
  refreshTrace,
  isRefreshingTrace = false,
}: {
  modelTrace: ModelTrace;
  selectedSpanIdOnRender?: string;
  children: React.ReactNode;
  assessmentsPaneEnabled: boolean;
  initialAssessmentsPaneCollapsed?: boolean | 'force-open';
  isTraceInitialLoading?: boolean;
  readOnly?: boolean;
  initialShowTimelineTreeGantt?: boolean;
  refreshTrace?: () => Promise<void>;
  isRefreshingTrace?: boolean;
}): JSX.Element => {
  const topLevelNodes = useMemo(() => parseModelTraceToTreeWithMultipleRoots(modelTrace), [modelTrace]);
  const rootNode = topLevelNodes.length === 1 ? topLevelNodes[0] : null;

  const nodeMap = useMemo(() => getTimelineTreeNodesMap(topLevelNodes), [topLevelNodes]);
  const selectedSpanOnRender = searchTreeBySpanId(rootNode, selectedSpanIdOnRender);
  const defaultSelectedNode = selectedSpanOnRender ?? rootNode ?? undefined;
  const hasAssessments = (defaultSelectedNode?.assessments?.length ?? 0) > 0;

  const preferences = useModelTraceExplorerPreferences();

  // Stores the pane size rations. Uses mutable ref instead of useState to avoid unnecessary rerenders,
  // as the pane size ratios are used only during the initial render.
  const paneSizeRatiosRef = useRef<PaneSizeRatios>(getDefaultPaneSizeRatios());

  // The getter function to get the current pane size ratios
  const getPaneSizeRatios = useCallback(() => paneSizeRatiosRef.current, []);

  const updatePaneSizeRatios = useCallback((sizes: Partial<PaneSizeRatios>) => {
    paneSizeRatiosRef.current = {
      ...paneSizeRatiosRef.current,
      ...sizes,
    };
  }, []);

  const [activeView, setActiveView] = useState<ModelTraceExplorerView>('detail');

  const [selectedNode, setSelectedNode] = useState<ModelTraceSpanNode | undefined>(defaultSelectedNode);
  const defaultActiveTab = getDefaultActiveTab(selectedNode);
  const [activeTab, setActiveTab] = useState<ModelTraceExplorerTab>(defaultActiveTab);
  const [showGraph, setShowGraph] = useState(false);
  const [showTimelineTreeGantt, setShowTimelineTreeGantt] = useState(initialShowTimelineTreeGantt);

  useEffect(() => {
    if (!selectedNode) {
      if (rootNode) setSelectedNode(rootNode);
      return;
    }
    setSelectedNode(nodeMap[String(selectedNode.key)] ?? rootNode ?? topLevelNodes[0]);
  }, [nodeMap, rootNode, selectedNode, topLevelNodes]);
  const [assessmentsPaneExpanded, setAssessmentsPaneExpandedInternal] = useState(() => {
    if (preferences.assessmentsPaneExpanded !== undefined) {
      return preferences.assessmentsPaneExpanded;
    }
    return (
      (initialAssessmentsPaneCollapsed === false && hasAssessments) || initialAssessmentsPaneCollapsed === 'force-open'
    );
  });

  const setAssessmentsPaneExpanded = useCallback(
    (expanded: boolean) => {
      setAssessmentsPaneExpandedInternal(expanded);
      preferences.setAssessmentsPaneExpanded(expanded);
    },
    [preferences],
  );

  const pendingHighlightRef = useRef<string | null>(null);
  const highlightListenersRef = useRef<Map<string, Set<() => void>>>(new Map());

  const subscribeToHighlightEvent = useCallback((assessmentId: string, callback: () => void) => {
    let listeners = highlightListenersRef.current.get(assessmentId);
    if (!listeners) {
      listeners = new Set();
      highlightListenersRef.current.set(assessmentId, listeners);
    }
    listeners.add(callback);

    if (pendingHighlightRef.current === assessmentId) {
      callback();
      pendingHighlightRef.current = null;
    }

    return () => {
      listeners.delete(callback);
    };
  }, []);

  const highlightAssessment = useCallback((assessmentId: string) => {
    const listeners = highlightListenersRef.current.get(assessmentId);
    if (listeners && listeners.size > 0) {
      listeners.forEach((cb) => cb());
    } else {
      pendingHighlightRef.current = assessmentId;
    }
  }, []);

  useEffect(() => {
    const defaultActiveTab = getDefaultActiveTab(selectedNode);
    setActiveTab(defaultActiveTab);
  }, [selectedNode]);

  const value = useMemo(
    () => ({
      rootNode,
      nodeMap,
      activeView,
      setActiveView,
      activeTab,
      setActiveTab,
      selectedNode,
      setSelectedNode,
      showGraph,
      setShowGraph,
      showTimelineTreeGantt,
      setShowTimelineTreeGantt,
      assessmentsPaneExpanded: !readOnly && assessmentsPaneExpanded,
      setAssessmentsPaneExpanded,
      assessmentsPaneEnabled,
      isTraceInitialLoading,
      updatePaneSizeRatios,
      getPaneSizeRatios,
      readOnly,
      topLevelNodes,
      subscribeToHighlightEvent,
      highlightAssessment,
      refreshTrace,
      isRefreshingTrace,
    }),
    [
      activeView,
      setActiveView,
      nodeMap,
      activeTab,
      rootNode,
      selectedNode,
      showGraph,
      showTimelineTreeGantt,
      setShowTimelineTreeGantt,
      assessmentsPaneExpanded,
      setAssessmentsPaneExpanded,
      assessmentsPaneEnabled,
      isTraceInitialLoading,
      updatePaneSizeRatios,
      getPaneSizeRatios,
      readOnly,
      topLevelNodes,
      refreshTrace,
      isRefreshingTrace,
      subscribeToHighlightEvent,
      highlightAssessment,
    ],
  );

  return (
    <ModelTraceExplorerViewStateContext.Provider value={value}>{children}</ModelTraceExplorerViewStateContext.Provider>
  );
};
