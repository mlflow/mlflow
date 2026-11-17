import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

import type {
  ModelTrace,
  ModelTraceComparisonRole,
  ModelTraceExplorerTab,
  ModelTraceExplorerView,
  ModelTraceSpanNode,
} from './ModelTrace.types';
import {
  buildModelTraceComparisonForest,
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

const EMPTY_COMPARISON_NODE_MAP: Record<string, ModelTraceSpanNode> = {};
const EMPTY_COMPARISON_NODE_ROLE_MAP: Record<string, ModelTraceComparisonRole> = {};

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
  hideAssessments: boolean;
  // NB: There can be multiple top-level spans in the trace when it is in-progress. They are not
  // root spans, but used as a tentative roots until the trace is complete.
  topLevelNodes: ModelTraceSpanNode[];
  subscribeToHighlightEvent: (assessmentId: string, callback: () => void) => () => void;
  highlightAssessment: (assessmentId: string) => void;
  refreshTrace?: () => Promise<void>;
  isRefreshingTrace?: boolean;
  comparisonRole?: ModelTraceComparisonRole;
  comparisonNodeMap: Record<string, ModelTraceSpanNode>;
  comparisonNodeRoleMap: Record<string, ModelTraceComparisonRole>;
  selectedComparisonNode?: ModelTraceSpanNode;
  selectedNodeComparisonRole?: ModelTraceComparisonRole;
  selectedSpanIdOnRenderDisplayId?: string;
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
    hideAssessments: false,
    topLevelNodes: [],
    subscribeToHighlightEvent: () => () => {},
    highlightAssessment: () => {},
    refreshTrace: undefined,
    isRefreshingTrace: false,
    comparisonRole: undefined,
    comparisonNodeMap: {},
    comparisonNodeRoleMap: {},
    selectedComparisonNode: undefined,
    selectedNodeComparisonRole: undefined,
    selectedSpanIdOnRenderDisplayId: undefined,
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
  hideAssessments = false,
  // Seeds the initial tree-vs-timeline toggle; the AI Gateway opts into timeline (true).
  // All other callers omit this prop and keep the default tree view (false).
  initialShowTimelineTreeGantt = false,
  refreshTrace,
  isRefreshingTrace = false,
  comparisonModelTrace,
  comparisonRole,
}: {
  modelTrace: ModelTrace;
  selectedSpanIdOnRender?: string;
  children: React.ReactNode;
  assessmentsPaneEnabled: boolean;
  initialAssessmentsPaneCollapsed?: boolean | 'force-open';
  isTraceInitialLoading?: boolean;
  readOnly?: boolean;
  hideAssessments?: boolean;
  initialShowTimelineTreeGantt?: boolean;
  refreshTrace?: () => Promise<void>;
  isRefreshingTrace?: boolean;
  comparisonModelTrace?: ModelTrace;
  comparisonRole?: ModelTraceComparisonRole;
}): JSX.Element => {
  const modelTopLevelNodes = useMemo(() => parseModelTraceToTreeWithMultipleRoots(modelTrace), [modelTrace]);
  const comparisonTopLevelNodes = useMemo(
    () => (comparisonModelTrace ? parseModelTraceToTreeWithMultipleRoots(comparisonModelTrace) : []),
    [comparisonModelTrace],
  );
  // A comparison merges the baseline and candidate span trees into one tree of paired spans.
  const comparisonForest = useMemo(() => {
    if (!comparisonRole || !comparisonModelTrace) {
      return undefined;
    }
    const baselineNodes = comparisonRole === 'baseline' ? modelTopLevelNodes : comparisonTopLevelNodes;
    const candidateNodes = comparisonRole === 'candidate' ? modelTopLevelNodes : comparisonTopLevelNodes;
    return buildModelTraceComparisonForest(baselineNodes, candidateNodes);
  }, [comparisonModelTrace, comparisonRole, comparisonTopLevelNodes, modelTopLevelNodes]);
  const topLevelNodes = comparisonForest?.topLevelNodes ?? modelTopLevelNodes;
  const rootNode = topLevelNodes.length === 1 ? topLevelNodes[0] : null;
  const comparisonNodeMap = comparisonForest?.comparisonNodeMap ?? EMPTY_COMPARISON_NODE_MAP;
  const comparisonNodeRoleMap = comparisonForest?.comparisonNodeRoleMap ?? EMPTY_COMPARISON_NODE_ROLE_MAP;

  const nodeMap = useMemo(() => getTimelineTreeNodesMap(topLevelNodes), [topLevelNodes]);
  const selectedSpanOnRender =
    searchTreeBySpanId(rootNode, selectedSpanIdOnRender) ??
    (comparisonForest && selectedSpanIdOnRender
      ? (comparisonForest.sourceNodeToDisplayNodeMap[`baseline:${selectedSpanIdOnRender}`] ??
        comparisonForest.sourceNodeToDisplayNodeMap[`candidate:${selectedSpanIdOnRender}`])
      : undefined);
  const defaultSelectedNode = selectedSpanOnRender ?? rootNode ?? undefined;
  const selectedSpanIdOnRenderDisplayId = selectedSpanOnRender ? String(selectedSpanOnRender.key) : undefined;
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
  const selectedComparisonNode = selectedNode ? comparisonNodeMap[String(selectedNode.key)] : undefined;
  const selectedNodeComparisonRole = selectedNode ? comparisonNodeRoleMap[String(selectedNode.key)] : undefined;
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
    setActiveTab((currentTab) =>
      currentTab === 'links' && (selectedNode?.links?.length ?? 0) > 0 ? currentTab : defaultActiveTab,
    );
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
      hideAssessments,
      topLevelNodes,
      subscribeToHighlightEvent,
      highlightAssessment,
      refreshTrace,
      isRefreshingTrace,
      comparisonRole,
      comparisonNodeMap,
      comparisonNodeRoleMap,
      selectedComparisonNode,
      selectedNodeComparisonRole,
      selectedSpanIdOnRenderDisplayId,
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
      hideAssessments,
      topLevelNodes,
      refreshTrace,
      isRefreshingTrace,
      comparisonRole,
      comparisonNodeMap,
      comparisonNodeRoleMap,
      selectedComparisonNode,
      selectedNodeComparisonRole,
      selectedSpanIdOnRenderDisplayId,
      subscribeToHighlightEvent,
      highlightAssessment,
    ],
  );

  return (
    <ModelTraceExplorerViewStateContext.Provider value={value}>{children}</ModelTraceExplorerViewStateContext.Provider>
  );
};
