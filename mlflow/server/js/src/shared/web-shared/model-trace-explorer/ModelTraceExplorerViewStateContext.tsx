import { isNil } from 'lodash';
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

import type { ModelTrace, ModelTraceExplorerTab, ModelTraceSpanNode } from './ModelTrace.types';
import {
  getDefaultActiveTab,
  parseModelTraceToTreeWithMultipleRoots,
  searchTreeBySpanId,
} from './ModelTraceExplorer.utils';
import { getTimelineTreeNodesMap } from './timeline-tree/TimelineTree.utils';

type PaneSizeRatios = {
  summarySidebar: number;
  detailsSidebar: number;
  detailsPane: number;
};

// Default ratios of pane sizes in the model trace explorer.
const getDefaultPaneSizeRatios = (): PaneSizeRatios => ({
  // Summary sidebar
  summarySidebar: 0.75,
  // Details sidebar
  detailsSidebar: 0.7,
  // Details pane (based on the window width)
  detailsPane: window.innerWidth <= 768 ? 0.33 : 0.25,
});

export type ModelTraceExplorerViewState = {
  rootNode: ModelTraceSpanNode | null;
  nodeMap: Record<string, ModelTraceSpanNode>;
  activeView: 'summary' | 'detail';
  setActiveView: (view: 'summary' | 'detail') => void;
  selectedNode: ModelTraceSpanNode | undefined;
  setSelectedNode: (node: ModelTraceSpanNode | undefined) => void;
  activeTab: ModelTraceExplorerTab;
  setActiveTab: (tab: ModelTraceExplorerTab) => void;
  showTimelineTreeGantt: boolean;
  setShowTimelineTreeGantt: (show: boolean) => void;
  assessmentsPaneExpanded: boolean;
  setAssessmentsPaneExpanded: (expanded: boolean) => void;
  isTraceInitialLoading?: boolean;
  assessmentsPaneEnabled: boolean;
  isInComparisonView: boolean;
  updatePaneSizeRatios: (sizes: Partial<PaneSizeRatios>) => void;
  getPaneSizeRatios: () => PaneSizeRatios;
  readOnly?: boolean;
  // NB: There can be multiple top-level spans in the trace when it is in-progress. They are not
  // root spans, but used as a tentative roots until the trace is complete.
  topLevelNodes: ModelTraceSpanNode[];
};

export const ModelTraceExplorerViewStateContext = createContext<ModelTraceExplorerViewState>({
  rootNode: null,
  nodeMap: {},
  activeView: 'summary',
  setActiveView: () => {},
  selectedNode: undefined,
  setSelectedNode: () => {},
  activeTab: 'content',
  setActiveTab: () => {},
  showTimelineTreeGantt: false,
  setShowTimelineTreeGantt: () => {},
  assessmentsPaneExpanded: false,
  setAssessmentsPaneExpanded: () => {},
  isTraceInitialLoading: false,
  assessmentsPaneEnabled: true,
  isInComparisonView: false,
  updatePaneSizeRatios: () => {},
  getPaneSizeRatios: () => getDefaultPaneSizeRatios(),
  readOnly: false,
  topLevelNodes: [],
});

export const useModelTraceExplorerViewState = () => {
  return useContext(ModelTraceExplorerViewStateContext);
};

export const ModelTraceExplorerViewStateProvider = ({
  modelTrace,
  initialActiveView,
  selectedSpanIdOnRender,
  // assessments pane is disabled if
  // the trace doesn't exist in the backend
  // (i.e. if the traceinfo fetch fails)
  assessmentsPaneEnabled,
  initialAssessmentsPaneCollapsed,
  isTraceInitialLoading = false,
  isInComparisonView = false,
  children,
  readOnly = false,
}: {
  modelTrace: ModelTrace;
  initialActiveView?: 'summary' | 'detail';
  selectedSpanIdOnRender?: string;
  children: React.ReactNode;
  assessmentsPaneEnabled: boolean;
  initialAssessmentsPaneCollapsed?: boolean | 'force-open';
  isTraceInitialLoading?: boolean;
  isInComparisonView?: boolean;
  readOnly?: boolean;
}) => {
  const topLevelNodes = useMemo(() => parseModelTraceToTreeWithMultipleRoots(modelTrace), [modelTrace]);
  const rootNode = topLevelNodes.length === 1 ? topLevelNodes[0] : null;

  const nodeMap = useMemo(() => (rootNode ? getTimelineTreeNodesMap([rootNode]) : {}), [rootNode]);
  const selectedSpanOnRender = searchTreeBySpanId(rootNode, selectedSpanIdOnRender);
  const defaultSelectedNode = selectedSpanOnRender ?? rootNode ?? undefined;
  const hasAssessments = (defaultSelectedNode?.assessments?.length ?? 0) > 0;
  const hasInputsOrOutputs = !isNil(rootNode?.inputs) || !isNil(rootNode?.outputs);

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

  const [activeView, setActiveView] = useState<'summary' | 'detail'>(() => {
    // Default to detail view when rootNode is null
    if (!rootNode) {
      return 'detail';
    }
    return initialActiveView ?? (hasInputsOrOutputs ? 'summary' : 'detail');
  });

  const [selectedNode, setSelectedNode] = useState<ModelTraceSpanNode | undefined>(defaultSelectedNode);
  const defaultActiveTab = getDefaultActiveTab(selectedNode);
  const [activeTab, setActiveTab] = useState<ModelTraceExplorerTab>(defaultActiveTab);
  const [showTimelineTreeGantt, setShowTimelineTreeGantt] = useState(false);
  const [assessmentsPaneExpanded, setAssessmentsPaneExpanded] = useState(
    !initialAssessmentsPaneCollapsed || initialAssessmentsPaneCollapsed === 'force-open',
  );

  useEffect(() => {
    const defaultActiveTab = getDefaultActiveTab(selectedNode);
    setActiveTab(defaultActiveTab);
  }, [selectedNode]);

  // Switch to detail view if currently on summary and rootNode becomes null
  useEffect(() => {
    if (!rootNode && activeView === 'summary') {
      setActiveView('detail');
    }
  }, [rootNode, activeView]);

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
      showTimelineTreeGantt,
      setShowTimelineTreeGantt,
      assessmentsPaneExpanded: !readOnly && assessmentsPaneExpanded,
      setAssessmentsPaneExpanded,
      assessmentsPaneEnabled,
      isTraceInitialLoading,
      isInComparisonView,
      updatePaneSizeRatios,
      getPaneSizeRatios,
      readOnly,
      topLevelNodes,
    }),
    [
      activeView,
      nodeMap,
      activeTab,
      rootNode,
      selectedNode,
      showTimelineTreeGantt,
      setShowTimelineTreeGantt,
      assessmentsPaneExpanded,
      setAssessmentsPaneExpanded,
      assessmentsPaneEnabled,
      isTraceInitialLoading,
      isInComparisonView,
      updatePaneSizeRatios,
      getPaneSizeRatios,
      readOnly,
      topLevelNodes,
    ],
  );

  return (
    <ModelTraceExplorerViewStateContext.Provider value={value}>{children}</ModelTraceExplorerViewStateContext.Provider>
  );
};
