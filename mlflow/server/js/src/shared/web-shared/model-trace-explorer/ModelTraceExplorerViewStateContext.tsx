import { isNil } from 'lodash';
import { createContext, useContext, useEffect, useMemo, useState } from 'react';

import type { ModelTrace, ModelTraceExplorerTab, ModelTraceSpanNode } from './ModelTrace.types';
import { getDefaultActiveTab, parseModelTraceToTree, searchTreeBySpanId } from './ModelTraceExplorer.utils';
import { getTimelineTreeNodesMap } from './timeline-tree/TimelineTree.utils';

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
  children,
}: {
  modelTrace: ModelTrace;
  initialActiveView?: 'summary' | 'detail';
  selectedSpanIdOnRender?: string;
  children: React.ReactNode;
  assessmentsPaneEnabled: boolean;
  initialAssessmentsPaneCollapsed?: boolean;
  isTraceInitialLoading?: boolean;
}) => {
  const rootNode = useMemo(() => parseModelTraceToTree(modelTrace), [modelTrace]);
  const nodeMap = useMemo(() => (rootNode ? getTimelineTreeNodesMap([rootNode]) : {}), [rootNode]);
  const selectedSpanOnRender = searchTreeBySpanId(rootNode, selectedSpanIdOnRender);
  const defaultSelectedNode = selectedSpanOnRender ?? rootNode ?? undefined;
  const hasAssessments = (defaultSelectedNode?.assessments?.length ?? 0) > 0;
  const hasInputsOrOutputs = !isNil(rootNode?.inputs) || !isNil(rootNode?.outputs);

  const [activeView, setActiveView] = useState<'summary' | 'detail'>(
    initialActiveView ?? (hasInputsOrOutputs ? 'summary' : 'detail'),
  );
  const [selectedNode, setSelectedNode] = useState<ModelTraceSpanNode | undefined>(defaultSelectedNode);
  const defaultActiveTab = getDefaultActiveTab(selectedNode);
  const [activeTab, setActiveTab] = useState<ModelTraceExplorerTab>(defaultActiveTab);
  const [showTimelineTreeGantt, setShowTimelineTreeGantt] = useState(false);
  const [assessmentsPaneExpanded, setAssessmentsPaneExpanded] = useState(
    !initialAssessmentsPaneCollapsed && hasAssessments,
  );

  useEffect(() => {
    const defaultActiveTab = getDefaultActiveTab(selectedNode);
    setActiveTab(defaultActiveTab);
  }, [selectedNode]);

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
      showTimelineTreeGantt,
      setShowTimelineTreeGantt,
      assessmentsPaneExpanded,
      setAssessmentsPaneExpanded,
      assessmentsPaneEnabled,
      isTraceInitialLoading,
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
    ],
  );

  return (
    <ModelTraceExplorerViewStateContext.Provider value={value}>{children}</ModelTraceExplorerViewStateContext.Provider>
  );
};
