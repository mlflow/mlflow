import React, { createContext, useContext, useMemo } from 'react';
import type { ReactNode } from 'react';

import { Drawer } from '@databricks/design-system';

import type { ModelTraceInfoV3 } from './ModelTrace.types';

export interface RenderExportTracesToDatasetsModalParams {
  selectedTraceInfos: ModelTraceInfoV3[];
  experimentId: string;
  visible: boolean;
  setVisible: (visible: boolean) => void;
}

export interface RenderAddToReviewQueueDropdownParams {
  selectedTraceInfos: ModelTraceInfoV3[];
  experimentId: string;
  /** The trigger the dropdown wraps in its Popover. */
  children: React.ReactNode;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  popoverAlign?: 'start' | 'end';
  /**
   * Closes the host trace-detail drawer, if the dropdown is rendered inside one.
   * The drawer is modal, so while it is open, Radix disables pointer events on
   * everything outside it, including success toasts portaled to the document body.
   */
  onCloseDrawer?: () => void;
}

export type DrawerComponentType = {
  Root: (props: {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    modal?: boolean;
    children: ReactNode;
  }) => React.ReactElement;
  Content: (props: Drawer.DrawerContentProps) => React.ReactElement;
};

export interface AddToDatasetAction {
  openModal: () => void;
}

export type ModelTraceExplorerDisplayMode = 'default' | 'custom';

export interface OpenTraceAssistantParams {
  prompt: string;
  traceInfo?: ModelTraceInfoV3;
}

export interface ModelTraceExplorerContextValue {
  renderExportTracesToDatasetsModal?: (params: RenderExportTracesToDatasetsModalParams) => React.ReactNode;
  /** Renders the review-queue trigger as a Popover that routes traces into a review queue. */
  renderAddToReviewQueueDropdown?: React.ComponentType<RenderAddToReviewQueueDropdownParams>;
  DrawerComponent: DrawerComponentType;
  /** When set, content can show an Add to dataset affordance that calls openModal. */
  addToDatasetAction?: AddToDatasetAction;
  rightPaneHeaderActions?: ReactNode;
  experimentId?: string;
  drawerWidth?: string | number;
  isSearchVisible?: boolean;
  traceExplorerDisplayMode?: ModelTraceExplorerDisplayMode;
  setTraceExplorerDisplayMode?: (mode: ModelTraceExplorerDisplayMode) => void;
  /**
   * Whether the Genie assistant panel is currently open. The shared drawer cannot
   * depend on product-specific assistant hooks, so MLflow injects this state.
   */
  isGeniePanelOpen?: boolean;
  openTraceAssistant?: (params: OpenTraceAssistantParams) => void;
  isTraceAssistantStreaming?: boolean;
}

const ModelTraceExplorerContext = createContext<ModelTraceExplorerContextValue>({
  renderExportTracesToDatasetsModal: () => null,
  DrawerComponent: Drawer,
  addToDatasetAction: undefined,
  isSearchVisible: false,
  traceExplorerDisplayMode: 'default',
});

interface ModelTraceExplorerContextProviderProps {
  children: React.ReactNode;
  renderExportTracesToDatasetsModal?: (params: RenderExportTracesToDatasetsModalParams) => React.ReactNode;
  renderAddToReviewQueueDropdown?: React.ComponentType<RenderAddToReviewQueueDropdownParams>;
  DrawerComponent?: DrawerComponentType;
  drawerWidth?: string | number;
  isGeniePanelOpen?: boolean;
  isSearchVisible?: boolean;
  openTraceAssistant?: (params: OpenTraceAssistantParams) => void;
  isTraceAssistantStreaming?: boolean;
}

export const ModelTraceExplorerContextProvider: React.FC<ModelTraceExplorerContextProviderProps> = ({
  children,
  renderExportTracesToDatasetsModal,
  renderAddToReviewQueueDropdown,
  DrawerComponent = Drawer,
  drawerWidth,
  isGeniePanelOpen,
  isSearchVisible,
  openTraceAssistant,
  isTraceAssistantStreaming,
}) => {
  const value = useMemo(
    () => ({
      renderExportTracesToDatasetsModal,
      renderAddToReviewQueueDropdown,
      DrawerComponent,
      drawerWidth,
      isGeniePanelOpen,
      isSearchVisible,
      openTraceAssistant,
      isTraceAssistantStreaming,
    }),
    [
      renderExportTracesToDatasetsModal,
      renderAddToReviewQueueDropdown,
      DrawerComponent,
      drawerWidth,
      isGeniePanelOpen,
      isSearchVisible,
      openTraceAssistant,
      isTraceAssistantStreaming,
    ],
  );

  return <ModelTraceExplorerContext.Provider value={value}>{children}</ModelTraceExplorerContext.Provider>;
};

/** Use inside the drawer to expose Add to dataset to trace content. */
export const ModelTraceExplorerAddToDatasetProvider: React.FC<{
  openModal: () => void;
  children: ReactNode;
}> = ({ openModal, children }) => {
  const parent = useContext(ModelTraceExplorerContext);
  const value = useMemo(() => ({ ...parent, addToDatasetAction: { openModal } }), [parent, openModal]);
  return <ModelTraceExplorerContext.Provider value={value}>{children}</ModelTraceExplorerContext.Provider>;
};

export const ModelTraceExplorerRightPaneHeaderActionsProvider: React.FC<{
  openAddToDatasetModal?: () => void;
  rightPaneHeaderActions?: ReactNode;
  experimentId?: string;
  isSearchVisible?: boolean;
  traceExplorerDisplayMode?: ModelTraceExplorerDisplayMode;
  setTraceExplorerDisplayMode?: (mode: ModelTraceExplorerDisplayMode) => void;
  children: ReactNode;
}> = ({
  openAddToDatasetModal,
  rightPaneHeaderActions,
  experimentId,
  isSearchVisible = false,
  traceExplorerDisplayMode = 'default',
  setTraceExplorerDisplayMode,
  children,
}) => {
  const parent = useContext(ModelTraceExplorerContext);
  const value = useMemo(
    () => ({
      ...parent,
      addToDatasetAction: openAddToDatasetModal ? { openModal: openAddToDatasetModal } : parent.addToDatasetAction,
      rightPaneHeaderActions,
      experimentId,
      isSearchVisible,
      traceExplorerDisplayMode,
      setTraceExplorerDisplayMode,
    }),
    [
      parent,
      openAddToDatasetModal,
      rightPaneHeaderActions,
      experimentId,
      isSearchVisible,
      traceExplorerDisplayMode,
      setTraceExplorerDisplayMode,
    ],
  );
  return <ModelTraceExplorerContext.Provider value={value}>{children}</ModelTraceExplorerContext.Provider>;
};

export const useModelTraceExplorerContext = (): ModelTraceExplorerContextValue => {
  return useContext(ModelTraceExplorerContext);
};
