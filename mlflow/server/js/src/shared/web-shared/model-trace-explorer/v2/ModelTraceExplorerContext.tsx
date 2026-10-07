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
   * Whether the MLflow Assistant panel is currently open. When set, the custom-view
   * drawer snaps to the left to sit beside the panel. Optional: hosts that dock the
   * drawer around the assistant themselves (e.g. AssistantAwareDrawer) can omit it.
   */
  isAssistantPanelOpen?: boolean;
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
  isAssistantPanelOpen?: boolean;
  isSearchVisible?: boolean;
}

export const ModelTraceExplorerContextProvider: React.FC<ModelTraceExplorerContextProviderProps> = ({
  children,
  renderExportTracesToDatasetsModal,
  renderAddToReviewQueueDropdown,
  DrawerComponent = Drawer,
  drawerWidth,
  isAssistantPanelOpen,
  isSearchVisible,
}) => {
  const value = useMemo(
    () => ({
      renderExportTracesToDatasetsModal,
      renderAddToReviewQueueDropdown,
      DrawerComponent,
      drawerWidth,
      isAssistantPanelOpen,
      isSearchVisible,
    }),
    [
      renderExportTracesToDatasetsModal,
      renderAddToReviewQueueDropdown,
      DrawerComponent,
      drawerWidth,
      isAssistantPanelOpen,
      isSearchVisible,
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
