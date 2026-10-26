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
  children: React.ReactNode;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  popoverAlign?: 'start' | 'end';
}

export type DrawerComponentType = {
  Root: (props: {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    modal?: boolean;
    children: ReactNode;
  }) => React.ReactElement;
  Content: (props: Drawer.DrawerContentProps & { isFullscreen?: boolean }) => React.ReactElement;
};

export interface AddToDatasetAction {
  openModal: () => void;
}

export interface ModelTraceExplorerContextValue {
  renderExportTracesToDatasetsModal?: (params: RenderExportTracesToDatasetsModalParams) => React.ReactNode;
  renderAddToReviewQueueDropdown?: React.ComponentType<RenderAddToReviewQueueDropdownParams>;
  DrawerComponent: DrawerComponentType;
  /** When set (e.g. by the evaluation review drawer), content can show "Add to dataset" that calls openModal */
  addToDatasetAction?: AddToDatasetAction;
  drawerWidth?: string | number;
  /** Optional boundary used to contain the drawer and its auxiliary UI (new eval runs UI only). */
  getDrawerPortalContainer?: () => HTMLElement;
  /** Opens the trace drawer at the full size of its viewport or scoped container (new eval runs UI only). */
  initiallyFullscreen?: boolean;
}

const ModelTraceExplorerContext = createContext<ModelTraceExplorerContextValue>({
  renderExportTracesToDatasetsModal: () => null,
  DrawerComponent: Drawer,
  addToDatasetAction: undefined,
});

interface ModelTraceExplorerContextProviderProps {
  children: React.ReactNode;
  renderExportTracesToDatasetsModal?: (params: RenderExportTracesToDatasetsModalParams) => React.ReactNode;
  renderAddToReviewQueueDropdown?: React.ComponentType<RenderAddToReviewQueueDropdownParams>;
  DrawerComponent?: DrawerComponentType;
  drawerWidth?: string | number;
  getDrawerPortalContainer?: () => HTMLElement;
  initiallyFullscreen?: boolean;
}

export const ModelTraceExplorerContextProvider: React.FC<ModelTraceExplorerContextProviderProps> = ({
  children,
  renderExportTracesToDatasetsModal,
  renderAddToReviewQueueDropdown,
  DrawerComponent = Drawer,
  drawerWidth,
  getDrawerPortalContainer,
  initiallyFullscreen,
}) => {
  const value = useMemo(
    () => ({
      renderExportTracesToDatasetsModal,
      renderAddToReviewQueueDropdown,
      DrawerComponent,
      drawerWidth,
      getDrawerPortalContainer,
      initiallyFullscreen,
    }),
    [
      renderExportTracesToDatasetsModal,
      renderAddToReviewQueueDropdown,
      DrawerComponent,
      drawerWidth,
      getDrawerPortalContainer,
      initiallyFullscreen,
    ],
  );

  return <ModelTraceExplorerContext.Provider value={value}>{children}</ModelTraceExplorerContext.Provider>;
};

/** Use inside the drawer to expose "Add to dataset" to trace content (e.g. next to Show assessments). */
export const ModelTraceExplorerAddToDatasetProvider: React.FC<{
  openModal: () => void;
  children: ReactNode;
}> = ({ openModal, children }) => {
  const parent = useContext(ModelTraceExplorerContext);
  const value = useMemo(() => ({ ...parent, addToDatasetAction: { openModal } }), [parent, openModal]);
  return <ModelTraceExplorerContext.Provider value={value}>{children}</ModelTraceExplorerContext.Provider>;
};

export const useModelTraceExplorerContext = () => {
  return useContext(ModelTraceExplorerContext);
};
