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

export interface ModelTraceExplorerContextValue {
  renderExportTracesToDatasetsModal?: (params: RenderExportTracesToDatasetsModalParams) => React.ReactNode;
  renderAddToReviewQueueDropdown?: (params: RenderAddToReviewQueueDropdownParams) => React.ReactNode;
  DrawerComponent: DrawerComponentType;
  /** When set (e.g. by the evaluation review drawer), content can show "Add to dataset" that calls openModal */
  addToDatasetAction?: AddToDatasetAction;
  drawerWidth?: string | number;
}

const ModelTraceExplorerContext = createContext<ModelTraceExplorerContextValue>({
  renderExportTracesToDatasetsModal: () => null,
  renderAddToReviewQueueDropdown: () => null,
  DrawerComponent: Drawer,
  addToDatasetAction: undefined,
});

interface ModelTraceExplorerContextProviderProps {
  children: React.ReactNode;
  renderExportTracesToDatasetsModal?: (params: RenderExportTracesToDatasetsModalParams) => React.ReactNode;
  renderAddToReviewQueueDropdown?: (params: RenderAddToReviewQueueDropdownParams) => React.ReactNode;
  DrawerComponent?: DrawerComponentType;
  drawerWidth?: string | number;
}

export const ModelTraceExplorerContextProvider: React.FC<ModelTraceExplorerContextProviderProps> = ({
  children,
  renderExportTracesToDatasetsModal,
  renderAddToReviewQueueDropdown,
  DrawerComponent = Drawer,
  drawerWidth,
}) => {
  const value = useMemo(
    () => ({
      renderExportTracesToDatasetsModal,
      renderAddToReviewQueueDropdown,
      DrawerComponent,
      drawerWidth,
    }),
    [renderExportTracesToDatasetsModal, renderAddToReviewQueueDropdown, DrawerComponent, drawerWidth],
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
