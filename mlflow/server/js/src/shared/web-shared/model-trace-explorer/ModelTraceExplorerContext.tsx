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

export type DrawerComponentType = {
  Root: (props: {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    modal?: boolean;
    children: ReactNode;
  }) => React.ReactElement;
  Content: (props: Drawer.DrawerContentProps) => React.ReactElement;
};

export interface ModelTraceExplorerContextValue {
  renderExportTracesToDatasetsModal?: (params: RenderExportTracesToDatasetsModalParams) => React.ReactNode;
  DrawerComponent: DrawerComponentType;
}

const ModelTraceExplorerContext = createContext<ModelTraceExplorerContextValue>({
  renderExportTracesToDatasetsModal: () => null,
  DrawerComponent: Drawer,
});

interface ModelTraceExplorerContextProviderProps {
  children: React.ReactNode;
  renderExportTracesToDatasetsModal?: (params: RenderExportTracesToDatasetsModalParams) => React.ReactNode;
  DrawerComponent?: DrawerComponentType;
}

export const ModelTraceExplorerContextProvider: React.FC<ModelTraceExplorerContextProviderProps> = ({
  children,
  renderExportTracesToDatasetsModal,
  DrawerComponent = Drawer,
}) => {
  const value = useMemo(
    () => ({
      renderExportTracesToDatasetsModal,
      DrawerComponent,
    }),
    [renderExportTracesToDatasetsModal, DrawerComponent],
  );

  return <ModelTraceExplorerContext.Provider value={value}>{children}</ModelTraceExplorerContext.Provider>;
};

export const useModelTraceExplorerContext = () => {
  return useContext(ModelTraceExplorerContext);
};
