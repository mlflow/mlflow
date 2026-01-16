import React, { createContext, useContext, useMemo } from 'react';

import type { ModelTraceInfoV3 } from './ModelTrace.types';

export interface RenderExportTracesToDatasetsModalParams {
  selectedTraceInfos: ModelTraceInfoV3[];
  experimentId: string;
  visible: boolean;
  setVisible: (visible: boolean) => void;
}

export interface ModelTraceExplorerContextValue {
  renderExportTracesToDatasetsModal?: (params: RenderExportTracesToDatasetsModalParams) => React.ReactNode;
}

const ModelTraceExplorerContext = createContext<ModelTraceExplorerContextValue>({
  renderExportTracesToDatasetsModal: () => null,
});

interface ModelTraceExplorerContextProviderProps {
  children: React.ReactNode;
  renderExportTracesToDatasetsModal?: (params: RenderExportTracesToDatasetsModalParams) => React.ReactNode;
}

export const ModelTraceExplorerContextProvider: React.FC<ModelTraceExplorerContextProviderProps> = ({
  children,
  renderExportTracesToDatasetsModal,
}) => {
  const value = useMemo(
    () => ({
      renderExportTracesToDatasetsModal,
    }),
    [renderExportTracesToDatasetsModal],
  );

  return <ModelTraceExplorerContext.Provider value={value}>{children}</ModelTraceExplorerContext.Provider>;
};

export const useModelTraceExplorerContext = () => {
  return useContext(ModelTraceExplorerContext);
};
