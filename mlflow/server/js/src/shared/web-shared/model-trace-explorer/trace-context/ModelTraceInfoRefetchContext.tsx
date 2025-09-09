import { createContext, useContext } from 'react';

export interface ModelTraceInfoRefetchContextType {
  // the result of the refetch is not used, it
  // is meant to refresh the trace info before
  // it is passed into the ModelTraceExplorer component
  refetchTraceInfo: null | (() => Promise<any>);
}

export const ModelTraceInfoRefetchContext = createContext<ModelTraceInfoRefetchContextType>({
  refetchTraceInfo: null,
});

export const useModelTraceInfoRefetchContext = () => {
  return useContext(ModelTraceInfoRefetchContext);
};
