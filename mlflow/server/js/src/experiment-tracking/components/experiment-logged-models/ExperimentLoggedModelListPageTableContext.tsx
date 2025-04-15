import { createContext, useContext, useMemo } from 'react';

type ExperimentLoggedModelListPageTableContextType = {
  moreResultsAvailable?: boolean;
  isLoadingMore?: boolean;
  loadMoreResults?: () => void;
};

const ExperimentLoggedModelListPageTableContext = createContext<{
  moreResultsAvailable?: boolean;
  isLoadingMore?: boolean;
  loadMoreResults?: () => void;
}>({});

export const ExperimentLoggedModelListPageTableContextProvider = ({
  loadMoreResults,
  moreResultsAvailable,
  isLoadingMore,
  children,
}: React.PropsWithChildren<ExperimentLoggedModelListPageTableContextType>) => {
  const contextValue = useMemo(
    () => ({
      moreResultsAvailable,
      loadMoreResults,
      isLoadingMore,
    }),
    [moreResultsAvailable, loadMoreResults, isLoadingMore],
  );

  return (
    <ExperimentLoggedModelListPageTableContext.Provider value={contextValue}>
      {children}
    </ExperimentLoggedModelListPageTableContext.Provider>
  );
};

export const useExperimentLoggedModelListPageTableContext = () => useContext(ExperimentLoggedModelListPageTableContext);
