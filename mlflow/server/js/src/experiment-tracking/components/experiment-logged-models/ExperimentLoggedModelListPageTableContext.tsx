import { createContext, useContext, useMemo } from 'react';

type ExperimentLoggedModelListPageTableContextType = {
  moreResultsAvailable?: boolean;
  isLoadingMore?: boolean;
  loadMoreResults?: () => void;
  expandedGroups?: string[];
  onGroupToggle?: (groupId: string) => void;
};

const ExperimentLoggedModelListPageTableContext = createContext<ExperimentLoggedModelListPageTableContextType>({});

export const ExperimentLoggedModelListPageTableContextProvider = ({
  loadMoreResults,
  moreResultsAvailable,
  isLoadingMore,
  children,
  expandedGroups,
  onGroupToggle,
}: React.PropsWithChildren<ExperimentLoggedModelListPageTableContextType>) => {
  const contextValue = useMemo(
    () => ({
      moreResultsAvailable,
      loadMoreResults,
      isLoadingMore,
      expandedGroups,
      onGroupToggle,
    }),
    [moreResultsAvailable, loadMoreResults, isLoadingMore, expandedGroups, onGroupToggle],
  );

  return (
    <ExperimentLoggedModelListPageTableContext.Provider value={contextValue}>
      {children}
    </ExperimentLoggedModelListPageTableContext.Provider>
  );
};

export const useExperimentLoggedModelListPageTableContext = () => useContext(ExperimentLoggedModelListPageTableContext);
