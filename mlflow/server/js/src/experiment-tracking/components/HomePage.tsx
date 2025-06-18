import ExperimentListView from './ExperimentListView';
import { useExperimentListQuery } from './experiment-page/hooks/useExperimentListQuery';
import { useSearchFilter } from './experiment-page/hooks/useSearchFilter';
import { Spinner } from '@databricks/design-system';

const HomePage = () => {
  const [searchFilter, setSearchFilter] = useSearchFilter();
  const {
    data: experiments,
    error,
    isLoading,
    hasNextPage,
    hasPreviousPage,
    onNextPage,
    onPreviousPage,
  } = useExperimentListQuery({ searchFilter });

  const loadingState = (
    <div css={{ height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <Spinner size="large" />
    </div>
  );

  if (isLoading) {
    return loadingState;
  }

  return (
    <ExperimentListView
      experiments={experiments || []}
      error={error}
      searchFilter={searchFilter}
      setSearchFilter={setSearchFilter}
      cursorPaginationProps={{
        hasNextPage,
        hasPreviousPage,
        onNextPage,
        onPreviousPage,
      }}
    />
  );
};

export default HomePage;
