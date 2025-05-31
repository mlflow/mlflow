import ExperimentListView from './ExperimentListView';
import { useExperimentIds } from './experiment-page/hooks/useExperimentIds';
import { useExperimentListQuery } from './experiment-page/hooks/useExperimentListQuery';
import { Spinner } from '@databricks/design-system';

const HomePage = () => {
  const experimentIds = useExperimentIds();

  const {
    data: experiments,
    isLoading,
    onNextPage,
    onPreviousPage,
    hasNextPage,
    hasPreviousPage,
  } = useExperimentListQuery();

  const loadingState = (
    <div css={{ height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <Spinner size="large" />
    </div>
  );

  if (isLoading) {
    return loadingState;
  } else if (!experiments) {
    throw new Error('No experiments found'); // FIXME
  } else {
    return (
      <>
        <ExperimentListView activeExperimentIds={experimentIds || []} experiments={experiments} />
        <div>
          <button onClick={onPreviousPage} disabled={!hasPreviousPage}>
            Previous Page
          </button>
          <button onClick={onNextPage} disabled={!hasNextPage}>
            Next Page
          </button>
        </div>
      </>
    );
  }
};

export default HomePage;
