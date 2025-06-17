import ExperimentListView from './ExperimentListView';
import { useExperimentListQuery } from './experiment-page/hooks/useExperimentListQuery';
import { Spinner } from '@databricks/design-system';

const HomePage = () => {
  const { data: experiments, error, isLoading } = useExperimentListQuery();

  const loadingState = (
    <div css={{ height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <Spinner size="large" />
    </div>
  );

  if (isLoading) {
    return loadingState;
  }

  return <ExperimentListView experiments={experiments || []} error={error} />;
};

export default HomePage;
