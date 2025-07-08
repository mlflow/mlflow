import ExperimentListView from './ExperimentListView';
import { useSearchFilter } from './experiment-page/hooks/useSearchFilter';

const ExperimentListPage = () => {
  const [searchFilter, setSearchFilter] = useSearchFilter();

  return <ExperimentListView searchFilter={searchFilter} setSearchFilter={setSearchFilter} />;
};

export default ExperimentListPage;
