import ExperimentListView from './ExperimentListView';
import { useSearchFilter } from './experiment-page/hooks/useSearchFilter';

const HomePage = () => {
  const [searchFilter, setSearchFilter] = useSearchFilter();

  return <ExperimentListView searchFilter={searchFilter} setSearchFilter={setSearchFilter} />;
};

export default HomePage;
