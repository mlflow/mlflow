// prettier-ignore
import {
  getExperimentApi,
  setCompareExperiments,
  setExperimentTagApi,
} from '../../actions';
import { GetExperimentsContextProvider } from '../../components/experiment-page/contexts/GetExperimentsContext';
import { ExperimentView } from '../../components/experiment-page/ExperimentView';

/**
 * Concrete actions for GetExperiments context
 */
const getExperimentActions = {
  setExperimentTagApi,
  getExperimentApi,
  setCompareExperiments,
};

const ExperimentRunsPage = () => (
  <GetExperimentsContextProvider actions={getExperimentActions}>
    <ExperimentView showHeader={false} />
  </GetExperimentsContextProvider>
);

export default ExperimentRunsPage;
