// prettier-ignore
import {
  getExperimentApi,
  setCompareExperiments,
  setExperimentTagApi,
} from '../../actions';
import { GetExperimentsContextProvider } from '../../components/experiment-page/contexts/GetExperimentsContext';
import { ExperimentView } from '../../components/experiment-page/ExperimentView';
import { IndexedDBInitializationContextProvider } from '@mlflow/mlflow/src/experiment-tracking/components/contexts/IndexedDBInitializationContext';

/**
 * Concrete actions for GetExperiments context
 */
const getExperimentActions = {
  setExperimentTagApi,
  getExperimentApi,
  setCompareExperiments,
};

const ExperimentRunsPage = () => (
  <IndexedDBInitializationContextProvider>
    <GetExperimentsContextProvider actions={getExperimentActions}>
      <ExperimentView showHeader={false} />
    </GetExperimentsContextProvider>
  </IndexedDBInitializationContextProvider>
);

export default ExperimentRunsPage;
