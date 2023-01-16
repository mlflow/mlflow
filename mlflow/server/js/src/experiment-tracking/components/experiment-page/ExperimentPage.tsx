// prettier-ignore
import {
  getExperimentApi,
  setCompareExperiments,
  setExperimentTagApi,
} from '../../actions';
import { GetExperimentsContextProvider } from './contexts/GetExperimentsContext';
import { ExperimentView } from './ExperimentView';

/**
 * Concrete actions for GetExperiments context
 */
const getExperimentActions = {
  setExperimentTagApi,
  getExperimentApi,
  setCompareExperiments,
};

/**
 * Main entry point for the experiment page. This component
 * provides underlying structure with context containing
 * concrete versions of store actions.
 */
export const ExperimentPage = () => (
  <GetExperimentsContextProvider actions={getExperimentActions}>
    <ExperimentView />
  </GetExperimentsContextProvider>
);

export default ExperimentPage;
