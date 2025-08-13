import { useEffect } from 'react';
import { shouldEnableExperimentPageHeaderV2 } from '../../../common/utils/FeatureUtils';
// prettier-ignore
import {
  getExperimentApi,
  setCompareExperiments,
  setExperimentTagApi,
} from '../../actions';
import { GetExperimentsContextProvider } from '../../components/experiment-page/contexts/GetExperimentsContext';
import { ExperimentView } from '../../components/experiment-page/ExperimentView';
import { useNavigate, useParams } from '../../../common/utils/RoutingUtils';
import { useExperimentIds } from '../../components/experiment-page/hooks/useExperimentIds';
import Routes from '../../routes';

/**
 * Concrete actions for GetExperiments context
 */
const getExperimentActions = {
  setExperimentTagApi,
  getExperimentApi,
  setCompareExperiments,
};

const ExperimentRunsPage = () => {
  const navigate = useNavigate();
  const { experimentId } = useParams();
  useEffect(() => {
    if (experimentId && !shouldEnableExperimentPageHeaderV2()) {
      navigate(Routes.getExperimentPageRoute(experimentId));
    }
  }, [experimentId, navigate]);
  return (
    <GetExperimentsContextProvider actions={getExperimentActions}>
      <ExperimentView showHeader={false} />
    </GetExperimentsContextProvider>
  );
};

export default ExperimentRunsPage;
