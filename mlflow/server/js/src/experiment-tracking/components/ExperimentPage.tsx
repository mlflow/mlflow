/* eslint-disable react-hooks/rules-of-hooks */
import { getExperimentApi, setCompareExperiments, setExperimentTagApi } from '../actions';
import { Navigate, useParams } from '../../common/utils/RoutingUtils';
import { useExperimentIds } from './experiment-page/hooks/useExperimentIds';
import { Spinner, useDesignSystemTheme } from '@databricks/design-system';
import { GetExperimentsContextProvider } from './experiment-page/contexts/GetExperimentsContext';
import { ExperimentView } from './experiment-page/ExperimentView';
import { NoExperimentView } from './NoExperimentView';
import Utils from '../../common/utils/Utils';
import { ExperimentEntity } from '../types';
import Routes from '../routes';
import { isExperimentLoggedModelsUIEnabled } from '../../common/utils/FeatureUtils';
import ExperimentPageTabs from '../pages/experiment-page-tabs/ExperimentPageTabs';
import { useExperimentListQuery } from './experiment-page/hooks/useExperimentListQuery';

const getExperimentActions = {
  setExperimentTagApi,
  getExperimentApi,
  setCompareExperiments,
};

const getFirstActiveExperiment = (experiments: ExperimentEntity[]) => {
  const sorted = [...experiments].sort(Utils.compareExperiments);
  return sorted.find(({ lifecycleStage }) => lifecycleStage === 'active');
};

const ExperimentPage = () => {
  const { theme } = useDesignSystemTheme();

  const { tabName } = useParams();
  const shouldRenderTabbedView = isExperimentLoggedModelsUIEnabled() && Boolean(tabName);

  const experimentIds = useExperimentIds();

  const { data: experiments, isLoading } = useExperimentListQuery();

  if (isLoading) {
    return (
      <div css={{ height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Spinner size="large" />
      </div>
    );
  } else if (!experiments) {
    throw new Error('No experiments found'); // FIXME
  }

  const hasExperiments = experiments.length > 0;

  // If no experiments are currently selected, navigate to the first one
  if (!experimentIds.length) {
    const firstExp = getFirstActiveExperiment(experiments);
    if (firstExp) {
      return <Navigate to={Routes.getExperimentPageRoute(firstExp.experimentId)} replace />;
    }
  }

  return (
    <div css={{ display: 'flex', height: 'calc(100% - 60px)' }}>
      {shouldRenderTabbedView && <ExperimentPageTabs />}
      {!shouldRenderTabbedView && (
        // Main content with the experiment view
        <div css={{ height: '100%', flex: 1, padding: theme.spacing.md, paddingTop: theme.spacing.lg }}>
          <GetExperimentsContextProvider actions={getExperimentActions}>
            {hasExperiments ? <ExperimentView /> : <NoExperimentView />}
          </GetExperimentsContextProvider>
        </div>
      )}
    </div>
  );
};

export default ExperimentPage;
