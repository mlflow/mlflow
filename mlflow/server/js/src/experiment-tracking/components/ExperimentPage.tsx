/* eslint-disable react-hooks/rules-of-hooks */
import { getExperimentApi, setCompareExperiments, setExperimentTagApi } from '../actions';
import { useParams } from '../../common/utils/RoutingUtils';
import { useDesignSystemTheme } from '@databricks/design-system';
import { GetExperimentsContextProvider } from './experiment-page/contexts/GetExperimentsContext';
import { ExperimentView } from './experiment-page/ExperimentView';
import ExperimentPageTabs from '../pages/experiment-page-tabs/ExperimentPageTabs';
import { shouldEnableExperimentPageChildRoutes } from '../../common/utils/FeatureUtils';

const getExperimentActions = {
  setExperimentTagApi,
  getExperimentApi,
  setCompareExperiments,
};

const ExperimentPage = () => {
  const { theme } = useDesignSystemTheme();

  const { tabName } = useParams();
  const shouldRenderTabbedView = shouldEnableExperimentPageChildRoutes() || Boolean(tabName);

  return (
    <div css={{ display: 'flex', height: '100%' }}>
      {shouldRenderTabbedView && <ExperimentPageTabs />}
      {!shouldRenderTabbedView && (
        // Main content with the experiment view
        <div css={{ height: '100%', flex: 1, padding: theme.spacing.md, width: '100%' }}>
          <GetExperimentsContextProvider actions={getExperimentActions}>
            <ExperimentView />
          </GetExperimentsContextProvider>
        </div>
      )}
    </div>
  );
};

export default ExperimentPage;
