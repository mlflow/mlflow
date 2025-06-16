/* eslint-disable react-hooks/rules-of-hooks */
import { getExperimentApi, setCompareExperiments, setExperimentTagApi } from '../actions';
import { useParams } from '../../common/utils/RoutingUtils';
import { useDesignSystemTheme } from '@databricks/design-system';
import { GetExperimentsContextProvider } from './experiment-page/contexts/GetExperimentsContext';
import { ExperimentView } from './experiment-page/ExperimentView';
import { isExperimentLoggedModelsUIEnabled } from '../../common/utils/FeatureUtils';
import ExperimentPageTabs from '../pages/experiment-page-tabs/ExperimentPageTabs';

const getExperimentActions = {
  setExperimentTagApi,
  getExperimentApi,
  setCompareExperiments,
};

const ExperimentPage = () => {
  const { theme } = useDesignSystemTheme();

  const { tabName } = useParams();
  const shouldRenderTabbedView = isExperimentLoggedModelsUIEnabled() && Boolean(tabName);

  return (
    <div css={{ display: 'flex', height: 'calc(100% - 60px)' }}>
      {shouldRenderTabbedView && <ExperimentPageTabs />}
      {!shouldRenderTabbedView && (
        // Main content with the experiment view
        <div css={{ height: '100%', flex: 1, padding: theme.spacing.md, paddingTop: theme.spacing.lg }}>
          <GetExperimentsContextProvider actions={getExperimentActions}>
            <ExperimentView />
          </GetExperimentsContextProvider>
        </div>
      )}
    </div>
  );
};

export default ExperimentPage;
