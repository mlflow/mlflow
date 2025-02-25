/* eslint-disable react-hooks/rules-of-hooks */
import { useEffect, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { ReduxState, type ThunkDispatch } from '../../redux-types';
import { getExperimentApi, searchExperimentsApi, setCompareExperiments, setExperimentTagApi } from '../actions';
import { Navigate } from '../../common/utils/RoutingUtils';
import { getUUID } from '../../common/utils/ActionUtils';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import ExperimentListView from './ExperimentListView';
import { useExperimentIds } from './experiment-page/hooks/useExperimentIds';
import { values } from 'lodash';
import { Spinner, useDesignSystemTheme } from '@databricks/design-system';
import { GetExperimentsContextProvider } from './experiment-page/contexts/GetExperimentsContext';
import { ExperimentView } from './experiment-page/ExperimentView';
import { NoExperimentView } from './NoExperimentView';
import Utils from '../../common/utils/Utils';
import { ExperimentEntity } from '../types';
import Routes from '../routes';
import { ExperimentPage } from './experiment-page/ExperimentPage';

const getExperimentActions = {
  setExperimentTagApi,
  getExperimentApi,
  setCompareExperiments,
};

const getFirstActiveExperiment = (experiments: ExperimentEntity[]) => {
  const sorted = [...experiments].sort(Utils.compareExperiments);
  return sorted.find(({ lifecycleStage }) => lifecycleStage === 'active');
};

const HomePage = () => {
  const dispatch = useDispatch<ThunkDispatch>();
  const { theme } = useDesignSystemTheme();
  const searchRequestId = useRef(getUUID());

  const experimentIds = useExperimentIds();
  const experiments = useSelector((state: ReduxState) => values(state.entities.experimentsById));

  const hasExperiments = experiments.length > 0;

  useEffect(() => {
    dispatch(searchExperimentsApi(searchRequestId.current));
  }, [dispatch]);

  // If no experiments are currently selected, navigate to the first one
  if (!experimentIds.length) {
    const firstExp = getFirstActiveExperiment(experiments);
    if (firstExp) {
      return <Navigate to={Routes.getExperimentPageRoute(firstExp.experimentId)} replace />;
    }
  }

  const loadingState = (
    <div css={{ height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <Spinner size="large" />
    </div>
  );

  return (
    <RequestStateWrapper requestIds={[searchRequestId.current]} customSpinner={loadingState}>
      <div css={{ display: 'flex', height: 'calc(100% - 60px)' }}>
        {/* Left sidebar containing experiment list */}
        <div css={{ height: '100%', paddingTop: 24, display: 'flex' }}>
          <ExperimentListView activeExperimentIds={experimentIds || []} experiments={experiments} />
        </div>

        {/* Main content with the experiment view */}
        <div css={{ height: '100%', flex: 1, padding: theme.spacing.md, paddingTop: theme.spacing.lg }}>
          <GetExperimentsContextProvider actions={getExperimentActions}>
            {hasExperiments ? <ExperimentView /> : <NoExperimentView />}
          </GetExperimentsContextProvider>
        </div>
      </div>
    </RequestStateWrapper>
  );
};

export default HomePage;
