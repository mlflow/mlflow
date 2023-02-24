import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { Redirect } from 'react-router';
import { PageWrapper, Skeleton } from '@databricks/design-system';
import ExperimentListView from './ExperimentListView';
import { getExperiments } from '../reducers/Reducers';
import { NoExperimentView } from './NoExperimentView';
import Utils from '../../common/utils/Utils';
import Routes from '../routes';

// Lazy load experiment page in order to promote bundle splitting
const ExperimentPage = React.lazy(() => import('./experiment-page/ExperimentPage'));

export const getFirstActiveExperiment = (experiments) => {
  const sorted = experiments.concat().sort(Utils.compareExperiments);
  return sorted.find((e) => e.lifecycle_stage === 'active');
};

class HomeView extends Component {
  static propTypes = {
    experiments: PropTypes.arrayOf(PropTypes.object),
    experimentIds: PropTypes.arrayOf(PropTypes.string),
    compareExperiments: PropTypes.bool,
  };

  render() {
    const { experimentIds, experiments, compareExperiments } = this.props;
    const hasExperiments = experimentIds?.length > 0;

    if (experimentIds === undefined) {
      const firstExp = getFirstActiveExperiment(experiments);
      if (firstExp) {
        return <Redirect to={Routes.getExperimentPageRoute(firstExp.experiment_id)} />;
      }
    }

    if (process.env.HIDE_EXPERIMENT_LIST === 'true') {
      return (
        <>
          {hasExperiments ? (
            <PageWrapper css={{ height: `100%`, paddingTop: 16, width: '100%' }}>
              <React.Suspense fallback={<Skeleton />}>
                <ExperimentPage
                  experimentIds={experimentIds}
                  compareExperiments={compareExperiments}
                />
              </React.Suspense>
            </PageWrapper>
          ) : (
            <NoExperimentView />
          )}
        </>
      );
    }
    return (
      <div css={{ display: 'flex', height: 'calc(100% - 60px)' }}>
        <div css={{ height: '100%', paddingTop: 24, display: 'flex' }}>
          <ExperimentListView activeExperimentIds={experimentIds || []} experiments={experiments} />
        </div>
        <PageWrapper css={{ height: '100%', flex: '1', paddingTop: 24 }}>
          {hasExperiments ? (
            <React.Suspense fallback={<Skeleton />}>
              <ExperimentPage
                experimentIds={experimentIds}
                compareExperiments={compareExperiments}
              />
            </React.Suspense>
          ) : (
            <NoExperimentView />
          )}
        </PageWrapper>
      </div>
    );
  }
}

const mapStateToProps = (state) => {
  const experiments = getExperiments(state);
  return { experiments };
};

export default connect(mapStateToProps)(HomeView);
