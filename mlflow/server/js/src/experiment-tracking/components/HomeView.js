import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { Redirect } from 'react-router';
import { Skeleton } from '@databricks/design-system';

const ExperimentListView = React.lazy(() => import('./ExperimentListView'));
import { getExperiments } from '../reducers/Reducers';
import { NoExperimentView } from './NoExperimentView';
import { PageContainer } from '../../common/components/PageContainer';
import Routes from '../routes';

// Lazy load experiment page in order to promote bundle splitting
const ExperimentPage = React.lazy(() => import('./experiment-page/ExperimentPage'));

// Experiments are returned from state sorted
export const getFirstActiveExperiment = (experiments) => {
  return experiments.find((e) => e.lifecycle_stage === 'active');
};

class HomeView extends Component {
  static propTypes = {
    experiments: PropTypes.arrayOf(PropTypes.object),
    experimentIds: PropTypes.arrayOf(PropTypes.string),
    nextPageToken: PropTypes.string,
    compareExperiments: PropTypes.bool,
  };

  render() {
    const { experimentIds, experiments, compareExperiments } = this.props;
    const headerHeight = process.env.HIDE_HEADER === 'true' ? 0 : 60;
    const containerHeight = 'calc(100% - ' + headerHeight + 'px)';
    const hasExperiments = experimentIds?.length > 0;

    if (experimentIds === undefined) {
      const firstExp = getFirstActiveExperiment(experiments);
      if (firstExp) {
        return <Redirect to={Routes.getExperimentPageRoute(firstExp.experiment_id)} />;
      }
    }

    if (process.env.HIDE_EXPERIMENT_LIST === 'true') {
      return (
        <div style={{ height: containerHeight }}>
          {hasExperiments ? (
            <PageContainer>
              <React.Suspense fallback={<Skeleton />}>
                <ExperimentPage
                  experimentIds={experimentIds}
                  compareExperiments={compareExperiments}
                />
              </React.Suspense>
            </PageContainer>
          ) : (
            <NoExperimentView />
          )}
        </div>
      );
    }
    return (
      <div className='outer-container' style={{ height: containerHeight }}>
        <React.Suspense fallback={<Skeleton />}>
          <ExperimentListView activeExperimentIds={experimentIds || []} />
        </React.Suspense>
        <PageContainer>
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
        </PageContainer>
      </div>
    );
  }
}

const mapStateToProps = (state) => {
  const experiments = getExperiments(state);
  return { experiments };
};

export default connect(mapStateToProps)(HomeView);
