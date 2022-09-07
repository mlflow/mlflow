import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { Redirect } from 'react-router';
import { Skeleton, Spacer } from '@databricks/design-system';
import ExperimentListView from './ExperimentListView';
import ExperimentPage from './ExperimentPage';
import { getExperiments } from '../reducers/Reducers';
import { NoExperimentView } from './NoExperimentView';
import Utils from '../../common/utils/Utils';
import { PageContainer } from '../../common/components/PageContainer';
import Routes from '../routes';

// eslint-disable-next-line prefer-const
let ExperimentPageComponent = ExperimentPage;

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
                <ExperimentPageComponent
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
        <div>
          <Spacer />
          <ExperimentListView activeExperimentIds={experimentIds || []} experiments={experiments} />
        </div>
        <PageContainer>
          {hasExperiments ? (
            <ExperimentPageComponent
              experimentIds={experimentIds}
              compareExperiments={compareExperiments}
            />
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
